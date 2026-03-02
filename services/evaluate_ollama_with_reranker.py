import sys
import time
from pathlib import Path

import ollama
import pandas as pd

from embedding_manager import Embedder
from qdrant_manager import QdrantManager
from llm_judge import LLMJudge
from reranker import JinaReranker


def load_questions_from_excel(
    excel_path: Path,
    question_col: str = "C",
    answer_col: str = "D",
    source_col: str | None = "E",
):
    """
    Load question/answer pairs from Excel.

    Columns are specified by letter (e.g. 'C' for questions, 'D' for answers).
    Optionally, a source column (e.g. 'B') can be provided to indicate
    the ground-truth PDF filename for retrieval evaluation.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)

    q_idx = ord(question_col.upper()) - ord("A")
    a_idx = ord(answer_col.upper()) - ord("A")

    questions = df.iloc[:, q_idx].dropna().tolist()
    answers = df.iloc[:, a_idx].dropna().tolist()

    sources: list[str] | None = None
    if source_col is not None:
        s_idx = ord(source_col.upper()) - ord("A")
        sources = df.iloc[:, s_idx].dropna().tolist()

    n = min(len(questions), len(answers), len(sources) if sources is not None else len(questions))
    questions = questions[:n]
    answers = answers[:n]
    if sources is not None:
        sources = sources[:n]

    return questions, answers, sources


class LocalRAGEvaluatorWithReranker:
    """
    Evaluate RAG using local components (Ollama + Qdrant) with a Jina-based reranker.

        - Retrieval: Embedder + QdrantManager
        - Reranking: Jina Reranker API (e.g. jina-reranker-v3)
        - Generation: Ollama (e.g. llama3.1:8b-instruct-q4_0)
        - Judge: LLMJudge (Ollama)
    """

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        embedding_model: str = "nomic-embed-text",
        top_k: int = 20,
        score_threshold: float = 0.5,
        generation_model: str = "llama3.1:8b-instruct-q4_0",
        judge_model: str = "llama3.1:8b-instruct-q4_0",
        reranker_model: str = "jina-reranker-v3",
        results_dir: Path | None = None,
    ):
        self.collection_name = collection_name
        # top_k: how many to retrieve from Qdrant before reranking (e.g. 20)
        self.top_k = top_k
        # final_top_k: how many top reranked chunks to pass to the generator (e.g. 5)
        self.final_top_k = 5
        self.score_threshold = score_threshold
        self.generation_model = generation_model

        self.embedder = Embedder(model_name=embedding_model)
        self.qdrant_manager = QdrantManager()
        self.judge = LLMJudge(model_name=judge_model)
        # Use Jina Reranker instead of local Qwen cross-encoder
        self.reranker = JinaReranker(model=reranker_model)

        if results_dir is None:
            results_dir = Path(__file__).resolve().parent / "results"
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results: list[dict] = []

    def retrieve_and_rerank_contexts(self, query: str) -> list[dict]:
        """Retrieve contexts from Qdrant and rerank them with Jina Reranker."""
        print(f"Embedding query: {query[:60]}...")
        query_embedding = self.embedder.embed_text(query)

        print(f"Searching Qdrant (top_k={self.top_k}, threshold={self.score_threshold})...")
        try:
            results = self.qdrant_manager.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=self.top_k,
                score_threshold=self.score_threshold,
            )

            hits = results.points if hasattr(results, "points") else results

            processed = [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                }
                for hit in hits
            ]
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            processed = []

        if not processed:
            print("No contexts retrieved from Qdrant.")
            return []

        print(
            f"Retrieved {len(processed)} contexts from Qdrant. "
            f"Reranking with Jina Reranker model '{self.reranker.model}'..."
        )
        reranked = self.reranker.rerank(query, processed)

        for i, r in enumerate(reranked, 1):
            base_score = r.get("score", 0.0)
            rerank_score = r.get("rerank_score", 0.0)
            print(
                f"  Context {i}: qdrant_score={base_score:.3f}, "
                f"rerank_score={rerank_score:.3f}, "
                f"source={r['metadata'].get('source', 'unknown')}"
            )

        return reranked

    def generate_with_rag(self, query: str, contexts: list[str]) -> str:
        """Generate an answer using Ollama with (reranked) contexts."""
        if not contexts:
            print("No contexts found, generating without RAG (local Ollama)...")
            prompt = (
                "You are a helpful AI assistant. Answer the user's question directly.\n\n"
                f"User Question: {query}\n\nAnswer:"
            )
        else:
            contexts_text = "\n\n".join(
                [f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)]
            )
            prompt = (
                "You are a helpful AI assistant. Answer the user's question based on the "
                "provided context documents.\n\n"
                f"Context Documents:\n{contexts_text}\n\n"
                f"User Question: {query}\n\n"
                "Instructions:\n"
                "- Answer based primarily on the information in the context documents.\n"
                "- If the context doesn't contain enough information, say so clearly.\n"
                "- Be concise and accurate.\n"
                "- If relevant, mention which context(s) you used.\n\n"
                "Answer:"
            )

        print("Generating answer with local Ollama model...")
        response = ollama.chat(
            model=self.generation_model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.get("message", {}).get("content", "").strip()

    def process_one(self, question: str, true_answer: str, true_source: str | None = None) -> dict:
        """Process a single QA pair: retrieve, rerank, generate, judge, check retrieval."""
        from datetime import datetime

        start = time.time()

        reranked = self.retrieve_and_rerank_contexts(question)

        # Take top-N reranked chunks for generation
        top_chunks = reranked[: self.final_top_k]
        contexts = [r["text"] for r in top_chunks]
        answer = self.generate_with_rag(question, contexts)
        judgment = self.judge.evaluate(answer, true_answer, query=question)

        elapsed = time.time() - start

        # Retrieval metrics on the final top-k chunks
        retrieved_sources = [r["metadata"].get("source") for r in top_chunks]

        # Chunking configuration (assumed identical for all chunks in collection)
        config_chunk_size = None
        config_chunk_overlap = None
        if top_chunks:
            cfg_meta = top_chunks[0].get("metadata", {})
            config_chunk_size = cfg_meta.get("config_chunk_size")
            config_chunk_overlap = cfg_meta.get("config_chunk_overlap")

        retrieval_correct = None
        retrieval_precision = None
        if true_source and retrieved_sources:
            correct_count = sum(1 for src in retrieved_sources if src == true_source)
            retrieval_correct = correct_count > 0
            retrieval_precision = correct_count / len(retrieved_sources)

        result = {
            "timestamp": datetime.now().isoformat(),
            "query": question,
            "response": answer,
            "true_response": true_answer,
            "contexts": contexts,
            "num_contexts": len(contexts),
            "judgment": judgment,
            "elapsed_time": elapsed,
            "retrieval_ground_truth_source": true_source,
            "retrieved_sources": retrieved_sources,
            "retrieval_correct": retrieval_correct,
            "retrieval_precision": retrieval_precision,
            "config_chunk_size": config_chunk_size,
            "config_chunk_overlap": config_chunk_overlap,
        }

        self.results.append(result)
        return result

    def save_text_summary(self):
        """Save a simple text summary for answer and retrieval accuracy."""
        from datetime import datetime

        judged = [r for r in self.results if r["judgment"] is not None]
        correct = sum(1 for r in judged if r["judgment"].get("judgment"))

        accuracy = (correct / len(judged) * 100) if judged else 0.0

        retrieval_labeled = [
            r for r in self.results if r.get("retrieval_precision") is not None
        ]
        # Binary "any correct" accuracy over final top-k
        retrieval_correct = sum(
            1 for r in retrieval_labeled if r.get("retrieval_correct")
        )
        retrieval_accuracy = (
            retrieval_correct / len(retrieval_labeled) * 100 if retrieval_labeled else 0.0
        )
        # Average precision over final top-k
        avg_precision = (
            sum(r["retrieval_precision"] for r in retrieval_labeled) / len(retrieval_labeled)
            if retrieval_labeled
            else 0.0
        )

        # Chunking config (take from first result that has it, if any)
        cfg_size = None
        cfg_overlap = None
        for r in self.results:
            if r.get("config_chunk_size") is not None:
                cfg_size = r.get("config_chunk_size")
                cfg_overlap = r.get("config_chunk_overlap")
                break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"evaluation_result_reranker_{timestamp}.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            if cfg_size is not None:
                f.write(
                    f"Chunking config: chunk_size={cfg_size}, "
                    f"chunk_overlap={cfg_overlap}\n"
                )
            f.write(
                f"Right answer (with reranker): {correct}/{len(judged)}, "
                f"accuracy: {accuracy:.0f}%\n"
            )
            f.write(
                f"Retrieval correct (any chunk from true PDF in final top-{self.final_top_k}): "
                f"{retrieval_correct}/{len(retrieval_labeled)}, "
                f"accuracy: {retrieval_accuracy:.0f}%\n"
            )
            f.write(
                f"Average retrieval precision over final top-{self.final_top_k}: "
                f"{avg_precision:.2f}"
            )

        print(
            f"\nRight answer (with reranker): {correct}/{len(judged)}, "
            f"accuracy: {accuracy:.0f}%"
        )
        print(
            f"Retrieval correct (any chunk from true PDF in final top-{self.final_top_k}): "
            f"{retrieval_correct}/{len(retrieval_labeled)}, "
            f"accuracy: {retrieval_accuracy:.0f}%"
        )
        print(
            f"Average retrieval precision over final top-{self.final_top_k}: "
            f"{avg_precision:.2f}"
        )
        print(f"Simple accuracy summary (with reranker) saved to: {summary_file}")

    def save_retrieval_accuracy_excel(self):
        """
        Save per-question retrieval metrics to an Excel file:
        question_retrieval_accuracy.xlsx
        """
        if not self.results:
            return

        rows = []
        for r in self.results:
            retrieval_correct = r.get("retrieval_correct")
            answer_correct = (
                r.get("judgment", {}).get("judgment") if r.get("judgment") else None
            )

            # Normalize booleans to explicit English strings for Excel
            def bool_to_str(val):
                if val is True:
                    return "True"
                if val is False:
                    return "False"
                return ""

            rows.append(
                {
                    "timestamp": r.get("timestamp"),
                    "question": r.get("query"),
                    "ground_truth_source": r.get("retrieval_ground_truth_source"),
                    "retrieved_sources": ", ".join(r.get("retrieved_sources") or []),
                    "retrieval_correct": bool_to_str(retrieval_correct),
                    "retrieval_precision": r.get("retrieval_precision"),
                    "answer_correct": bool_to_str(answer_correct),
                    "answer_confidence": r.get("judgment", {}).get("confidence")
                    if r.get("judgment")
                    else None,
                }
            )

        df = pd.DataFrame(rows)
        out_path = self.results_dir / "question_retrieval_accuracy.xlsx"

        from datetime import datetime
        import os

        sheet_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        if out_path.exists():
            # Append as a new sheet in the existing workbook
            with pd.ExcelWriter(
                out_path, engine="openpyxl", mode="a", if_sheet_exists="new"
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Create new workbook with a single sheet
            with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Per-question retrieval accuracy saved to: {out_path} (sheet: {sheet_name})")


def run_excel_evaluation_with_reranker():
    """
    Evaluate all question–answer pairs from RAG Documents.xlsx
    using local RAG + Qwen3-Reranker + Ollama judge.
    """
    print("\n" + "=" * 80)
    print("RAG EXCEL EVALUATION (LOCAL OLLAMA + QDRANT + RERANKER)")
    print("=" * 80)

    project_root = Path(__file__).resolve().parent.parent
    excel_path = project_root / "RAG Documents.xlsx"

    print(f"\nLoading questions from: {excel_path}")
    questions, answers, sources = load_questions_from_excel(
        excel_path, question_col="C", answer_col="D", source_col="E"
    )
    total = len(questions)
    print(f"Loaded {total} question–answer pairs.\n")

    evaluator = LocalRAGEvaluatorWithReranker(collection_name="pdf_documents")

    start_time = time.time()

    for i, (q, true_ans, src) in enumerate(zip(questions, answers, sources), start=1):
        print("\n" + "-" * 80)
        print(f"Question {i}/{total}")
        print("-" * 80)
        print(f"Q: {q}")
        print(f"Ground-truth source PDF: {src}")

        result = evaluator.process_one(q, true_ans, true_source=src)

        print("\nModel answer:")
        print(result["response"])

        j = result["judgment"]
        status = "CORRECT" if j.get("judgment") else "INCORRECT"
        print(
            f"\nJudge: {status} (confidence: {j.get('confidence', 0.0):.2f})"
        )
        print(f"Explanation: {j.get('explanation', '')}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE (WITH RERANKER)")
    print("=" * 80)
    print(f"Total time: {elapsed/60:.1f} minutes")

    evaluator.save_text_summary()
    evaluator.save_retrieval_accuracy_excel()


if __name__ == "__main__":
    try:
        run_excel_evaluation_with_reranker()
        sys.exit(0)
    except Exception as e:
        print(f"\nError during evaluation with reranker: {e}")
        sys.exit(1)

