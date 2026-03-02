import os
from typing import List, Dict

import requests


class JinaReranker:
    """
    Reranker using Jina Reranker API.

    Docs: https://jina.ai/reranker/
    Endpoint: POST https://api.jina.ai/v1/rerank

    Expects an environment variable JINA_API_KEY to be set.
    """

    def __init__(
        self,
        model: str = "jina-reranker-v3",
        api_key: str | None = None,
        endpoint: str = "https://api.jina.ai/v1/rerank",
        timeout: float = 15.0,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.api_key = api_key or os.getenv("JINA_API_KEY")

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Rerank retrieved contexts for a given query using Jina Reranker.

        Args:
            query: User query string.
            candidates: List of dicts with at least
                        {"text": str, "score": float, "metadata": dict}.

        Returns:
            New list of candidates sorted by reranker score (desc).
            Each candidate will have an extra key "rerank_score".
        """
        if not candidates:
            return candidates

        if not self.api_key:
            print("Warning: JINA_API_KEY not set; returning original order.")
            return candidates

        documents = [c["text"] for c in candidates]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            resp = requests.post(
                self.endpoint, json=payload, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()

            data = resp.json()
            # Jina returns an ordered list of results with "index" (0-based) and "relevance_score"
            results = data.get("results") or data.get("data") or []

            scores: Dict[int, float] = {}
            for item in results:
                idx = int(item.get("index", 0))
                score = float(item.get("relevance_score", item.get("score", 0.0)))
                scores[idx] = score

            indexed_candidates = list(enumerate(candidates))
            for idx, cand in indexed_candidates:
                cand["rerank_score"] = scores.get(idx, 0.0)

            sorted_candidates = sorted(
                (c for _, c in indexed_candidates),
                key=lambda c: (c.get("rerank_score", 0.0), c.get("score", 0.0)),
                reverse=True,
            )

            return sorted_candidates

        except Exception as e:
            print(f"Warning: Jina reranker failed, using original order. Error: {e}")
            return candidates
