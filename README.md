# AI-RAG-System – Router Agent + RAG Generator + Reranker + LLM Judge

A complete **Retrieval-Augmented Generation (RAG)** system implementation with intelligent routing, document retrieval, and automated evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results & Accuracy Reporting](#results--accuracy-reporting)

---

## Overview

This project implements a production-ready RAG system with four main components:

1. **Router Agent** (Gemini) – Intelligently decides whether a query requires RAG retrieval or can be answered directly
2. **RAG Generator** (Gemini + Qdrant) – Retrieves relevant document chunks and generates context-aware answers
3. **Reranker** (Jina Reranker API) – Re-scores retrieved chunks to prioritize the most relevant ones
4. **LLM Judge** (Ollama) – Evaluates answer quality by comparing generated responses to ground-truth answers

### Key Features

- ✅ **Intelligent Routing** – Automatically determines when to use RAG vs. direct LLM responses
- ✅ **Document Retrieval** – Semantic search over PDF documents using Qdrant vector database
- ✅ **Reranking Layer** – Optional Jina Reranker (e.g. `jina-reranker-v3`) re-scores the top‑k retrieved chunks and keeps only the best top‑N for the generator
- ✅ **Automated Evaluation** – LLM-as-judge assessment with accuracy metrics
- ✅ **Retrieval Accuracy Metrics** – Per-question retrieval precision and correctness exported to `question_retrieval_accuracy.xlsx`
- ✅ **API Key Rotation** – Handles quota limits with automatic key/model switching
- ✅ **Checkpoint & Resume** – Saves progress and can resume interrupted evaluations
- ✅ **Comprehensive Reporting** – CSV statistics, Excel retrieval analysis, and TXT accuracy summaries

---

## Architecture

```
User Query
    ↓
Router Agent (Gemini)
    ├─→ Decision: "rag"
    │       ↓
    │   Retriever (Qdrant, top‑k)
    │       ↓
    │   Reranker (Jina Reranker API, re-score → top‑N)
    │       ↓
    │   RAG Generator (Gemini + contexts)
    └─→ Decision: "direct" → Direct LLM (Gemini)
            ↓
        Generated Answer
            ↓
        LLM Judge (Ollama) ← Ground Truth Answer
            ↓
        Evaluation Results (Accuracy, Confidence, Explanation, Retrieval Metrics)
```

### Component Details

- **Router Agent** (`services/router_agent.py`)
  - Uses Gemini to analyze query intent
  - Returns routing decision (`"rag"` or `"direct"`) with reasoning

- **RAG Generator** (`services/rag_generator.py`)
  - Embeds queries using Ollama (`nomic-embed-text`)
  - Retrieves top‑k relevant chunks from Qdrant (e.g. top‑20)
  - Optionally calls **Jina Reranker** to re-rank and keep only the best top‑N chunks (e.g. top‑5)
  - Generates answers using Gemini with reranked contexts

- **Reranker** (`services/reranker.py`, used in `rag_generator.py` and `evaluate_ollama_with_reranker.py`)
  - Thin client around [Jina Reranker API](https://jina.ai/reranker/)
  - Default model: `jina-reranker-v3`
  - Input: user query + list of candidate chunks from Qdrant
  - Output: the same list of chunks with an extra `rerank_score`, sorted by relevance

- **LLM Judge** (`services/llm_judge.py`)
  - Uses Ollama (e.g., `llama3.1:8b-instruct`) for semantic comparison
  - Evaluates if generated answer matches ground truth
  - Returns judgment (True/False), confidence score, and explanation

---

## Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: Installed and running locally
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b-instruct-q4_0  # Recommended for judge
  ```
- **Qdrant**: Running on `http://localhost:6333`
  ```bash
  docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
  ```
- **Google Gemini API Key**: Free tier supports 20 requests/day per model
- **Jina Reranker API Key** (optional, recommended for best retrieval accuracy)

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd RAG_Tutorial
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install qdrant-client ollama numpy PyPDF2 python-dotenv pydantic-ai google-generativeai pandas openpyxl
   ```

4. **Configure environment variables**:
   
   Create `.env` file in project root:
   ```env
   # Main Gemini API key (for interactive chat)
   GEMINI_API_KEY=your_main_api_key_here
   PDF_DOCUMENTS=pdf_documents
   GEMINI_MODEL_NAME=gemini-2.5-flash-lite
   # Jina Reranker (used in online RAG + offline evaluation)
   JINA_API_KEY=your_jina_reranker_key_here
   
   # Multiple API keys for evaluation (optional)
   GEMINI_API_KEY_1=your_key_1
   GEMINI_MODELS_1=gemini-2.5-flash-lite,gemini-2.5-flash
   GEMINI_API_KEY_2=your_key_2
   GEMINI_MODELS_2=gemini-2.5-pro
   # ... add more as needed
   ```

---

## Quick Start

### 1. Populate Qdrant with Documents

Place PDF files in `data/` directory, then:

```bash
cd services
python populate_qdrant.py
```

Choose option **1** or **2** to populate the collection.

### 2. Run Interactive Chat

```bash
cd services
python main.py
```

Type your questions and get answers with automatic RAG routing!

### 3. Evaluate on Question-Answer Pairs

**With Ollama (local, no quota limits) – baseline (without reranker)**:
```bash
cd services
python evaluate_ollama.py
```

**With reranker + Ollama judge (local)**:
```bash
cd services
python evaluate_ollama_with_reranker.py
```

This evaluation uses:
- Qdrant retrieval (e.g. top‑20),
- Jina Reranker (e.g. top‑5 for the generator),
- Ollama as judge,
and writes:
- text summary with answer accuracy + retrieval metrics
- per‑question retrieval analysis into `results/question_retrieval_accuracy.xlsx`.

## Usage

### Interactive Chat (`main.py`)

Simple chat interface with automatic routing:

```bash
python services/main.py
```

**Commands**:
- Type your question → Get answer (with RAG or direct)
- Type `stats` → See current session statistics
- Type `quit` / `exit` → End chat and save results

**Features**:
- Router decides RAG vs direct automatically
- Optional LLM judge evaluation (paste reference answer)
- Statistics saved on exit

### Ollama Evaluation (`evaluate_ollama.py`)

Evaluates all 60 question-answer pairs using **local Ollama** (no API limits):

```bash
python services/evaluate_ollama.py
```

**What it does**:
- Loads questions from `RAG Documents.xlsx` (columns C & D)
- Uses Ollama for generation + judge (no Gemini)
- Saves TXT summary: `Right answer: X/60, accuracy: Y%`

## Results & Accuracy Reporting

All results are saved in `services/results/` directory.

### Evaluation Results

#### Ollama Model Performance

**Evaluation Date**: February 13, 2026  
**Model**: Local Ollama (llama3.1:8b-instruct-q4_0)  
**Dataset**: 60 question-answer pairs from `RAG Documents.xlsx`

**Baseline (WITHOUT reranker, local RAG + Ollama judge)**  
Chunking: `chunk_size=1000`, `overlap=200`

```
Right answer: 43/60, accuracy: 72%
```

- **Correct Answers**: 43 out of 60
- **Accuracy**: 72%
- **Evaluation Method**: LLM-as-judge (Ollama) semantic comparison

---

#### Effect of Jina Reranker on Answer Accuracy

All evaluations below use:
- Retriever: Qdrant
- Reranker: Jina Reranker API (`jina-reranker-v3`)
- Judge: Ollama (LLM-as-judge)

**WITH RERANKER**

- **chunk size 1000, overlap 200**

  ```
  Right answer: 44/60, accuracy: 73%
  ```

- **chunk size 512, overlap 100**

  ```
  Right answer: 41/60, accuracy: 68%
  ```

- **chunk size 1500, overlap 300**

  ```
  Right answer: 53/60, accuracy: 88%
  ```

Summary:
- Adding the reranker on top of the original chunking (1000 / 200) gives a small accuracy gain (72% → 73%).
- The best configuration so far is **larger chunks with moderate overlap (1500 / 300)**, which significantly improves answer accuracy to **88% (53/60)**.
- These experiments are backed by per-question retrieval precision exported to `question_retrieval_accuracy.xlsx`.

### Output Files

#### 1. **Accuracy Summary** (TXT)
- **File**: `evaluation_result_YYYYMMDD_HHMMSS.txt`
- **Format**: `Right answer: X/Y, accuracy: Z%`
- **Examples**:
  ```
  Right answer: 43/60, accuracy: 72%
  ```

#### 2. **Statistics CSV**
- **File**: `stats_YYYYMMDD_HHMMSS.csv`
- **Contains**:
  - Total queries processed
  - RAG vs Direct routing counts and percentages
  - Judgment statistics (total judged, correct, incorrect, accuracy)
  - Performance metrics (avg time, avg contexts retrieved)

#### 3. **Detailed Results CSV**
- **File**: `results_YYYYMMDD_HHMMSS.csv`
- **Contains** (one row per question):
  - Timestamp
  - Query text
  - Routing decision & reasoning
  - Generated response
  - Number of contexts used
  - True response (if provided)
  - Judge decision (True/False)
  - Judge confidence score
  - Judge explanation
  - Elapsed time

## Project Structure

```
AI-RAG-System/
├── services/
│   ├── main.py                 # Interactive chat with RAG system
│   ├── evaluate_ollama.py      # Ollama-based evaluation (60 Q&A pairs)
│   ├── router_agent.py         # Gemini router (RAG vs Direct)
│   ├── rag_generator.py        # RAG pipeline (Gemini + Qdrant + reranker)
│   ├── reranker.py             # Jina Reranker client
│   ├── llm_judge.py            # Ollama-based judge
│   ├── embedding_manager.py    # Ollama embeddings
│   ├── qdrant_manager.py       # Qdrant operations
│   ├── preprocessing.py        # PDF chunking
│   ├── populate_qdrant.py      # Populate vector DB + chunking experiments
│   └── results/                # Output directory
│       ├── *.csv               # Statistics & detailed results
│       ├── *.txt               # Accuracy summaries
│       ├── question_retrieval_accuracy.xlsx  # Retrieval metrics per question
│       └── *.json              # Checkpoints
├── data/                       # PDF documents
├── qdrant_data/                # Qdrant storage
├── .env                        # Environment variables (API keys)
├── RAG Documents.xlsx          # Question-answer pairs (with source PDF column)
└── README.md
```

## Acknowledgments

- **Qdrant** – Vector database for semantic search
- **Ollama** – Local LLM inference
- **Google Gemini** – Cloud LLM API
- **Pydantic AI** – LLM framework

---

