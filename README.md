# SciRAG-UQ

**Uncertainty-Aware Multi-Source Retrieval-Augmented Generation for Scientific Literature Synthesis**

> Submitted to **14th International Big Data & AI Conference (BDA 2026)**
> KK Birla Goa Campus, BITS Pilani, India | September 17–20, 2026

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Llama%203.1%2070B-orange)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is SciRAG-UQ?

SciRAG-UQ is a production-grade RAG system that knows **when to answer and when to abstain**.

Standard RAG generates answers regardless of evidence quality. SciRAG-UQ adds three complementary uncertainty signals — **retrieval confidence**, **generation entropy**, and **semantic consistency** — fused into a composite score that drives a cascaded abstention policy.

**Key results on BDA-Sci benchmark (500 questions):**
- Faithfulness: **0.847** (+6.8% vs. Self-RAG)
- Hallucination rate: **0.209** (−38.7% vs. Vanilla RAG)
- Abstention precision: **0.912**
- Expected Calibration Error: **0.043**

---

## Architecture

```
User Query
    │
    ▼
Hybrid Retriever (Dense HNSW + BM25 + MMR)
    │
    ├──► Retrieval Confidence ─────────────┐
    │                                      │
    ▼                                      ▼
LLM Generation (Groq / Llama 3.1)    Composite UQ Score
    │                                      │
    ├──► Generation Entropy ───────────────┤
    │                                      │
    └──► Semantic Consistency ─────────────┤
                                           │
                                           ▼
                                  Cascade Abstention Policy
                                           │
                                           ▼
                              Answer + Confidence Badge
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/kaushalrog/scirag-uq
cd scirag-uq
cp .env.example .env       # add your GROQ_API_KEY
make install

# 2. Ingest papers
python cli/main.py ingest --query "retrieval augmented generation" --max 30

# 3. Ask a question
python cli/main.py ask "What are the main limitations of RAG systems?"

# 4. Run API + UI
make run-api       # terminal 1 → FastAPI on :8000
make run-ui        # terminal 2 → Streamlit on :8501

# 5. Docker (all-in-one)
make docker-up
```

---

## Project Structure

```
scirag-uq/
├── config/               # Pydantic settings
├── src/
│   ├── ingestion/        # arXiv + Semantic Scholar + PDF extraction
│   ├── embeddings/       # SentenceTransformers + chunking strategies
│   ├── vectorstore/      # ChromaDB manager
│   ├── retrieval/        # Hybrid dense+sparse retrieval + MMR
│   ├── generation/       # Groq client + RAG chain + prompts
│   ├── uncertainty/      # UQ estimator + abstention policies
│   ├── evaluation/       # Metrics + benchmark + runner
│   └── api/              # FastAPI endpoints
├── frontend/             # Streamlit chat UI
├── cli/                  # Typer CLI
├── tests/                # Unit + integration tests
├── paper/                # BDA 2026 LaTeX submission
└── scripts/              # Utility scripts
```

---

## Uncertainty Signals

| Signal | Formula | Weight |
|--------|---------|--------|
| Retrieval Confidence $C_r$ | Mean hybrid score over top-k | 0.40 |
| Generation Entropy $C_g$ | $1/(1 + \bar{H})$ from logprobs | 0.35 |
| Semantic Consistency $C_s$ | Mean cosine sim across alternates | 0.25 |
| **Composite** | $0.40 C_r + 0.35 C_g + 0.25 C_s$ | — |

---

## Evaluation

```bash
# Build benchmark from ingested corpus
python -c "
from src.evaluation.benchmark import BenchmarkBuilder
from src.vectorstore.chroma_manager import ChromaManager
import json

chroma = ChromaManager()
metas = chroma._collection.get(include=['metadatas'])['metadatas']
builder = BenchmarkBuilder()
items = builder.build_from_documents(metas, output_path='data/benchmarks/benchmark.json')
print(f'Built {len(items)} benchmark items')
"

# Run evaluation
make eval
```

---

## Paper

Full paper in `paper/scirag_uq_BDA2026.tex` — IEEE conference format, ready for BDA 2026 submission.

Submission deadline: **June 30, 2026** | [https://www.bits-pilani.ac.in/bda-26/](https://www.bits-pilani.ac.in/bda-26/)

---

## Citation

```bibtex
@inproceedings{rao2026sciragUQ,
  title     = {SciRAG-UQ: Uncertainty-Aware Multi-Source Retrieval-Augmented
               Generation for Scientific Literature Synthesis},
  author    = {Rao, Kaushal G},
  booktitle = {Proc. 14th International Conference on Big Data Analytics (BDA)},
  year      = {2026},
  address   = {KK Birla Goa Campus, BITS Pilani, India}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
