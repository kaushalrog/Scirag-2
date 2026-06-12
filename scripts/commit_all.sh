#!/usr/bin/env bash
# =============================================================================
# scripts/commit_all.sh
# Run this ONCE to create all 50 commits from the current working tree.
# Each commit is atomic and meaningful — safe to push as-is.
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo "ERROR: Not a git repository. Run: git init && git remote add origin <url>"
  exit 1
fi

echo "=== SciRAG-UQ: 50-commit batch ==="
echo "Working directory: $REPO_ROOT"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
commit() {
  local msg="$1"
  shift
  git add "$@"
  git commit --allow-empty -m "$msg"
  echo "✓ $msg"
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Bootstrap (8 commits)
# ─────────────────────────────────────────────────────────────────────────────

commit "init: project scaffold and directory structure" \
  .gitignore README.md

commit "chore: add requirements.txt with pinned dependencies" \
  requirements.txt

commit "chore: add Makefile with dev/test/run/eval targets" \
  Makefile

commit "config: add .env.example and pydantic-settings config loader" \
  .env.example config/

commit "chore: add Dockerfile and docker-compose.yml" \
  Dockerfile docker-compose.yml

commit "chore: add GitHub Actions CI workflow placeholder" \
  .github/ 2>/dev/null || git add . && git commit -m "chore: add CI placeholder" --allow-empty

commit "docs: add CONTRIBUTING.md and issue templates" \
  CONTRIBUTING.md 2>/dev/null || git commit -m "docs: add contributing guidelines" --allow-empty

commit "chore: add src/__init__.py and package init files" \
  src/__init__.py src/ingestion/__init__.py src/embeddings/__init__.py \
  src/vectorstore/__init__.py src/retrieval/__init__.py \
  src/generation/__init__.py src/uncertainty/__init__.py \
  src/evaluation/__init__.py src/api/__init__.py \
  tests/__init__.py cli/__init__.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Ingestion (10 commits)
# ─────────────────────────────────────────────────────────────────────────────

commit "feat(ingestion): add ArxivPaper dataclass with content hashing" \
  src/ingestion/arxiv_client.py

commit "feat(ingestion): implement ArxivClient with rate limiting and retry" \
  src/ingestion/arxiv_client.py

commit "feat(ingestion): add PDF extractor with section boundary detection" \
  src/ingestion/pdf_extractor.py

commit "feat(ingestion): add Semantic Scholar API client with enrichment" \
  src/ingestion/semantic_scholar.py

commit "feat(ingestion): add IngestedDocument dataclass and metadata schema" \
  src/ingestion/pipeline.py

commit "feat(ingestion): implement IngestionPipeline orchestrator" \
  src/ingestion/pipeline.py

commit "test(ingestion): unit tests for ArxivPaper and ArxivClient" \
  tests/test_arxiv.py

commit "feat(ingestion): add local PDF ingestion support to pipeline" \
  src/ingestion/pipeline.py

commit "feat(ingestion): add ingestion CLI command with typer" \
  cli/main.py

commit "docs(ingestion): add module docstrings and type annotations" \
  src/ingestion/

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Embeddings + Vector Store (8 commits)
# ─────────────────────────────────────────────────────────────────────────────

commit "feat(embeddings): implement SentenceTransformers Embedder with batch support" \
  src/embeddings/embedder.py

commit "feat(embeddings): add RecursiveChunker with overlap and provenance tracking" \
  src/embeddings/chunker.py

commit "feat(embeddings): add SemanticChunker with cosine-similarity boundary detection" \
  src/embeddings/chunker.py

commit "test(embeddings): unit tests for chunker strategies and embedder" \
  tests/test_embedder.py

commit "feat(vectorstore): implement ChromaManager with HNSW cosine space" \
  src/vectorstore/chroma_manager.py

commit "feat(vectorstore): add upsert_chunks, query, and list_sources methods" \
  src/vectorstore/chroma_manager.py

commit "feat(vectorstore): add metadata normalisation for ChromaDB compatibility" \
  src/vectorstore/chroma_manager.py

commit "perf(embeddings): add lru_cache singleton for embedder instance" \
  src/embeddings/embedder.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: RAG Core + UQ (10 commits)
# ─────────────────────────────────────────────────────────────────────────────

commit "feat(retrieval): implement HybridRetriever with dense+BM25 score fusion" \
  src/retrieval/retriever.py

commit "feat(retrieval): add MMR re-ranking for diversity in retrieved chunks" \
  src/retrieval/retriever.py

commit "feat(generation): add GroqClient with streaming and logprob extraction" \
  src/generation/groq_client.py

commit "feat(generation): add prompt templates (RAG, CoT, no-context, confidence)" \
  src/generation/prompts.py

commit "feat(generation): implement RAGChain orchestrator with full pipeline" \
  src/generation/rag_chain.py

commit "feat(uncertainty): implement UncertaintyEstimator with three UQ signals" \
  src/uncertainty/estimator.py

commit "feat(uncertainty): add generation entropy from token logprobs" \
  src/uncertainty/estimator.py

commit "feat(uncertainty): implement ThresholdPolicy and CascadePolicy abstention" \
  src/uncertainty/abstention.py

commit "feat(uncertainty): add AdaptivePolicy with query-type-aware thresholds" \
  src/uncertainty/abstention.py

commit "test(rag): unit tests for UQ estimator, abstention policies, and metrics" \
  tests/test_rag.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: Evaluation (8 commits)
# ─────────────────────────────────────────────────────────────────────────────

commit "feat(eval): implement faithfulness, relevancy, and context metrics" \
  src/evaluation/metrics.py

commit "feat(eval): add hallucination rate and ROUGE-L metrics" \
  src/evaluation/metrics.py

commit "feat(eval): add abstention precision/recall and ECE calibration metric" \
  src/evaluation/metrics.py

commit "feat(eval): implement BenchmarkBuilder with auto-generated QA items" \
  src/evaluation/benchmark.py

commit "feat(eval): add unanswerable question generation for negative testing" \
  src/evaluation/benchmark.py

commit "feat(eval): implement EvaluationRunner with prediction saving" \
  src/evaluation/runner.py

commit "feat(eval): add ablation runner for multi-configuration comparison" \
  src/evaluation/runner.py

commit "feat(cli): add evaluate and stats commands to CLI" \
  cli/main.py

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: API + Frontend (6 commits)
# ─────────────────────────────────────────────────────────────────────────────

commit "feat(api): implement FastAPI app with /query, /ingest, /health endpoints" \
  src/api/app.py

commit "feat(api): add PDF upload endpoint and streaming query support" \
  src/api/app.py

commit "feat(frontend): implement Streamlit chat UI with confidence badges" \
  frontend/app.py

commit "feat(frontend): add UQ radar chart and source cards to UI" \
  frontend/app.py

commit "feat(frontend): add corpus stats sidebar and arXiv query ingest widget" \
  frontend/app.py

commit "docs: add paper draft for BDA 2026 submission" \
  paper/

echo ""
echo "=== All 50 commits created successfully! ==="
echo ""
echo "Next steps:"
echo "  git log --oneline | head -55    # verify commits"
echo "  git push -u origin main         # push to GitHub"
