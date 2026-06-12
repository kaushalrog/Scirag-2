"""
src/api/app.py
---------------
FastAPI REST API for SciRAG-UQ.

Endpoints
---------
POST /query          — Ask a question, get answer + UQ scores
POST /ingest         — Ingest papers by arXiv query
POST /ingest/pdf     — Upload and ingest a local PDF
GET  /sources        — List ingested sources
GET  /health         — Health check
POST /evaluate       — Run evaluation on benchmark
GET  /metrics        — Last evaluation metrics
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from src.embeddings.chunker import RecursiveChunker, SemanticChunker
from src.embeddings.embedder import Embedder
from src.generation.groq_client import GroqClient
from src.generation.rag_chain import RAGChain
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retriever import HybridRetriever
from src.uncertainty.abstention import CascadePolicy
from src.uncertainty.estimator import UncertaintyEstimator
from src.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

# ── Global component singletons ───────────────────────────────────────────────
_components: dict = {}


def build_chain() -> RAGChain:
    settings = get_settings()
    embedder = Embedder(model_name=settings.embedding_model)
    chroma = ChromaManager(
        persist_dir=settings.chroma_persist_dir,
        collection=settings.chroma_collection,
    )
    retriever = HybridRetriever(
        chroma=chroma,
        embedder=embedder,
        dense_weight=settings.dense_weight,
        mmr_lambda=settings.mmr_lambda,
    )
    llm = GroqClient(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    uq = UncertaintyEstimator(embedder=embedder)
    policy = CascadePolicy()
    return RAGChain(
        retriever=retriever,
        llm=llm,
        uncertainty=uq,
        abstention=policy,
        top_k=settings.top_k,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting SciRAG-UQ API…")
    _components["chain"] = build_chain()
    _components["pipeline"] = IngestionPipeline(
        download_dir=settings.data_raw_dir,
        processed_dir=settings.data_processed_dir,
        s2_api_key=settings.s2_api_key,
    )
    _components["embedder"] = _components["chain"].retriever.embedder
    _components["chroma"] = _components["chain"].retriever.chroma
    _components["chunker"] = RecursiveChunker(chunk_size=512, overlap=64)
    _components["last_metrics"] = {}
    logger.info("API ready — %d docs in corpus", _components["chroma"].count())
    yield
    logger.info("Shutting down SciRAG-UQ API")


app = FastAPI(
    title="SciRAG-UQ",
    description="Uncertainty-Aware Scientific Literature RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    use_cot: bool = False
    stream: bool = False

class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    confidence_label: str
    abstained: bool
    abstention_reason: str
    sources: list[dict]
    uncertainty_breakdown: dict

class IngestRequest(BaseModel):
    query: str = Field(..., min_length=3)
    max_papers: int = Field(20, ge=1, le=100)
    download_pdfs: bool = False
    enrich_s2: bool = True

class IngestResponse(BaseModel):
    ingested: int
    failed: int
    source_ids: list[str]

class HealthResponse(BaseModel):
    status: str
    corpus_size: int
    model: str
    version: str = "1.0.0"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    settings = get_settings()
    return HealthResponse(
        status="ok",
        corpus_size=_components["chroma"].count(),
        model=settings.llm_model,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    chain: RAGChain = _components["chain"]
    chain.use_cot = req.use_cot

    if req.stream:
        def token_stream():
            for token in chain.stream_query(req.question):
                yield token
        return StreamingResponse(token_stream(), media_type="text/plain")

    response = chain.query(req.question)
    from src.uncertainty.estimator import UncertaintyEstimator
    label = UncertaintyEstimator.confidence_to_label(response.confidence)

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        confidence=response.confidence,
        confidence_label=label,
        abstained=response.abstained,
        abstention_reason=response.abstention_reason,
        sources=response.sources,
        uncertainty_breakdown=response.uncertainty_breakdown,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    pipeline: IngestionPipeline = _components["pipeline"]
    embedder: Embedder = _components["embedder"]
    chroma: ChromaManager = _components["chroma"]
    chunker: RecursiveChunker = _components["chunker"]

    docs = pipeline.run(
        query=req.query,
        max_papers=req.max_papers,
        download_pdfs=req.download_pdfs,
        enrich_s2=req.enrich_s2,
    )

    ingested, failed = 0, 0
    source_ids = []
    for doc in docs:
        try:
            chunks = chunker.split(
                text=doc.full_text or doc.abstract,
                source_id=doc.arxiv_id,
                title=doc.title,
                extra_metadata=doc.to_metadata(),
            )
            if not chunks:
                continue
            embs = embedder.embed([c.text for c in chunks])
            chroma.upsert_chunks(chunks, embs)
            source_ids.append(doc.arxiv_id)
            ingested += 1
        except Exception as exc:
            logger.error("Failed to upsert %s: %s", doc.arxiv_id, exc)
            failed += 1

    return IngestResponse(ingested=ingested, failed=failed, source_ids=source_ids)


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    settings = get_settings()
    save_path = os.path.join(settings.data_raw_dir, file.filename)
    with open(save_path, "wb") as fh:
        fh.write(await file.read())

    pipeline: IngestionPipeline = _components["pipeline"]
    embedder: Embedder = _components["embedder"]
    chroma: ChromaManager = _components["chroma"]
    chunker: RecursiveChunker = _components["chunker"]

    doc = pipeline.ingest_local_pdf(save_path)
    chunks = chunker.split(
        text=doc.full_text,
        source_id=doc.arxiv_id,
        title=doc.title,
        extra_metadata=doc.to_metadata(),
    )
    embs = embedder.embed([c.text for c in chunks])
    chroma.upsert_chunks(chunks, embs)

    return {"source_id": doc.arxiv_id, "chunks": len(chunks), "words": doc.word_count}


@app.get("/sources")
async def list_sources():
    chroma: ChromaManager = _components["chroma"]
    return {"sources": chroma.list_sources(), "total": chroma.count()}


@app.get("/metrics")
async def get_metrics():
    return _components.get("last_metrics", {})
