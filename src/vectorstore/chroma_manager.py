"""
src/vectorstore/chroma_manager.py
-----------------------------------
ChromaDB collection manager.
Handles upsert, query, and deletion with full metadata support.
Designed around TextChunk objects from the embeddings module.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from src.embeddings.chunker import TextChunk

logger = logging.getLogger(__name__)


class RetrievedChunk:
    """A chunk returned from a vector search, with its distance/score."""

    __slots__ = ("chunk_id", "text", "metadata", "distance", "score")

    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: dict,
        distance: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.distance = distance
        self.score = max(0.0, 1.0 - distance)   # cosine: distance ∈ [0,2], score ∈ [-1,1]

    def __repr__(self) -> str:
        return f"<RetrievedChunk id={self.chunk_id} score={self.score:.3f}>"


class ChromaManager:
    """
    Thin wrapper around ChromaDB for SciRAG-UQ.

    Parameters
    ----------
    persist_dir  : Path where ChromaDB persists data
    collection   : Collection name
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection: str = "scirag_papers",
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready — collection='%s', %d existing docs",
            collection,
            self._collection.count(),
        )

    # ── Write ────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self, chunks: list[TextChunk], embeddings: np.ndarray
    ) -> int:
        """
        Upsert chunks with pre-computed embeddings.

        Returns the number of new/updated docs.
        """
        if not chunks:
            return 0
        assert len(chunks) == len(embeddings), "chunks/embeddings length mismatch"

        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [self._build_meta(c) for c in chunks]
        embs = embeddings.tolist()

        self._collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )
        logger.info("Upserted %d chunks into ChromaDB", len(chunks))
        return len(chunks)

    def delete_by_source(self, source_id: str) -> None:
        """Remove all chunks belonging to a source document."""
        self._collection.delete(where={"source_id": source_id})
        logger.info("Deleted chunks for source_id='%s'", source_id)

    # ── Read ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Dense vector search.

        Parameters
        ----------
        query_embedding : Shape (dim,) query vector
        top_k           : Number of results
        where           : Optional ChromaDB metadata filter

        Returns
        -------
        List of RetrievedChunk sorted by descending score
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        chunks: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(RetrievedChunk(cid, doc, meta or {}, dist))

        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Return unique source_ids in the collection."""
        if self.count() == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        metas = results.get("metadatas") or []
        return list({m.get("source_id", "") for m in metas if m})

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_meta(chunk: TextChunk) -> dict:
        base = {
            "source_id": chunk.source_id,
            "title": chunk.title[:200],
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "strategy": chunk.strategy,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "word_count": chunk.word_count,
        }
        base.update(chunk.metadata)
        # ChromaDB requires all values to be str/int/float/bool
        return {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in base.items()}
