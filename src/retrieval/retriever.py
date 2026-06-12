"""
src/retrieval/retriever.py
---------------------------
Hybrid retriever combining dense (ChromaDB) and sparse (BM25) search,
with Maximal Marginal Relevance (MMR) re-ranking for diversity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings.embedder import Embedder
from src.vectorstore.chroma_manager import ChromaManager, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Merged result from hybrid retrieval."""

    chunk_id: str
    text: str
    metadata: dict
    dense_score: float      # cosine similarity [0,1]
    sparse_score: float     # normalised BM25 score [0,1]
    hybrid_score: float     # weighted combination
    mmr_score: float        # after MMR re-ranking
    rank: int               # final rank (0 = best)

    @property
    def source_id(self) -> str:
        return self.metadata.get("source_id", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")


class HybridRetriever:
    """
    Hybrid dense + sparse retrieval with MMR re-ranking.

    Architecture
    ------------
    1. Dense retrieval: ChromaDB HNSW cosine search
    2. Sparse retrieval: BM25 over the same candidate pool
    3. Score fusion: ``alpha * dense + (1-alpha) * sparse``
    4. MMR re-ranking: balances relevance vs. diversity
    """

    def __init__(
        self,
        chroma: ChromaManager,
        embedder: Embedder,
        dense_weight: float = 0.7,
        mmr_lambda: float = 0.5,
        candidate_multiplier: int = 3,
    ) -> None:
        self.chroma = chroma
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.mmr_lambda = mmr_lambda
        self.candidate_multiplier = candidate_multiplier

    # ── Public ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_mmr: bool = True,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """
        Full hybrid retrieval pipeline.

        Parameters
        ----------
        query   : Natural language query string
        top_k   : Number of final results to return
        use_mmr : Apply MMR re-ranking (recommended for diverse results)
        where   : ChromaDB metadata pre-filter
        """
        # Candidate pool: fetch more for re-ranking
        n_candidates = min(top_k * self.candidate_multiplier, self.chroma.count() or top_k)

        # Step 1: Dense retrieval
        query_emb = self.embedder.embed_single(query)
        dense_hits = self.chroma.query(query_emb, top_k=n_candidates, where=where)

        if not dense_hits:
            logger.warning("No dense hits for query: %s", query[:60])
            return []

        # Step 2: BM25 over the candidate pool
        sparse_scores = self._bm25_scores(query, [h.text for h in dense_hits])

        # Step 3: Score fusion
        results = self._fuse_scores(dense_hits, sparse_scores)

        # Step 4: MMR or plain top-k
        if use_mmr and len(results) > top_k:
            final = self._mmr(results, query_emb, top_k)
        else:
            final = results[:top_k]

        # Assign final ranks
        for i, r in enumerate(final):
            r.rank = i

        logger.info("Retrieved %d results for query '%s…'", len(final), query[:40])
        return final

    # ── Private ──────────────────────────────────────────────────────────────

    def _bm25_scores(self, query: str, docs: list[str]) -> list[float]:
        tokenized_docs = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenized_docs)
        raw = bm25.get_scores(query.lower().split())
        max_score = raw.max() if raw.max() > 0 else 1.0
        return (raw / max_score).tolist()

    def _fuse_scores(
        self,
        dense_hits: list[RetrievedChunk],
        sparse_scores: list[float],
    ) -> list[RetrievalResult]:
        results = []
        for hit, sp in zip(dense_hits, sparse_scores):
            hybrid = self.dense_weight * hit.score + self.sparse_weight * sp
            results.append(
                RetrievalResult(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    metadata=hit.metadata,
                    dense_score=hit.score,
                    sparse_score=sp,
                    hybrid_score=hybrid,
                    mmr_score=hybrid,   # placeholder until MMR
                    rank=0,
                )
            )
        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results

    def _mmr(
        self,
        candidates: list[RetrievalResult],
        query_emb: np.ndarray,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Maximal Marginal Relevance selection.

        Score(d) = λ * sim(d, q) - (1-λ) * max_sim(d, selected)
        """
        candidate_embs = self.embedder.embed([c.text for c in candidates])
        query_sims = self.embedder.batch_similarity(query_emb, candidate_embs)

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_indices) < top_k and remaining:
            if not selected_indices:
                best = max(remaining, key=lambda i: query_sims[i])
            else:
                sel_embs = candidate_embs[selected_indices]
                mmr_scores: list[float] = []
                for i in remaining:
                    rel = self.mmr_lambda * query_sims[i]
                    red_sims = self.embedder.batch_similarity(candidate_embs[i], sel_embs)
                    red = (1 - self.mmr_lambda) * float(red_sims.max())
                    mmr_scores.append(rel - red)
                best = remaining[int(np.argmax(mmr_scores))]

            selected_indices.append(best)
            remaining.remove(best)

        reranked = []
        for idx in selected_indices:
            r = candidates[idx]
            r.mmr_score = float(query_sims[idx])
            reranked.append(r)

        return reranked
