"""
src/embeddings/embedder.py
---------------------------
SentenceTransformers-based embedder with batch processing,
caching, and timing instrumentation.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps SentenceTransformers for batch embedding.

    Parameters
    ----------
    model_name : SentenceTransformers model identifier
    batch_size : GPU/CPU batch size for encoding
    normalize  : L2-normalise output vectors (recommended for cosine sim)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dim: %d", self.dim)

    # ── Public ───────────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray of shape (N, dim), dtype float32
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        t0 = time.perf_counter()
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - t0
        logger.debug(
            "Embedded %d texts in %.2fs (%.1f texts/s)",
            len(texts),
            elapsed,
            len(texts) / elapsed,
        )
        return vectors.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (dim,)."""
        return self.embed([text])[0]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two (possibly unnormalised) vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def batch_similarity(
        self, query: np.ndarray, corpus: np.ndarray
    ) -> np.ndarray:
        """
        Cosine similarity between one query vector and N corpus vectors.
        Returns shape (N,).
        """
        norms = np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10
        normed = corpus / norms
        qn = query / (np.linalg.norm(query) + 1e-10)
        return (normed @ qn).astype(np.float32)


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Return a cached singleton Embedder."""
    return Embedder(model_name=model_name)
