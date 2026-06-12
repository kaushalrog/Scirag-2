"""
src/uncertainty/estimator.py
------------------------------
Core uncertainty quantification module for SciRAG-UQ.

Three complementary signals are combined:
  1. Retrieval confidence  — how well retrieved chunks match the query
  2. Generation entropy    — token-level entropy from logprobs
  3. Semantic consistency  — self-consistency via multiple generations (optional)

Together they form a composite confidence score in [0, 1].
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.embeddings.embedder import Embedder
from src.generation.groq_client import GenerationResult
from src.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class UQMetrics:
    retrieval_confidence: float      # mean retrieval score of top-k hits
    retrieval_coverage: float        # fraction of hits above threshold
    generation_entropy: float        # mean token entropy [0, ∞)
    generation_confidence: float     # 1 / (1 + entropy) ∈ (0, 1]
    semantic_consistency: float      # embedding sim across multiple answers (0-1)
    composite_confidence: float      # weighted final score
    signals: dict                    # raw signal values for analysis


class UncertaintyEstimator:
    """
    Multi-signal uncertainty estimator.

    Weights
    -------
    w_retrieval = 0.40
    w_generation = 0.35
    w_consistency = 0.25

    These were tuned on the BDA-2026 benchmark (see paper §4.3).
    """

    W_RETRIEVAL = 0.40
    W_GENERATION = 0.35
    W_CONSISTENCY = 0.25

    RETRIEVAL_SCORE_THRESHOLD = 0.45    # hits below this are considered weak

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        consistency_samples: int = 0,    # 0 = skip consistency (expensive)
    ) -> None:
        self.embedder = embedder
        self.consistency_samples = consistency_samples

    # ── Public ───────────────────────────────────────────────────────────────

    def estimate(
        self,
        query: str,
        answer: str,
        retrieval_results: list[RetrievalResult],
        generation_result: Optional[GenerationResult] = None,
        alternate_answers: Optional[list[str]] = None,
    ) -> dict:
        """
        Compute full UQ metrics and return as a flat dict.

        Parameters
        ----------
        query             : Original user query
        answer            : Generated answer text
        retrieval_results : Top-k retrieval hits
        generation_result : GenerationResult with optional logprobs
        alternate_answers : Optional list of alternate answers for consistency
        """
        # Signal 1: Retrieval
        r_conf, r_cov = self._retrieval_confidence(retrieval_results)

        # Signal 2: Generation entropy
        g_entropy, g_conf = self._generation_confidence(generation_result)

        # Signal 3: Semantic consistency
        consistency = self._semantic_consistency(answer, alternate_answers)

        # Composite
        composite = (
            self.W_RETRIEVAL * r_conf
            + self.W_GENERATION * g_conf
            + self.W_CONSISTENCY * consistency
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        metrics = {
            "retrieval_confidence": round(r_conf, 4),
            "retrieval_coverage": round(r_cov, 4),
            "generation_entropy": round(g_entropy, 4),
            "generation_confidence": round(g_conf, 4),
            "semantic_consistency": round(consistency, 4),
            "composite_confidence": round(composite, 4),
            "n_hits": len(retrieval_results),
            "top_hit_score": round(retrieval_results[0].hybrid_score, 4) if retrieval_results else 0.0,
            "weights": {
                "retrieval": self.W_RETRIEVAL,
                "generation": self.W_GENERATION,
                "consistency": self.W_CONSISTENCY,
            },
        }
        logger.debug("UQ metrics: %s", metrics)
        return metrics

    # ── Signals ──────────────────────────────────────────────────────────────

    def _retrieval_confidence(
        self, results: list[RetrievalResult]
    ) -> tuple[float, float]:
        """
        Returns (mean_score, coverage_fraction).
        mean_score   : average hybrid score of top-k hits
        coverage     : fraction of hits above RETRIEVAL_SCORE_THRESHOLD
        """
        if not results:
            return 0.0, 0.0
        scores = np.array([r.hybrid_score for r in results])
        mean_score = float(scores.mean())
        coverage = float((scores >= self.RETRIEVAL_SCORE_THRESHOLD).mean())
        return mean_score, coverage

    def _generation_confidence(
        self, result: Optional[GenerationResult]
    ) -> tuple[float, float]:
        """
        Returns (entropy, confidence).
        entropy    : mean token-level entropy across top-logprob distribution
        confidence : sigmoid-like mapping ∈ (0, 1]
        """
        if result is None or not result.logprobs:
            return 0.0, 0.8   # no logprobs → neutral confidence

        entropies: list[float] = []
        for token_data in result.logprobs:
            top_lps = token_data.get("top_logprobs", [])
            if not top_lps:
                continue
            lps = np.array([t["logprob"] for t in top_lps], dtype=float)
            # Normalise to probabilities
            probs = np.exp(lps - lps.max())
            probs /= probs.sum()
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            entropies.append(entropy)

        if not entropies:
            return 0.0, 0.8

        mean_entropy = float(np.mean(entropies))
        # Map entropy → confidence: low entropy = high confidence
        confidence = 1.0 / (1.0 + mean_entropy)
        return mean_entropy, confidence

    def _semantic_consistency(
        self,
        primary_answer: str,
        alternates: Optional[list[str]],
    ) -> float:
        """
        Cosine similarity between primary answer and each alternate.
        Returns mean similarity or 0.85 if no alternates (neutral prior).
        """
        if not alternates or self.embedder is None:
            return 0.85

        texts = [primary_answer] + alternates
        embs = self.embedder.embed(texts)
        primary_emb = embs[0]
        alt_embs = embs[1:]

        sims = self.embedder.batch_similarity(primary_emb, alt_embs)
        return float(sims.mean())

    # ── Calibration helpers ───────────────────────────────────────────────────

    @staticmethod
    def confidence_to_label(score: float) -> str:
        """Human-readable confidence label."""
        if score >= 0.80:
            return "HIGH"
        if score >= 0.60:
            return "MEDIUM"
        if score >= 0.40:
            return "LOW"
        return "VERY LOW"

    @staticmethod
    def entropy_from_logprob(logprob: float) -> float:
        """Single-token entropy from a scalar logprob."""
        p = math.exp(logprob)
        return -p * logprob if p > 0 else 0.0
