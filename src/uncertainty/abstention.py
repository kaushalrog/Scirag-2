"""
src/uncertainty/abstention.py
------------------------------
Abstention policy: decides when to withhold an answer based on UQ signals.

Three policies are provided:
  1. ThresholdPolicy    — simple composite confidence threshold
  2. CascadePolicy      — checks multiple signals in priority order
  3. AdaptivePolicy     — adjusts threshold based on query type
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AbstentionReason(str, Enum):
    LOW_COMPOSITE = "composite_confidence_below_threshold"
    LOW_RETRIEVAL = "retrieval_confidence_too_low"
    HIGH_ENTROPY = "generation_entropy_too_high"
    LOW_COVERAGE = "retrieval_coverage_insufficient"
    LOW_CONSISTENCY = "semantic_consistency_too_low"
    NO_CONTEXT = "no_relevant_context"
    NONE = "none"


@dataclass
class AbstentionConfig:
    composite_threshold: float = 0.45
    retrieval_threshold: float = 0.35
    entropy_threshold: float = 2.8
    coverage_threshold: float = 0.30
    consistency_threshold: float = 0.50


class AbstractionPolicy(ABC):
    @abstractmethod
    def should_abstain(self, uq_metrics: dict) -> tuple[bool, str]:
        """Return (should_abstain, reason_string)."""
        ...


class ThresholdPolicy(AbstractionPolicy):
    """
    Simple policy: abstain if composite_confidence < threshold.
    Fast and interpretable.
    """

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold

    def should_abstain(self, uq_metrics: dict) -> tuple[bool, str]:
        conf = uq_metrics.get("composite_confidence", 0.0)
        if conf < self.threshold:
            reason = (
                f"Composite confidence {conf:.3f} below threshold {self.threshold:.3f}"
            )
            logger.info("Abstaining: %s", reason)
            return True, reason
        return False, AbstentionReason.NONE


class CascadePolicy(AbstractionPolicy):
    """
    Checks signals in priority order. Abstains on first failure.

    Priority:
      1. Retrieval confidence (highest priority — if we can't retrieve, we can't answer)
      2. Retrieval coverage
      3. Generation entropy
      4. Semantic consistency
      5. Composite confidence
    """

    def __init__(self, config: AbstentionConfig | None = None) -> None:
        self.cfg = config or AbstentionConfig()

    def should_abstain(self, uq_metrics: dict) -> tuple[bool, str]:
        checks = [
            (
                uq_metrics.get("retrieval_confidence", 0) < self.cfg.retrieval_threshold,
                AbstentionReason.LOW_RETRIEVAL,
                f"Retrieval confidence {uq_metrics.get('retrieval_confidence', 0):.3f} < {self.cfg.retrieval_threshold}",
            ),
            (
                uq_metrics.get("retrieval_coverage", 0) < self.cfg.coverage_threshold,
                AbstentionReason.LOW_COVERAGE,
                f"Coverage {uq_metrics.get('retrieval_coverage', 0):.3f} < {self.cfg.coverage_threshold}",
            ),
            (
                uq_metrics.get("generation_entropy", 0) > self.cfg.entropy_threshold,
                AbstentionReason.HIGH_ENTROPY,
                f"Generation entropy {uq_metrics.get('generation_entropy', 0):.3f} > {self.cfg.entropy_threshold}",
            ),
            (
                uq_metrics.get("semantic_consistency", 1) < self.cfg.consistency_threshold,
                AbstentionReason.LOW_CONSISTENCY,
                f"Consistency {uq_metrics.get('semantic_consistency', 1):.3f} < {self.cfg.consistency_threshold}",
            ),
            (
                uq_metrics.get("composite_confidence", 0) < self.cfg.composite_threshold,
                AbstentionReason.LOW_COMPOSITE,
                f"Composite confidence {uq_metrics.get('composite_confidence', 0):.3f} < {self.cfg.composite_threshold}",
            ),
        ]

        for condition, reason_enum, message in checks:
            if condition:
                logger.info("CascadePolicy abstaining: %s", message)
                return True, message

        return False, AbstentionReason.NONE


class AdaptivePolicy(AbstractionPolicy):
    """
    Adjusts threshold based on query keywords.
    Medical/legal/factual queries get a higher threshold.
    Exploratory/comparison queries get a lower threshold.
    """

    HIGH_STAKES_KEYWORDS = {
        "diagnos", "treat", "legal", "law", "safety", "critical",
        "proof", "certain", "exact", "definitive",
    }
    EXPLORATORY_KEYWORDS = {
        "compare", "survey", "overview", "summarize", "explore",
        "what are", "list", "examples of",
    }

    def __init__(
        self,
        base_threshold: float = 0.45,
        high_stakes_delta: float = 0.15,
        exploratory_delta: float = -0.10,
    ) -> None:
        self.base = base_threshold
        self.high_delta = high_stakes_delta
        self.exp_delta = exploratory_delta

    def should_abstain(self, uq_metrics: dict) -> tuple[bool, str]:
        query = uq_metrics.get("query", "").lower()
        threshold = self.base

        if any(kw in query for kw in self.HIGH_STAKES_KEYWORDS):
            threshold = min(0.90, self.base + self.high_delta)
            logger.debug("AdaptivePolicy: high-stakes query, threshold=%.2f", threshold)
        elif any(kw in query for kw in self.EXPLORATORY_KEYWORDS):
            threshold = max(0.20, self.base + self.exp_delta)
            logger.debug("AdaptivePolicy: exploratory query, threshold=%.2f", threshold)

        conf = uq_metrics.get("composite_confidence", 0.0)
        if conf < threshold:
            return True, f"Adaptive threshold {threshold:.2f} not met (conf={conf:.3f})"
        return False, AbstentionReason.NONE
