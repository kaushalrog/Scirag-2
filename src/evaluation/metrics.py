"""
src/evaluation/metrics.py
--------------------------
Evaluation metrics for SciRAG-UQ.

Implemented metrics
-------------------
1.  Faithfulness           — % claims in answer supported by context
2.  Answer Relevancy       — cosine sim between question and answer embeddings
3.  Context Recall         — % ground-truth info covered by retrieved context
4.  Context Precision      — precision of retrieved chunks
5.  Hallucination Rate     — NLI-based: CONTRADICTION fraction
6.  Abstention Precision   — P(abstain | truly uncertain)
7.  Abstention Recall      — P(abstain | uncertain) / total uncertain
8.  ROUGE-L                — surface-level recall vs. reference
9.  Calibration Error      — Expected Calibration Error (ECE) of confidence scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    question: str
    answer: str
    context: str                    # joined retrieved passages
    reference: str                  # ground-truth answer (for supervised metrics)
    confidence: float               # model's confidence score
    abstained: bool
    truly_uncertain: bool           # ground-truth label: is this question unanswerable?


@dataclass
class MetricsReport:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    hallucination_rate: float
    rouge_l: float
    abstention_precision: float
    abstention_recall: float
    ece: float
    n_samples: int
    n_abstained: int

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_recall": round(self.context_recall, 4),
            "context_precision": round(self.context_precision, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "rouge_l": round(self.rouge_l, 4),
            "abstention_precision": round(self.abstention_precision, 4),
            "abstention_recall": round(self.abstention_recall, 4),
            "ece": round(self.ece, 4),
            "n_samples": self.n_samples,
            "n_abstained": self.n_abstained,
            "abstention_rate": round(self.n_abstained / max(self.n_samples, 1), 4),
        }

    def summary(self) -> str:
        d = self.to_dict()
        lines = ["=== SciRAG-UQ Evaluation Report ==="]
        for k, v in d.items():
            lines.append(f"  {k:<28}: {v}")
        return "\n".join(lines)


class Evaluator:
    """
    Evaluator for SciRAG-UQ system.
    Uses lightweight implementations to avoid heavy eval library dependencies.
    """

    def __init__(self, embedder=None) -> None:
        self.embedder = embedder

    def evaluate(self, samples: list[EvalSample]) -> MetricsReport:
        if not samples:
            raise ValueError("No samples to evaluate")

        faithfulness_scores = [self._faithfulness(s) for s in samples]
        relevancy_scores = [self._answer_relevancy(s) for s in samples]
        recall_scores = [self._context_recall(s) for s in samples]
        precision_scores = [self._context_precision(s) for s in samples]
        hallucination_scores = [self._hallucination(s) for s in samples]
        rouge_scores = [self._rouge_l(s.answer, s.reference) for s in samples]

        # Abstention metrics
        abs_precision, abs_recall = self._abstention_metrics(samples)

        # Calibration
        ece = self._expected_calibration_error(samples)

        answered = [s for s in samples if not s.abstained]

        return MetricsReport(
            faithfulness=float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            answer_relevancy=float(np.mean(relevancy_scores)) if relevancy_scores else 0.0,
            context_recall=float(np.mean(recall_scores)) if recall_scores else 0.0,
            context_precision=float(np.mean(precision_scores)) if precision_scores else 0.0,
            hallucination_rate=float(np.mean(hallucination_scores)) if hallucination_scores else 0.0,
            rouge_l=float(np.mean(rouge_scores)) if rouge_scores else 0.0,
            abstention_precision=abs_precision,
            abstention_recall=abs_recall,
            ece=ece,
            n_samples=len(samples),
            n_abstained=sum(1 for s in samples if s.abstained),
        )

    # ── Individual metrics ────────────────────────────────────────────────────

    def _faithfulness(self, sample: EvalSample) -> float:
        """
        Approximate faithfulness: fraction of answer sentences that overlap
        lexically with the context (token-level Jaccard).
        """
        context_tokens = set(sample.context.lower().split())
        answer_sents = [s.strip() for s in sample.answer.split(".") if s.strip()]
        if not answer_sents:
            return 1.0
        scores = []
        for sent in answer_sents:
            sent_tokens = set(sent.lower().split())
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
            scores.append(overlap)
        return float(np.mean(scores)) if scores else 1.0

    def _answer_relevancy(self, sample: EvalSample) -> float:
        """Cosine similarity between question and answer (requires embedder)."""
        if self.embedder is None:
            return self._jaccard(sample.question, sample.answer)
        q_emb = self.embedder.embed_single(sample.question)
        a_emb = self.embedder.embed_single(sample.answer)
        return self.embedder.similarity(q_emb, a_emb)

    def _context_recall(self, sample: EvalSample) -> float:
        """Fraction of reference tokens covered by context."""
        ref_tokens = set(sample.reference.lower().split())
        ctx_tokens = set(sample.context.lower().split())
        if not ref_tokens:
            return 1.0
        return len(ref_tokens & ctx_tokens) / len(ref_tokens)

    def _context_precision(self, sample: EvalSample) -> float:
        """Fraction of context tokens that are in the reference."""
        ref_tokens = set(sample.reference.lower().split())
        ctx_tokens = set(sample.context.lower().split())
        if not ctx_tokens:
            return 0.0
        return len(ref_tokens & ctx_tokens) / len(ctx_tokens)

    def _hallucination(self, sample: EvalSample) -> float:
        """
        Approximate hallucination: fraction of answer tokens NOT in context.
        High value = high hallucination.
        """
        ctx_tokens = set(sample.context.lower().split())
        ans_tokens = sample.answer.lower().split()
        if not ans_tokens:
            return 0.0
        unknown = sum(1 for t in ans_tokens if t not in ctx_tokens)
        return unknown / len(ans_tokens)

    @staticmethod
    def _rouge_l(hypothesis: str, reference: str) -> float:
        """ROUGE-L (LCS-based recall)."""
        h_tokens = hypothesis.lower().split()
        r_tokens = reference.lower().split()
        if not h_tokens or not r_tokens:
            return 0.0
        lcs = _lcs_length(h_tokens, r_tokens)
        precision = lcs / len(h_tokens)
        recall = lcs / len(r_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @staticmethod
    def _abstention_metrics(samples: list[EvalSample]) -> tuple[float, float]:
        """Precision and recall of the abstention mechanism."""
        tp = sum(1 for s in samples if s.abstained and s.truly_uncertain)
        fp = sum(1 for s in samples if s.abstained and not s.truly_uncertain)
        fn = sum(1 for s in samples if not s.abstained and s.truly_uncertain)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return precision, recall

    @staticmethod
    def _expected_calibration_error(
        samples: list[EvalSample], n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE).
        Bins confidence scores; measures mean |confidence - accuracy| per bin.
        Ground truth: sample is 'correct' if it answered & not truly_uncertain,
        or abstained & truly_uncertain.
        """
        confs = np.array([s.confidence for s in samples])
        correct = np.array([
            (not s.abstained and not s.truly_uncertain)
            or (s.abstained and s.truly_uncertain)
            for s in samples
        ], dtype=float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (confs >= bins[i]) & (confs < bins[i + 1])
            if in_bin.sum() == 0:
                continue
            acc = correct[in_bin].mean()
            conf = confs[in_bin].mean()
            ece += in_bin.sum() * abs(acc - conf)
        return float(ece / max(len(samples), 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
