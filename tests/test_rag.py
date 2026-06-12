"""tests/test_rag.py — Integration tests for RAG chain components."""

import pytest
from unittest.mock import MagicMock, patch
from src.uncertainty.estimator import UncertaintyEstimator
from src.uncertainty.abstention import (
    ThresholdPolicy, CascadePolicy, AbstentionConfig
)
from src.evaluation.metrics import Evaluator, EvalSample, _lcs_length


class TestUncertaintyEstimator:
    def test_confidence_labels(self):
        assert UncertaintyEstimator.confidence_to_label(0.90) == "HIGH"
        assert UncertaintyEstimator.confidence_to_label(0.70) == "MEDIUM"
        assert UncertaintyEstimator.confidence_to_label(0.50) == "LOW"
        assert UncertaintyEstimator.confidence_to_label(0.20) == "VERY LOW"

    def test_retrieval_confidence_empty(self):
        est = UncertaintyEstimator()
        conf, cov = est._retrieval_confidence([])
        assert conf == 0.0
        assert cov == 0.0

    def test_retrieval_confidence_with_mocks(self):
        est = UncertaintyEstimator()
        hits = [MagicMock(hybrid_score=0.8), MagicMock(hybrid_score=0.6),
                MagicMock(hybrid_score=0.3)]
        conf, cov = est._retrieval_confidence(hits)
        assert abs(conf - (0.8 + 0.6 + 0.3) / 3) < 0.001
        # 2 out of 3 above threshold 0.45
        assert abs(cov - 2/3) < 0.001

    def test_generation_confidence_no_logprobs(self):
        est = UncertaintyEstimator()
        entropy, conf = est._generation_confidence(None)
        assert entropy == 0.0
        assert conf == 0.8

    def test_semantic_consistency_no_embedder(self):
        est = UncertaintyEstimator()
        score = est._semantic_consistency("answer", ["alt1", "alt2"])
        assert score == 0.85  # neutral prior

    def test_estimate_returns_all_keys(self):
        est = UncertaintyEstimator()
        hits = [MagicMock(hybrid_score=0.7)]
        metrics = est.estimate("question", "answer", hits)
        required = ["retrieval_confidence", "generation_confidence",
                    "semantic_consistency", "composite_confidence",
                    "retrieval_coverage", "generation_entropy"]
        for key in required:
            assert key in metrics

    def test_composite_clipped_to_unit_interval(self):
        est = UncertaintyEstimator()
        hits = [MagicMock(hybrid_score=1.0)] * 5
        metrics = est.estimate("q", "a", hits)
        assert 0.0 <= metrics["composite_confidence"] <= 1.0


class TestAbstentionPolicies:
    def test_threshold_abstains_below(self):
        policy = ThresholdPolicy(threshold=0.6)
        abstain, reason = policy.should_abstain({"composite_confidence": 0.4})
        assert abstain is True
        assert "0.400" in reason

    def test_threshold_passes_above(self):
        policy = ThresholdPolicy(threshold=0.6)
        abstain, _ = policy.should_abstain({"composite_confidence": 0.8})
        assert abstain is False

    def test_cascade_abstains_on_low_retrieval(self):
        policy = CascadePolicy(AbstentionConfig(retrieval_threshold=0.5))
        abstain, reason = policy.should_abstain({
            "retrieval_confidence": 0.2,
            "retrieval_coverage": 0.8,
            "generation_entropy": 1.0,
            "semantic_consistency": 0.9,
            "composite_confidence": 0.7,
        })
        assert abstain is True

    def test_cascade_passes_all_signals(self):
        policy = CascadePolicy()
        abstain, _ = policy.should_abstain({
            "retrieval_confidence": 0.8,
            "retrieval_coverage": 0.8,
            "generation_entropy": 1.0,
            "semantic_consistency": 0.9,
            "composite_confidence": 0.85,
        })
        assert abstain is False


class TestEvaluator:
    def _sample(self, answer, reference, context=None, confidence=0.8,
                abstained=False, uncertain=False):
        return EvalSample(
            question="What is RAG?",
            answer=answer,
            context=context or reference,
            reference=reference,
            confidence=confidence,
            abstained=abstained,
            truly_uncertain=uncertain,
        )

    def test_rouge_l_identical(self):
        ev = Evaluator()
        score = ev._rouge_l("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(1.0)

    def test_rouge_l_no_overlap(self):
        ev = Evaluator()
        score = ev._rouge_l("aaa bbb ccc", "xxx yyy zzz")
        assert score == 0.0

    def test_faithfulness_full_context(self):
        ev = Evaluator()
        sample = self._sample(
            answer="The model uses attention mechanisms.",
            reference="The model uses attention mechanisms.",
            context="The model uses attention mechanisms in transformer architecture.",
        )
        score = ev._faithfulness(sample)
        assert score > 0.5

    def test_lcs_length(self):
        a = ["a", "b", "c", "d"]
        b = ["b", "c", "d", "e"]
        assert _lcs_length(a, b) == 3

    def test_evaluate_report_fields(self):
        ev = Evaluator()
        samples = [
            self._sample("RAG uses retrieval", "RAG uses retrieval and generation"),
            self._sample("unknown answer", "known reference", uncertain=True,
                         abstained=True, confidence=0.3),
        ]
        report = ev.evaluate(samples)
        d = report.to_dict()
        assert "faithfulness" in d
        assert "hallucination_rate" in d
        assert "abstention_precision" in d
        assert "ece" in d
        assert d["n_samples"] == 2
        assert d["n_abstained"] == 1
