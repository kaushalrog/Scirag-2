"""
src/evaluation/runner.py
-------------------------
Evaluation runner: runs the RAG chain over a benchmark dataset
and computes all metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.evaluation.benchmark import BenchmarkItem
from src.evaluation.metrics import EvalSample, Evaluator, MetricsReport
from src.generation.rag_chain import RAGChain

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Runs the full eval loop: benchmark items → RAG chain → metrics.

    Parameters
    ----------
    chain     : Configured RAGChain instance
    evaluator : Evaluator instance (with optional embedder)
    output_dir: Where to save results
    """

    def __init__(
        self,
        chain: RAGChain,
        evaluator: Evaluator,
        output_dir: str = "./results",
    ) -> None:
        self.chain = chain
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        benchmark: list[BenchmarkItem],
        run_name: str = "eval",
        save_predictions: bool = True,
    ) -> MetricsReport:
        """
        Run evaluation over all benchmark items.

        Returns MetricsReport and saves JSON results.
        """
        samples: list[EvalSample] = []
        predictions: list[dict] = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Evaluating [{run_name}]…", total=len(benchmark))

            for item in benchmark:
                response = self.chain.query(item.question)
                sample = EvalSample(
                    question=item.question,
                    answer=response.answer,
                    context=response.context_used,
                    reference=item.reference_answer,
                    confidence=response.confidence,
                    abstained=response.abstained,
                    truly_uncertain=item.truly_uncertain,
                )
                samples.append(sample)

                if save_predictions:
                    predictions.append({
                        "item_id": item.item_id,
                        "question": item.question,
                        "answer": response.answer,
                        "confidence": response.confidence,
                        "abstained": response.abstained,
                        "abstention_reason": response.abstention_reason,
                        "sources": response.sources,
                        "uq_breakdown": response.uncertainty_breakdown,
                        "reference": item.reference_answer,
                        "category": item.category,
                        "truly_uncertain": item.truly_uncertain,
                    })

                progress.advance(task)

        report = self.evaluator.evaluate(samples)

        # Save results
        report_path = self.output_dir / f"{run_name}_report.json"
        with open(report_path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        logger.info("Report saved: %s", report_path)

        if save_predictions:
            pred_path = self.output_dir / f"{run_name}_predictions.json"
            with open(pred_path, "w") as fh:
                json.dump(predictions, fh, indent=2, ensure_ascii=False)
            logger.info("Predictions saved: %s", pred_path)

        print(report.summary())
        return report

    def run_ablation(
        self,
        benchmark: list[BenchmarkItem],
        configurations: dict[str, RAGChain],
    ) -> dict[str, MetricsReport]:
        """
        Run evaluation for multiple chain configurations (ablation study).

        Parameters
        ----------
        configurations: dict mapping config name → RAGChain instance
        """
        results: dict[str, MetricsReport] = {}
        for name, chain in configurations.items():
            logger.info("Running ablation: %s", name)
            self.chain = chain
            results[name] = self.run(benchmark, run_name=name)

        # Save comparison
        comparison = {
            name: report.to_dict()
            for name, report in results.items()
        }
        with open(self.output_dir / "ablation_comparison.json", "w") as fh:
            json.dump(comparison, fh, indent=2)

        return results
