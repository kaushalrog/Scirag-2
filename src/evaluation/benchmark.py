"""
src/evaluation/benchmark.py
-----------------------------
Benchmark dataset builder for SciRAG-UQ evaluation.

Creates a curated Q&A dataset from ingested papers with
ground-truth labels for answerable/unanswerable questions.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkItem:
    item_id: str
    question: str
    reference_answer: str
    source_arxiv_id: str
    source_section: str
    category: str           # factual | comparative | exploratory | unanswerable
    truly_uncertain: bool   # True for unanswerable questions
    difficulty: str         # easy | medium | hard


class BenchmarkBuilder:
    """
    Builds a structured evaluation benchmark.

    Question categories
    -------------------
    - factual       : Single-hop questions with definite answers
    - comparative   : Multi-document comparison questions
    - exploratory   : Open-ended survey questions
    - unanswerable  : Questions outside the corpus (negative test)
    """

    FACTUAL_TEMPLATES = [
        "What method does {title} propose for {topic}?",
        "What dataset was used to evaluate {title}?",
        "What is the main contribution of {title}?",
        "What baseline did {title} compare against?",
        "What accuracy/F1/metric did {title} achieve?",
    ]

    COMPARATIVE_TEMPLATES = [
        "How does the approach in {title1} differ from {title2}?",
        "Which paper achieves better results on {task}: {title1} or {title2}?",
        "What are the common limitations of {title1} and {title2}?",
    ]

    UNANSWERABLE_TEMPLATES = [
        "What is the economic cost of implementing the system in {title}?",
        "How does the method in {title} perform on real-time video streams?",
        "What are the ethical implications of {title} in clinical settings?",
        "What is the carbon footprint of training the model in {title}?",
    ]

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)

    def build_from_documents(
        self,
        documents: list[dict],          # list of IngestedDocument.to_metadata() dicts
        n_factual: int = 50,
        n_comparative: int = 20,
        n_unanswerable: int = 30,
        output_path: Optional[str] = None,
    ) -> list[BenchmarkItem]:
        """
        Auto-generate a benchmark from ingested documents.
        Note: reference answers are stubs — fill them in manually or with LLM.
        """
        items: list[BenchmarkItem] = []
        docs = [d for d in documents if d.get("title")]
        if not docs:
            logger.warning("No documents to build benchmark from")
            return []

        # Factual
        for i in range(min(n_factual, len(docs))):
            doc = docs[i % len(docs)]
            template = random.choice(self.FACTUAL_TEMPLATES)
            q = template.format(
                title=doc["title"][:50],
                topic=self._extract_topic(doc),
            )
            items.append(BenchmarkItem(
                item_id=f"factual_{i:04d}",
                question=q,
                reference_answer=doc.get("abstract", "")[:200],
                source_arxiv_id=doc.get("arxiv_id", ""),
                source_section="abstract",
                category="factual",
                truly_uncertain=False,
                difficulty=self._assign_difficulty(doc),
            ))

        # Comparative (pairs)
        pairs = list(zip(docs[::2], docs[1::2]))[:n_comparative]
        for i, (d1, d2) in enumerate(pairs):
            template = random.choice(self.COMPARATIVE_TEMPLATES)
            q = template.format(
                title1=d1["title"][:40],
                title2=d2["title"][:40],
                task="the benchmark task",
            )
            items.append(BenchmarkItem(
                item_id=f"comparative_{i:04d}",
                question=q,
                reference_answer="[Multi-document reference — fill manually]",
                source_arxiv_id=f"{d1.get('arxiv_id', '')}|{d2.get('arxiv_id', '')}",
                source_section="multi-doc",
                category="comparative",
                truly_uncertain=False,
                difficulty="medium",
            ))

        # Unanswerable
        for i in range(min(n_unanswerable, len(docs))):
            doc = docs[i % len(docs)]
            q = random.choice(self.UNANSWERABLE_TEMPLATES).format(title=doc["title"][:50])
            items.append(BenchmarkItem(
                item_id=f"unanswerable_{i:04d}",
                question=q,
                reference_answer="[UNANSWERABLE — not in corpus]",
                source_arxiv_id=doc.get("arxiv_id", ""),
                source_section="N/A",
                category="unanswerable",
                truly_uncertain=True,
                difficulty="hard",
            ))

        random.shuffle(items)
        logger.info("Built benchmark: %d items", len(items))

        if output_path:
            self._save(items, output_path)

        return items

    def load(self, path: str) -> list[BenchmarkItem]:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return [BenchmarkItem(**d) for d in data]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_topic(doc: dict) -> str:
        cats = doc.get("categories", "").split(";")
        return cats[0].strip() if cats else "the proposed task"

    @staticmethod
    def _assign_difficulty(doc: dict) -> str:
        tldr = doc.get("tldr", "")
        if tldr:
            return "easy"
        abstract = doc.get("abstract", "")
        return "medium" if len(abstract) > 300 else "hard"

    @staticmethod
    def _save(items: list[BenchmarkItem], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(i) for i in items], fh, indent=2)
        logger.info("Benchmark saved to %s", path)
