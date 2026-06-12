"""
src/generation/rag_chain.py
-----------------------------
Main RAG chain orchestrator.
Ties together: retrieval → context building → generation → uncertainty scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, Optional

from src.generation.groq_client import GenerationResult, GroqClient
from src.generation.prompts import (
    COT_RAG_ANSWER,
    NO_CONTEXT_TEMPLATE,
    RAG_ANSWER,
    build_rag_context,
)
from src.retrieval.retriever import HybridRetriever, RetrievalResult
from src.uncertainty.estimator import UncertaintyEstimator
from src.uncertainty.abstention import AbstractionPolicy

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete response from the RAG chain."""

    question: str
    answer: str
    sources: list[dict]             # [{title, arxiv_id, section, score, url}]
    confidence: float               # overall confidence [0,1]
    uncertainty_breakdown: dict     # detailed UQ metrics
    abstained: bool                 # True if the system chose not to answer
    abstention_reason: str
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    generation_result: Optional[GenerationResult] = None
    context_used: str = ""

    @property
    def is_grounded(self) -> bool:
        return len(self.sources) > 0 and not self.abstained

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "sources": self.sources,
            "uncertainty": self.uncertainty_breakdown,
            "is_grounded": self.is_grounded,
        }


class RAGChain:
    """
    SciRAG-UQ main chain.

    Pipeline
    --------
    1. Retrieve relevant chunks via HybridRetriever
    2. Build formatted context string
    3. Check if context is sufficient (min score threshold)
    4. Generate answer via GroqClient
    5. Estimate uncertainty (UncertaintyEstimator)
    6. Apply abstention policy if confidence too low
    7. Return RAGResponse with full provenance
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: GroqClient,
        uncertainty: UncertaintyEstimator,
        abstention: AbstractionPolicy,
        top_k: int = 5,
        use_cot: bool = False,
        min_retrieval_score: float = 0.30,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.uncertainty = uncertainty
        self.abstention = abstention
        self.top_k = top_k
        self.use_cot = use_cot
        self.min_retrieval_score = min_retrieval_score

    # ── Public ───────────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """Full RAG pipeline — returns a RAGResponse."""
        # Step 1: Retrieve
        hits = self.retriever.retrieve(question, top_k=self.top_k, use_mmr=True)

        # Step 2: Check if retrieval found anything useful
        if not hits or hits[0].hybrid_score < self.min_retrieval_score:
            return self._no_context_response(question)

        # Step 3: Build context
        chunk_dicts = [self._hit_to_dict(h) for h in hits]
        context = build_rag_context(chunk_dicts)

        # Step 4: Generate
        template = COT_RAG_ANSWER if self.use_cot else RAG_ANSWER
        messages = template.format_messages(context=context, question=question)
        gen_result = self.llm.generate(messages, request_logprobs=True)

        # Step 5: Uncertainty estimation
        uq = self.uncertainty.estimate(
            query=question,
            answer=gen_result.text,
            retrieval_results=hits,
            generation_result=gen_result,
        )

        # Step 6: Abstention check
        should_abstain, reason = self.abstention.should_abstain(uq)

        sources = [self._hit_to_source(h) for h in hits[:3]]
        logger.info(
            "RAG query='%s…' conf=%.3f abstain=%s",
            question[:40], uq["composite_confidence"], should_abstain
        )

        return RAGResponse(
            question=question,
            answer=gen_result.text if not should_abstain else self._abstention_answer(reason),
            sources=sources,
            confidence=uq["composite_confidence"],
            uncertainty_breakdown=uq,
            abstained=should_abstain,
            abstention_reason=reason,
            retrieval_results=hits,
            generation_result=gen_result,
            context_used=context,
        )

    def stream_query(
        self, question: str
    ) -> Generator[str, None, None]:
        """Streaming version — yields text tokens. No uncertainty scoring."""
        hits = self.retriever.retrieve(question, top_k=self.top_k, use_mmr=True)
        if not hits:
            yield "I could not find relevant documents for your question."
            return
        chunk_dicts = [self._hit_to_dict(h) for h in hits]
        context = build_rag_context(chunk_dicts)
        messages = RAG_ANSWER.format_messages(context=context, question=question)
        yield from self.llm.stream(messages)

    # ── Private ──────────────────────────────────────────────────────────────

    def _no_context_response(self, question: str) -> RAGResponse:
        messages = NO_CONTEXT_TEMPLATE.format_messages(question=question)
        gen = self.llm.generate(messages)
        return RAGResponse(
            question=question,
            answer=gen.text,
            sources=[],
            confidence=0.0,
            uncertainty_breakdown={"composite_confidence": 0.0, "reason": "no_context"},
            abstained=True,
            abstention_reason="No relevant documents found in corpus",
            generation_result=gen,
        )

    @staticmethod
    def _abstention_answer(reason: str) -> str:
        return (
            f"⚠️ **Low Confidence — Answer Withheld**\n\n"
            f"Reason: {reason}\n\n"
            "The system detected insufficient evidence or high uncertainty in the retrieved "
            "passages. Please refine your query or expand the document corpus."
        )

    @staticmethod
    def _hit_to_dict(h: RetrievalResult) -> dict:
        return {
            "text": h.text,
            "title": h.title,
            "arxiv_id": h.metadata.get("arxiv_id", ""),
            "section": h.section,
            "score": h.hybrid_score,
            "url": h.metadata.get("url", ""),
        }

    @staticmethod
    def _hit_to_source(h: RetrievalResult) -> dict:
        return {
            "title": h.title,
            "arxiv_id": h.metadata.get("arxiv_id", ""),
            "section": h.section,
            "score": round(h.hybrid_score, 4),
            "url": h.metadata.get("url", ""),
            "published": h.metadata.get("published", ""),
        }
