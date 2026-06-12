"""
src/generation/prompts.py
--------------------------
All prompt templates for SciRAG-UQ.
Using simple f-string templates for maximum transparency and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    system: str
    user_template: str

    def format_messages(self, **kwargs) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_template.format(**kwargs)},
        ]


# ── RAG Answer Template ────────────────────────────────────────────────────

RAG_ANSWER = PromptTemplate(
    system="""You are SciRAG-UQ, a precise scientific literature assistant.
Your answers are grounded strictly in the provided context passages.

Rules:
1. Answer ONLY from the context. Do NOT use prior knowledge.
2. If the context is insufficient, say: "The provided documents do not contain enough information to answer this question."
3. Always cite the source paper (title or arXiv ID) when making a claim.
4. Be concise but complete. Use bullet points for multi-part answers.
5. If multiple papers disagree, present both views.
""",
    user_template="""Context passages (from scientific papers):

{context}

---
Question: {question}

Answer (cite sources):""",
)

# ── Chain-of-Thought RAG Template ─────────────────────────────────────────

COT_RAG_ANSWER = PromptTemplate(
    system="""You are SciRAG-UQ, a scientific reasoning assistant.
Think step-by-step before answering. Ground all reasoning in the provided context.

Rules:
1. First, identify which passages are relevant.
2. Reason through the evidence systematically.
3. Conclude with a clear, cited answer.
4. Flag any gaps or contradictions in the evidence.
""",
    user_template="""Context passages:

{context}

---
Question: {question}

Step-by-step reasoning then answer:""",
)

# ── No-Context Fallback Template ──────────────────────────────────────────

NO_CONTEXT_TEMPLATE = PromptTemplate(
    system="""You are SciRAG-UQ. You have been unable to retrieve relevant context.
Clearly inform the user that the answer is not in the current document corpus.
Suggest alternative search terms or related topics they could try.
""",
    user_template="""The retrieval system found no relevant passages for: "{question}"

Please inform the user and suggest how to proceed.""",
)

# ── Confidence Introspection Template ─────────────────────────────────────

CONFIDENCE_CHECK = PromptTemplate(
    system="""You are a scientific fact-checker. Given a question and an answer,
assess the answer's reliability on a scale from 0.0 to 1.0.

Respond ONLY with valid JSON:
{{"confidence": <float 0-1>, "reason": "<brief explanation>", "flags": ["<any concerns>"]}}
""",
    user_template="""Question: {question}

Answer: {answer}

Context used: {context_summary}

Assess confidence (JSON only):""",
)

# ── Summary Template ──────────────────────────────────────────────────────

SUMMARISE_PAPER = PromptTemplate(
    system="""You are a scientific summariser. Create structured summaries of academic papers.
Output format:
- **Problem**: What problem does this paper solve?
- **Method**: What approach do they use?
- **Key Results**: Top 2-3 findings with numbers if available.
- **Limitations**: What do the authors acknowledge as limitations?
- **Relevance to RAG/NLP**: How does this work relate to retrieval or language models?
""",
    user_template="""Paper: {title}
Authors: {authors}

Abstract:
{abstract}

Full text excerpt:
{excerpt}

Structured summary:""",
)

# ── Query Expansion Template ──────────────────────────────────────────────

QUERY_EXPANSION = PromptTemplate(
    system="""You are a search query expansion specialist for scientific literature.
Given a user query, generate 3 alternative phrasings that capture the same intent
but use different terminology (synonyms, acronyms, related concepts).

Output ONLY a JSON array of 3 strings, no explanation.
""",
    user_template='Expand this scientific query into 3 alternatives:\n"{query}"\n\nJSON array:',
)


def build_rag_context(chunks: list[dict], max_chars: int = 3000) -> str:
    """
    Build a formatted context string from retrieved chunks.

    Parameters
    ----------
    chunks    : List of dicts with keys: text, title, arxiv_id, section, score
    max_chars : Truncate total context to this length
    """
    parts: list[str] = []
    total = 0

    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Unknown")
        arxiv_id = chunk.get("arxiv_id", "")
        section = chunk.get("section", "")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")

        header = f"[{i}] {title}"
        if arxiv_id:
            header += f" (arXiv:{arxiv_id})"
        if section:
            header += f" — §{section}"
        header += f" [score={score:.3f}]"

        entry = f"{header}\n{text}\n"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(entry[:remaining] + "…")
            break
        parts.append(entry)
        total += len(entry)

    return "\n---\n".join(parts)
