"""
src/embeddings/chunker.py
--------------------------
Two-strategy chunking:
  1. RecursiveCharacterSplitter  — baseline, good for arbitrary text
  2. SemanticChunker             — sentence-level, groups by embedding similarity
     (used for high-quality section chunking)

Each chunk carries full provenance metadata for downstream attribution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SECTION = "section"       # one chunk per detected section


@dataclass
class TextChunk:
    """A single chunk of text with full provenance."""

    chunk_id: str
    text: str
    source_id: str            # arxiv_id or filename
    title: str
    section: str              # section name, if known
    char_start: int
    char_end: int
    chunk_index: int
    strategy: str
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class RecursiveChunker:
    """
    Splits text using a hierarchy of separators:
      paragraphs → sentences → words → chars
    Guarantees ``chunk_size`` with ``overlap`` for context continuity.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(
        self,
        text: str,
        source_id: str = "",
        title: str = "",
        section: str = "",
        extra_metadata: dict | None = None,
    ) -> list[TextChunk]:
        raw_chunks = self._recursive_split(text, self.separators)
        chunks: list[TextChunk] = []
        pos = 0
        for i, raw in enumerate(raw_chunks):
            start = text.find(raw, pos)
            end = start + len(raw)
            pos = max(pos, start)
            chunks.append(
                TextChunk(
                    chunk_id=f"{source_id}_rc_{i:04d}",
                    text=raw,
                    source_id=source_id,
                    title=title,
                    section=section,
                    char_start=start,
                    char_end=end,
                    chunk_index=i,
                    strategy=ChunkStrategy.RECURSIVE,
                    metadata=extra_metadata or {},
                )
            )
        logger.debug("RecursiveChunker: %d chunks from %d chars", len(chunks), len(text))
        return chunks

    # ── Internal ─────────────────────────────────────────────────────────────

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._split_by_size(text)

        sep = separators[0]
        rest = separators[1:]

        if sep == "":
            return self._split_by_size(text)

        splits = text.split(sep)
        good: list[str] = []
        current = ""

        for split in splits:
            candidate = current + (sep if current else "") + split
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good.append(current)
                if len(split) > self.chunk_size:
                    good.extend(self._recursive_split(split, rest))
                    current = ""
                else:
                    current = split

        if current:
            good.append(current)

        # Apply overlap
        return self._apply_overlap(good)

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.overlap:]
            result.append(prev_tail + chunks[i])
        return result

    def _split_by_size(self, text: str) -> list[str]:
        return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


class SemanticChunker:
    """
    Groups sentences into chunks based on embedding cosine similarity.
    Adjacent sentences with high similarity stay in the same chunk;
    a drop in similarity signals a semantic boundary.

    Requires an embedder function: (list[str]) -> np.ndarray of shape (N, D).
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray],
        similarity_threshold: float = 0.75,
        max_chunk_size: int = 600,
        min_chunk_size: int = 50,
    ) -> None:
        self.embed_fn = embed_fn
        self.threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split(
        self,
        text: str,
        source_id: str = "",
        title: str = "",
        section: str = "",
        extra_metadata: dict | None = None,
    ) -> list[TextChunk]:
        sentences = self._sentence_tokenize(text)
        if not sentences:
            return []

        embeddings = self.embed_fn(sentences)
        groups = self._group_sentences(sentences, embeddings)

        chunks: list[TextChunk] = []
        char_pos = 0
        for i, group in enumerate(groups):
            combined = " ".join(group)
            start = text.find(group[0], char_pos)
            end = start + len(combined)
            char_pos = start + 1
            chunks.append(
                TextChunk(
                    chunk_id=f"{source_id}_sem_{i:04d}",
                    text=combined,
                    source_id=source_id,
                    title=title,
                    section=section,
                    char_start=max(0, start),
                    char_end=min(len(text), end),
                    chunk_index=i,
                    strategy=ChunkStrategy.SEMANTIC,
                    metadata=extra_metadata or {},
                )
            )
        logger.debug("SemanticChunker: %d chunks from %d sentences", len(chunks), len(sentences))
        return chunks

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _sentence_tokenize(text: str) -> list[str]:
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _group_sentences(
        self, sentences: list[str], embeddings: np.ndarray
    ) -> list[list[str]]:
        groups: list[list[str]] = [[sentences[0]]]
        current_text = sentences[0]

        for i in range(1, len(sentences)):
            sim = self._cosine(embeddings[i - 1], embeddings[i])
            candidate = current_text + " " + sentences[i]

            if sim >= self.threshold and len(candidate) <= self.max_chunk_size:
                groups[-1].append(sentences[i])
                current_text = candidate
            else:
                groups.append([sentences[i]])
                current_text = sentences[i]

        return groups

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
