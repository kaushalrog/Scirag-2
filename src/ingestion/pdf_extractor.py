"""
src/ingestion/pdf_extractor.py
-------------------------------
PDF text extraction with section detection using PyMuPDF (fitz).
Returns structured document representation with per-section text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Common academic section headers
SECTION_PATTERNS = re.compile(
    r"^\s*(\d+[\.\s]+)?(abstract|introduction|related\s+work|background|"
    r"methodology|method|approach|experiments?|results?|evaluation|"
    r"discussion|conclusion|references?|appendix)\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class DocumentSection:
    title: str
    text: str
    page_start: int
    page_end: int
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class ExtractedDocument:
    """Full extraction result for a single PDF."""

    source_path: str
    total_pages: int
    full_text: str
    sections: list[DocumentSection]
    metadata: dict

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())

    @property
    def section_titles(self) -> list[str]:
        return [s.title for s in self.sections]

    def get_section(self, name: str) -> str:
        """Return text of the first section whose title contains `name`."""
        name_lower = name.lower()
        for sec in self.sections:
            if name_lower in sec.title.lower():
                return sec.text
        return ""


class PDFExtractor:
    """
    Robust academic PDF extractor.

    Features
    --------
    - Page-level text extraction with PyMuPDF
    - Heuristic section boundary detection
    - Noise filtering (headers, footers, figure captions)
    - Metadata extraction (title, author from PDF XMP/DocInfo)
    """

    MIN_SECTION_CHARS = 100  # ignore sections shorter than this

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        pages_text: list[tuple[int, str]] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # type: ignore[arg-type]
            cleaned = self._clean_page(text)
            if cleaned:
                pages_text.append((page_num + 1, cleaned))

        full_text = "\n".join(t for _, t in pages_text)
        sections = self._detect_sections(pages_text)
        metadata = self._extract_metadata(doc, pdf_path)
        doc.close()

        logger.info(
            "Extracted %s — %d pages, %d words, %d sections",
            pdf_path.name,
            len(pages_text),
            len(full_text.split()),
            len(sections),
        )
        return ExtractedDocument(
            source_path=str(pdf_path),
            total_pages=len(pages_text),
            full_text=full_text,
            sections=sections,
            metadata=metadata,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _clean_page(self, raw: str) -> str:
        """Remove common PDF noise: URLs, lone numbers, repeated whitespace."""
        lines = raw.splitlines()
        filtered = []
        for line in lines:
            stripped = line.strip()
            # Skip page numbers, empty lines, very short lines
            if not stripped or len(stripped) < 3:
                continue
            # Skip pure digit lines (page numbers)
            if stripped.isdigit():
                continue
            # Skip DOI / URL lines
            if stripped.lower().startswith(("doi:", "http", "arxiv")):
                continue
            filtered.append(stripped)
        return " ".join(filtered)

    def _detect_sections(
        self, pages_text: list[tuple[int, str]]
    ) -> list[DocumentSection]:
        full_text = "\n".join(f"[PAGE:{p}] {t}" for p, t in pages_text)
        sections: list[DocumentSection] = []
        boundaries: list[tuple[int, str]] = []

        for match in SECTION_PATTERNS.finditer(full_text):
            boundaries.append((match.start(), match.group().strip()))

        for i, (start, title) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            chunk = full_text[start:end].strip()

            # Determine page range from PAGE markers
            pages_in_chunk = [
                int(m.group(1))
                for m in re.finditer(r"\[PAGE:(\d+)\]", chunk)
            ]
            page_start = pages_in_chunk[0] if pages_in_chunk else 1
            page_end = pages_in_chunk[-1] if pages_in_chunk else page_start

            # Remove internal PAGE markers
            clean_chunk = re.sub(r"\[PAGE:\d+\]\s*", "", chunk)
            if len(clean_chunk) >= self.MIN_SECTION_CHARS:
                sections.append(
                    DocumentSection(
                        title=title,
                        text=clean_chunk,
                        page_start=page_start,
                        page_end=page_end,
                    )
                )

        return sections

    @staticmethod
    def _extract_metadata(doc: fitz.Document, path: Path) -> dict:
        meta = doc.metadata or {}
        return {
            "filename": path.name,
            "pdf_title": meta.get("title", ""),
            "pdf_author": meta.get("author", ""),
            "pdf_subject": meta.get("subject", ""),
            "num_pages": doc.page_count,
        }
