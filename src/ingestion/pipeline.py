"""
src/ingestion/pipeline.py
--------------------------
Orchestrates end-to-end ingestion:
  arXiv fetch → PDF download → text extraction → S2 enrichment →
  metadata assembly → vector store upsert.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .arxiv_client import ArxivClient, ArxivPaper
from .pdf_extractor import ExtractedDocument, PDFExtractor
from .semantic_scholar import S2Paper, SemanticScholarClient

logger = logging.getLogger(__name__)


@dataclass
class IngestedDocument:
    """Fully enriched document ready for chunking and embedding."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    tldr: str
    published: str
    categories: list[str]
    url: str
    citation_count: int
    influential_citations: int
    full_text: str
    sections: list[dict]        # [{title, text, page_start, page_end}]
    source: str = "arxiv"

    def to_metadata(self) -> dict:
        """Flat metadata dict suitable for ChromaDB."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title[:200],
            "authors": "; ".join(self.authors[:5]),
            "abstract": self.abstract[:500],
            "tldr": self.tldr[:300],
            "published": self.published,
            "categories": "; ".join(self.categories),
            "url": self.url,
            "citation_count": self.citation_count,
            "influential_citations": self.influential_citations,
            "source": self.source,
        }


class IngestionPipeline:
    """
    Full ingestion pipeline.

    Parameters
    ----------
    download_dir  : Directory for raw PDFs
    processed_dir : Directory to save JSON snapshots of ingested docs
    s2_api_key    : Optional Semantic Scholar API key
    """

    def __init__(
        self,
        download_dir: str = "./data/raw",
        processed_dir: str = "./data/processed",
        s2_api_key: str = "",
    ) -> None:
        self.arxiv = ArxivClient(download_dir=download_dir)
        self.extractor = PDFExtractor()
        self.s2 = SemanticScholarClient(api_key=s2_api_key)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ───────────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        max_papers: int = 50,
        download_pdfs: bool = True,
        enrich_s2: bool = True,
    ) -> list[IngestedDocument]:
        """
        Full pipeline run.

        Returns list of IngestedDocument ready for the vector store.
        """
        logger.info("Ingestion pipeline START — query='%s', max=%d", query, max_papers)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            transient=True,
        ) as progress:
            # Step 1: arXiv fetch
            task = progress.add_task("Fetching from arXiv…", total=None)
            papers = self.arxiv.search(
                query=query, max_results=max_papers, download_pdfs=download_pdfs
            )
            progress.update(task, completed=True)

            # Step 2: S2 enrichment
            s2_map: dict[str, S2Paper] = {}
            if enrich_s2:
                task2 = progress.add_task("Enriching via Semantic Scholar…", total=None)
                arxiv_ids = [p.arxiv_id for p in papers]
                s2_map = self.s2.enrich_papers(arxiv_ids)
                progress.update(task2, completed=True)

            # Step 3: PDF extraction + assembly
            docs: list[IngestedDocument] = []
            task3 = progress.add_task("Extracting text…", total=len(papers))
            for paper in papers:
                doc = self._assemble(paper, s2_map.get(paper.arxiv_id))
                docs.append(doc)
                self._save_snapshot(doc)
                progress.advance(task3)

        logger.info("Ingestion pipeline DONE — %d documents", len(docs))
        return docs

    def ingest_local_pdf(self, pdf_path: str, metadata: dict | None = None) -> IngestedDocument:
        """Ingest a local PDF file (not from arXiv)."""
        extracted: ExtractedDocument = self.extractor.extract(pdf_path)
        meta = metadata or {}
        return IngestedDocument(
            arxiv_id=meta.get("arxiv_id", Path(pdf_path).stem),
            title=meta.get("title", extracted.metadata.get("pdf_title", Path(pdf_path).stem)),
            authors=meta.get("authors", []),
            abstract=meta.get("abstract", ""),
            tldr="",
            published=meta.get("published", ""),
            categories=meta.get("categories", ["local"]),
            url=meta.get("url", ""),
            citation_count=0,
            influential_citations=0,
            full_text=extracted.full_text,
            sections=[
                {
                    "title": s.title,
                    "text": s.text,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                }
                for s in extracted.sections
            ],
            source="local",
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _assemble(self, paper: ArxivPaper, s2: S2Paper | None) -> IngestedDocument:
        # Extract text if PDF was downloaded
        sections = []
        full_text = paper.abstract
        if paper.local_pdf_path:
            try:
                extracted = self.extractor.extract(paper.local_pdf_path)
                full_text = extracted.full_text
                sections = [
                    {
                        "title": s.title,
                        "text": s.text,
                        "page_start": s.page_start,
                        "page_end": s.page_end,
                    }
                    for s in extracted.sections
                ]
            except Exception as exc:
                logger.warning("PDF extraction failed %s: %s", paper.arxiv_id, exc)

        return IngestedDocument(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            tldr=s2.tldr if s2 else "",
            published=paper.published,
            categories=paper.categories,
            url=paper.url,
            citation_count=s2.citation_count if s2 else 0,
            influential_citations=s2.influential_citation_count if s2 else 0,
            full_text=full_text,
            sections=sections,
        )

    def _save_snapshot(self, doc: IngestedDocument) -> None:
        out = self.processed_dir / f"{doc.arxiv_id}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(asdict(doc), fh, indent=2, ensure_ascii=False)
