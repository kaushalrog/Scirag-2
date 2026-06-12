"""
src/ingestion/arxiv_client.py
------------------------------
arXiv API client with rate limiting, retry logic, and deduplication.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import arxiv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Structured representation of a fetched arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str          # ISO date string
    url: str
    pdf_url: str
    local_pdf_path: str = ""
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        raw = f"{self.arxiv_id}{self.title}{self.abstract}"
        self.content_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_metadata(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": "; ".join(self.authors[:5]),
            "abstract": self.abstract[:500],
            "categories": "; ".join(self.categories),
            "published": self.published,
            "url": self.url,
            "source": "arxiv",
        }


class ArxivClient:
    """
    Thin wrapper around the `arxiv` Python library.

    Features
    --------
    - Configurable rate limiting (default 3 req/s, arxiv recommends ≤ 3)
    - Exponential-backoff retry on transient errors
    - SHA-256 content hashing for deduplication
    - Optional PDF download with path tracking
    """

    DEFAULT_RATE_LIMIT = 3  # requests per second

    def __init__(
        self,
        download_dir: str | Path = "./data/raw",
        rate_limit: float = DEFAULT_RATE_LIMIT,
    ) -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = 1.0 / rate_limit
        self._last_call = 0.0
        self._client = arxiv.Client(
            page_size=100,
            delay_seconds=self._min_interval,
            num_retries=3,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        max_results: int = 50,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        download_pdfs: bool = False,
    ) -> list[ArxivPaper]:
        """
        Search arXiv and return a list of ArxivPaper objects.

        Parameters
        ----------
        query        : arXiv search string (supports field prefixes like ti:, abs:)
        max_results  : Upper bound on papers to fetch
        sort_by      : arxiv.SortCriterion enum value
        download_pdfs: If True, download PDFs into ``download_dir``
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )
        papers: list[ArxivPaper] = []
        seen: set[str] = set()

        for result in self._iter_with_rate_limit(search):
            paper = self._to_paper(result)
            if paper.content_hash in seen:
                logger.debug("Duplicate skipped: %s", paper.arxiv_id)
                continue
            seen.add(paper.content_hash)

            if download_pdfs:
                paper.local_pdf_path = self._download_pdf(result, paper.arxiv_id)

            papers.append(paper)
            logger.debug("Fetched: %s — %s", paper.arxiv_id, paper.title[:60])

        logger.info("arXiv search '%s' → %d papers", query, len(papers))
        return papers

    def fetch_by_id(self, arxiv_id: str, download_pdf: bool = False) -> ArxivPaper:
        """Fetch a single paper by its arXiv ID."""
        self._throttle()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(self._client.results(search))
        if not results:
            raise ValueError(f"Paper not found: {arxiv_id}")
        paper = self._to_paper(results[0])
        if download_pdf:
            paper.local_pdf_path = self._download_pdf(results[0], arxiv_id)
        return paper

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _iter_with_rate_limit(
        self, search: arxiv.Search
    ) -> Iterator[arxiv.Result]:
        for result in self._client.results(search):
            self._throttle()
            yield result

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
    def _download_pdf(self, result: arxiv.Result, arxiv_id: str) -> str:
        safe_id = arxiv_id.replace("/", "_")
        out_path = self.download_dir / f"{safe_id}.pdf"
        if out_path.exists():
            logger.debug("PDF already exists: %s", out_path)
            return str(out_path)
        result.download_pdf(dirpath=str(self.download_dir), filename=f"{safe_id}.pdf")
        logger.info("Downloaded PDF: %s", out_path)
        return str(out_path)

    @staticmethod
    def _to_paper(result: arxiv.Result) -> ArxivPaper:
        return ArxivPaper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title.strip(),
            authors=[a.name for a in result.authors],
            abstract=result.summary.strip(),
            categories=result.categories,
            published=result.published.date().isoformat(),
            url=result.entry_id,
            pdf_url=result.pdf_url or "",
        )
