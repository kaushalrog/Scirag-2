"""
src/ingestion/semantic_scholar.py
----------------------------------
Semantic Scholar Academic Graph (S2AG) API client.
Augments arXiv metadata with citation counts, influential citations,
and TLDRs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = (
    "title,authors,year,abstract,tldr,citationCount,"
    "influentialCitationCount,externalIds,openAccessPdf"
)


@dataclass
class S2Paper:
    s2_id: str
    title: str
    year: Optional[int]
    abstract: str
    tldr: str
    citation_count: int
    influential_citation_count: int
    arxiv_id: str
    pdf_url: str


class SemanticScholarClient:
    """
    Lightweight S2AG client.

    Rate limits
    -----------
    - Without API key: 1 req/s, 100/5-min
    - With API key   : 10 req/s
    """

    def __init__(self, api_key: str = "") -> None:
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["x-api-key"] = api_key
        self._rate_limit = 1.0 if not api_key else 0.1  # seconds between calls
        self._last = 0.0

    # ── Public ───────────────────────────────────────────────────────────────

    def lookup_by_arxiv_id(self, arxiv_id: str) -> Optional[S2Paper]:
        """Lookup a paper via arXiv ID."""
        try:
            data = self._get(f"{S2_BASE}/paper/arXiv:{arxiv_id}", params={"fields": FIELDS})
            return self._parse(data)
        except Exception as exc:
            logger.warning("S2 lookup failed for %s: %s", arxiv_id, exc)
            return None

    def search(self, query: str, limit: int = 20) -> list[S2Paper]:
        """Free-text search against S2AG."""
        params = {"query": query, "limit": limit, "fields": FIELDS}
        try:
            data = self._get(f"{S2_BASE}/paper/search", params=params)
            return [self._parse(p) for p in data.get("data", []) if p]
        except Exception as exc:
            logger.warning("S2 search failed for '%s': %s", query, exc)
            return []

    def enrich_papers(self, arxiv_ids: list[str]) -> dict[str, S2Paper]:
        """Batch-enrich a list of arXiv IDs. Returns dict keyed by arxiv_id."""
        enriched: dict[str, S2Paper] = {}
        for aid in arxiv_ids:
            paper = self.lookup_by_arxiv_id(aid)
            if paper:
                enriched[aid] = paper
        return enriched

    # ── Private ──────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _get(self, url: str, params: dict | None = None) -> dict:
        self._throttle()
        with httpx.Client(headers=self._headers, timeout=15.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last = time.monotonic()

    @staticmethod
    def _parse(data: dict) -> S2Paper:
        tldr = ""
        if data.get("tldr") and data["tldr"].get("text"):
            tldr = data["tldr"]["text"]
        ext = data.get("externalIds") or {}
        pdf = ""
        if data.get("openAccessPdf"):
            pdf = data["openAccessPdf"].get("url", "")
        return S2Paper(
            s2_id=data.get("paperId", ""),
            title=data.get("title", ""),
            year=data.get("year"),
            abstract=data.get("abstract") or "",
            tldr=tldr,
            citation_count=data.get("citationCount", 0),
            influential_citation_count=data.get("influentialCitationCount", 0),
            arxiv_id=ext.get("ArXiv", ""),
            pdf_url=pdf,
        )
