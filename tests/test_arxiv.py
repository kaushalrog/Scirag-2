"""tests/test_arxiv.py — Unit tests for arXiv client."""

import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.arxiv_client import ArxivClient, ArxivPaper


class TestArxivPaper:
    def test_content_hash_deterministic(self):
        p = ArxivPaper(
            arxiv_id="2301.00001",
            title="Test Paper",
            authors=["Author A"],
            abstract="Abstract text",
            categories=["cs.AI"],
            published="2023-01-01",
            url="https://arxiv.org/abs/2301.00001",
            pdf_url="",
        )
        p2 = ArxivPaper(
            arxiv_id="2301.00001",
            title="Test Paper",
            authors=["Author A"],
            abstract="Abstract text",
            categories=["cs.AI"],
            published="2023-01-01",
            url="https://arxiv.org/abs/2301.00001",
            pdf_url="",
        )
        assert p.content_hash == p2.content_hash

    def test_content_hash_differs_on_content(self):
        base = ArxivPaper(
            arxiv_id="2301.00001", title="T", authors=[], abstract="AAA",
            categories=[], published="2023-01-01", url="", pdf_url="",
        )
        diff = ArxivPaper(
            arxiv_id="2301.00001", title="T", authors=[], abstract="BBB",
            categories=[], published="2023-01-01", url="", pdf_url="",
        )
        assert base.content_hash != diff.content_hash

    def test_to_metadata_keys(self):
        p = ArxivPaper(
            arxiv_id="2301.00001", title="T", authors=["A", "B"],
            abstract="abs", categories=["cs.AI"], published="2023-01-01",
            url="http://x", pdf_url="",
        )
        meta = p.to_metadata()
        assert "arxiv_id" in meta
        assert "title" in meta
        assert "source" in meta
        assert meta["source"] == "arxiv"
        assert meta["authors"] == "A; B"


class TestArxivClient:
    def test_throttle_enforces_rate_limit(self, tmp_path):
        import time
        client = ArxivClient(download_dir=tmp_path, rate_limit=10)
        client._last_call = time.monotonic()
        # Should not raise; just tests the method exists and runs
        client._throttle()

    def test_to_paper_parses_result(self):
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2301.00001v1"
        mock_result.title = "  Test Paper  "
        mock_result.authors = [MagicMock(name="Author A")]
        mock_result.summary = "An abstract."
        mock_result.categories = ["cs.AI", "cs.LG"]
        mock_result.published = MagicMock()
        mock_result.published.date.return_value = MagicMock(isoformat=lambda: "2023-01-01")
        mock_result.pdf_url = "http://arxiv.org/pdf/2301.00001"

        paper = ArxivClient._to_paper(mock_result)
        assert paper.arxiv_id == "2301.00001v1"
        assert paper.title == "Test Paper"
        assert "cs.AI" in paper.categories
