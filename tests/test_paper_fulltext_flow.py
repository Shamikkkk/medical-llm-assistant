from __future__ import annotations

import importlib.util
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch

REQUESTS_AVAILABLE = importlib.util.find_spec("requests") is not None


@skipUnless(REQUESTS_AVAILABLE, "requests dependency is not installed")
class PaperFullTextFlowTests(TestCase):
    @patch("src.papers.doi.requests.head")
    def test_doi_resolution_returns_redirect_target(self, mock_head: Mock) -> None:
        from src.papers.doi import resolve_doi_url

        mock_head.return_value = Mock(ok=True, url="https://publisher.example.org/article/123")
        resolved = resolve_doi_url("10.1234/abcd.1")
        self.assertEqual(resolved, "https://publisher.example.org/article/123")

    @patch("src.papers.fetch.fetch_pmc_full_text")
    @patch("src.papers.fetch.pubmed_efetch")
    def test_fetch_paper_content_uses_pmc_path_when_available(
        self,
        mock_efetch: Mock,
        mock_fetch_pmc: Mock,
    ) -> None:
        from src.papers.fetch import fetch_paper_content
        from src.papers.store import PAPER_TIER_FULL_TEXT

        mock_efetch.return_value = [
            {
                "pmid": "123456",
                "title": "Sample paper",
                "abstract": "Abstract text",
                "journal": "Journal",
                "year": "2025",
                "authors": ["A Author"],
                "doi": "10.1000/sample",
                "pmcid": "PMC1234567",
                "fulltext_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
            }
        ]
        mock_fetch_pmc.return_value = "Full text from PMC."
        paper = fetch_paper_content("123456")
        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.content_tier, PAPER_TIER_FULL_TEXT)
        self.assertEqual(paper.source_label, "PMC")
        self.assertIn("Full text from PMC", paper.full_text)

    @patch("src.papers.fetch_fulltext.requests.get")
    def test_link_ingestion_reports_paywalled_without_crashing(self, mock_get: Mock) -> None:
        from src.papers.fetch_fulltext import (
            STATUS_PAYWALLED_OR_BLOCKED,
            fetch_readable_text_from_url,
        )

        mock_get.return_value = Mock(
            status_code=200,
            headers={"Content-Type": "text/html"},
            text="<html><body>Please purchase this article to continue.</body></html>",
            url="https://example.org/paywalled",
        )
        text, meta, status = fetch_readable_text_from_url("https://example.org/paywalled")
        self.assertEqual(text, "")
        self.assertEqual(status, STATUS_PAYWALLED_OR_BLOCKED)
        self.assertEqual(meta.get("status_code"), 200)
