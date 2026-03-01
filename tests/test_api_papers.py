# Created by Codex - Section 1

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.dependencies import get_config
from api.main import app
from src.papers.store import PaperContent


class ApiPapersTests(TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    @patch("api.routers.papers.fetch_paper_content")
    def test_papers_endpoint_fetches_and_saves_when_cache_misses(self, mock_fetch) -> None:
        with TemporaryDirectory() as temp_dir:
            class _Config:
                data_dir = Path(temp_dir)

            mock_fetch.return_value = PaperContent(
                pmid="12345678",
                doi="10.1000/test",
                fulltext_url="https://doi.org/10.1000/test",
                title="Test paper",
                authors=["A. Author"],
                year="2024",
                journal="Test Journal",
                pubmed_url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                abstract="Abstract",
                full_text="",
                content_tier="abstract_only",
                source_label="PUBMED_ABSTRACT",
                fetched_at="2026-01-01T00:00:00+00:00",
            )
            app.dependency_overrides[get_config] = lambda: _Config()

            client = TestClient(app)
            response = client.get("/api/papers/12345678")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["pmid"], "12345678")
            self.assertTrue((Path(temp_dir) / "papers" / "cache" / "12345678.json").exists())
