# Created by Codex - Section 1

from __future__ import annotations

from unittest import TestCase

from fastapi.testclient import TestClient

from api.dependencies import get_config
from api.main import app


class ApiConfigTests(TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_config_endpoint_returns_masked_summary(self) -> None:
        class _Config:
            def masked_summary(self) -> dict:
                return {"app_title": "Test App", "nvidia_api_key": "NOT_SET"}

        app.dependency_overrides[get_config] = lambda: _Config()

        client = TestClient(app)
        response = client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["app_title"], "Test App")
