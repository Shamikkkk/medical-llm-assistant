from __future__ import annotations

import json
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

PIPELINE_DEPS_AVAILABLE = importlib.util.find_spec("langchain_core") is not None


class _FakeRetriever:
    def __init__(self, docs: list) -> None:
        self._docs = list(docs)

    def invoke(self, _: str) -> list:
        return list(self._docs)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeInvokeChain:
    def invoke(self, *_args, **_kwargs):
        return _FakeMessage("Answer with PMID 12345.")


@skipUnless(PIPELINE_DEPS_AVAILABLE, "langchain_core is not installed")
class PipelineGoldenTests(TestCase):
    def test_pipeline_payload_matches_fixture(self) -> None:
        from langchain_core.documents import Document
        from src.core.pipeline import _invoke_chat_impl
        from src.core.scope import ScopeResult

        config = SimpleNamespace(
            log_pipeline=False,
            use_reranker=False,
            max_abstracts=8,
            max_context_tokens=2500,
            context_trim_strategy="truncate",
            hybrid_retrieval=False,
            hybrid_alpha=0.5,
            citation_alignment=False,
            alignment_mode="disclaim",
            validator_enabled=False,
            metrics_mode=False,
            metrics_store_path=Path("./data/metrics/events.jsonl"),
            nvidia_api_key="test-key",
        )
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        docs = [
            Document(
                page_content="Trial evidence\n\nTrial evidence abstract showing reduced stroke risk with therapy.",
                metadata={
                    "pmid": "12345",
                    "title": "Trial evidence",
                    "journal": "Test Journal",
                    "year": "2024",
                    "doi": "10.1000/12345",
                    "fulltext_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                },
            )
        ]
        context = {
            "llm": object(),
            "retriever": _FakeRetriever(docs),
            "reranker_active": False,
            "pubmed_query": "trial evidence",
            "retrieval_query": "trial evidence",
            "reframe_note": "",
            "docs_preview": [
                {
                    "rank": 1,
                    "pmid": "12345",
                    "title": "Trial evidence",
                    "journal": "Test Journal",
                    "year": "2024",
                    "doi": "10.1000/12345",
                    "fulltext_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                }
            ],
            "cache_hit": False,
            "cache_status": "miss",
            "retrieval_ms": 12.5,
        }

        with (
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.93}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline._prepare_chat_context", return_value=context),
            patch("src.core.pipeline.build_rag_chain", return_value=object()),
            patch("src.core.pipeline.build_chat_chain", return_value=_FakeInvokeChain()),
            patch("src.core.pipeline.log_llm_usage", return_value={}),
            patch(
                "src.core.pipeline._run_optional_validation",
                return_value={"validation_warning": "support review pending"},
            ),
            patch("src.core.pipeline._log_request_success"),
            patch("src.core.pipeline.start_span", side_effect=lambda *args, **kwargs: nullcontext()),
        ):
            payload = _invoke_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=4,
                config=config,
                request_id="req-123",
                include_paper_links=True,
                start_time=0.0,
            )

        fixture_path = Path(__file__).parent / "fixtures" / "pipeline_payload_expected.json"
        expected = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(payload, expected)
