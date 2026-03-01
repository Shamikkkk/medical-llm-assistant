from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


class _FakeAnswerStore:
    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {"ids": [], "documents": [], "metadatas": []}

    def get(self, **_kwargs):
        return self._payload

    def similarity_search_with_relevance_scores(self, *_args, **_kwargs):
        return []


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeInvokeChain:
    def invoke(self, *_args, **_kwargs):
        return _FakeMessage("Fresh answer with PMID 12345.")


class _FakeRetriever:
    def __init__(self, docs: list) -> None:
        self._docs = list(docs)

    def invoke(self, _: str) -> list:
        return list(self._docs)


class AnswerCacheTests(TestCase):
    def test_answer_cache_fingerprint_changes_with_top_n(self) -> None:
        from src.integrations.storage import build_answer_cache_fingerprint

        config = SimpleNamespace(
            nvidia_model="model-a",
            use_reranker=False,
            hybrid_retrieval=False,
            hybrid_alpha=0.5,
            max_abstracts=8,
            max_context_tokens=2500,
            context_trim_strategy="truncate",
            citation_alignment=True,
            alignment_mode="disclaim",
            validator_enabled=False,
            validator_model_name="validator-a",
            validator_threshold=0.7,
            validator_margin=0.2,
        )

        first = build_answer_cache_fingerprint(
            config=config,
            top_n=4,
            include_paper_links=True,
            backend="baseline",
        )
        second = build_answer_cache_fingerprint(
            config=config,
            top_n=6,
            include_paper_links=True,
            backend="baseline",
        )

        self.assertNotEqual(first, second)

    def test_lookup_answer_cache_respects_ttl(self) -> None:
        from src.integrations.storage import lookup_answer_cache

        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
        response_payload = {"status": "answered", "answer": "Cached answer", "sources": []}
        store = _FakeAnswerStore(
            {
                "ids": ["cache-1"],
                "documents": ["heart failure treatment"],
                "metadatas": [
                    {
                        "normalized_query": "heart failure treatment",
                        "config_fingerprint": "fp-1",
                        "created_at": created_at,
                        "response_payload_json": json.dumps(response_payload),
                    }
                ],
            }
        )

        fresh = lookup_answer_cache(
            "heart failure treatment",
            store=store,
            config_fingerprint="fp-1",
            ttl_seconds=3600,
            min_similarity=0.9,
            now_epoch=datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc).timestamp(),
        )
        expired = lookup_answer_cache(
            "heart failure treatment",
            store=store,
            config_fingerprint="fp-1",
            ttl_seconds=3600,
            min_similarity=0.9,
            now_epoch=datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc).timestamp(),
        )

        self.assertIsNotNone(fresh)
        self.assertIsNone(expired)

    def test_pipeline_reuses_cached_answer_and_skips_context_preparation(self) -> None:
        from langchain_core.documents import Document
        from src.core.pipeline import _invoke_chat_impl
        from src.core.scope import ScopeResult

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
        retriever = _FakeRetriever(docs)
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        context = {
            "llm": object(),
            "retriever": retriever,
            "reranker_active": False,
            "pubmed_query": "trial evidence",
            "retrieval_query": "trial evidence",
            "reframe_note": "",
            "docs_preview": [],
            "cache_hit": False,
            "cache_status": "miss",
            "context_top_k": 4,
            "retrieval_ms": 12.5,
        }
        config = SimpleNamespace(
            data_dir=Path("./data"),
            log_pipeline=False,
            use_reranker=False,
            max_abstracts=8,
            max_context_abstracts=4,
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
            nvidia_model="model-a",
            answer_cache_ttl_seconds=604800,
            answer_cache_min_similarity=0.9,
            answer_cache_strict_fingerprint=True,
        )

        cached_answer = {
            "response_payload": {
                "status": "answered",
                "answer": "Cached answer with PMID 12345.",
                "sources": [{"rank": 1, "pmid": "12345", "title": "Trial evidence"}],
                "retrieved_contexts": [],
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "matched_query": "sample question",
            "similarity": 0.98,
            "match_type": "similar",
            "config_match": True,
        }

        with (
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.8}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline.get_answer_cache_store", return_value=object()),
            patch("src.core.pipeline.build_answer_cache_fingerprint", return_value="fp-1"),
            patch("src.core.pipeline.lookup_answer_cache", side_effect=[None, cached_answer]),
            patch("src.core.pipeline._prepare_chat_context", return_value=context) as mock_prepare_context,
            patch("src.core.pipeline.build_rag_chain", return_value=object()),
            patch("src.core.pipeline.build_chat_chain", return_value=_FakeInvokeChain()),
            patch("src.core.pipeline.log_llm_usage", return_value={}),
            patch("src.core.pipeline.store_answer_cache"),
            patch("src.core.pipeline._run_optional_validation", return_value={}),
            patch("src.core.pipeline._log_request_success"),
            patch("src.core.pipeline.start_span", side_effect=lambda *args, **kwargs: nullcontext()),
        ):
            first = _invoke_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=4,
                config=config,
                request_id="req-fresh",
                include_paper_links=True,
                start_time=1.0,
            )
            second = _invoke_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=4,
                config=config,
                request_id="req-cached",
                include_paper_links=True,
                start_time=1.0,
            )

        self.assertEqual(first["answer"], "Fresh answer with PMID 12345.")
        self.assertEqual(second["answer"], "Cached answer with PMID 12345.")
        self.assertTrue(second["answer_cache_hit"])
        self.assertEqual(mock_prepare_context.call_count, 1)
