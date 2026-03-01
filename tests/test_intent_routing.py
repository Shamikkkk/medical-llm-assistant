from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from unittest import TestCase, skipUnless
from unittest.mock import patch

PIPELINE_DEPS_AVAILABLE = importlib.util.find_spec("langchain_core") is not None


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=Path("./data"),
        log_pipeline=False,
        use_reranker=False,
        max_abstracts=8,
        max_context_abstracts=8,
        max_context_tokens=2500,
        context_trim_strategy="truncate",
        hybrid_retrieval=False,
        hybrid_alpha=0.5,
        citation_alignment=False,
        alignment_mode="disclaim",
        validator_enabled=False,
        metrics_mode=False,
        metrics_store_path=Path("./data/metrics/events.jsonl"),
        nvidia_api_key=None,
    )


class IntentRoutingTests(TestCase):
    def test_smoking_typo_query_forces_medical_intent(self) -> None:
        from src.intent import classify_intent_details

        result = classify_intent_details("how to quti smoking?", llm=None)

        self.assertEqual(result["label"], "medical")
        self.assertGreaterEqual(float(result["confidence"]), 0.9)

    def test_smokign_typo_query_forces_medical_intent(self) -> None:
        from src.intent import classify_intent_details

        result = classify_intent_details("how to quit smokign", llm=None)

        self.assertEqual(result["label"], "medical")
        self.assertGreaterEqual(float(result["confidence"]), 0.9)

    def test_smoking_cessation_query_is_in_scope(self) -> None:
        from src.core.scope import classify_scope

        result = classify_scope(
            "how to quit smoking",
            session_id="scope-smoking",
            llm=None,
        )

        self.assertTrue(result.allow)
        self.assertEqual(result.label, "BIOMEDICAL")

    def test_hi_remains_smalltalk(self) -> None:
        from src.intent import classify_intent_details, should_short_circuit_smalltalk

        details = classify_intent_details("hi", llm=None)

        self.assertEqual(details["label"], "smalltalk")
        self.assertTrue(should_short_circuit_smalltalk(details, "hi"))

    def test_low_confidence_smalltalk_does_not_short_circuit_medical_query(self) -> None:
        from src.intent import should_short_circuit_smalltalk

        self.assertFalse(
            should_short_circuit_smalltalk(
                {"label": "smalltalk", "confidence": 0.45},
                "pneumonia antibiotic duration",
            )
        )


@skipUnless(PIPELINE_DEPS_AVAILABLE, "langchain_core is not installed")
class PipelineIntentRoutingTests(TestCase):
    def test_pipeline_routes_typo_smoking_query_to_retrieval(self) -> None:
        from src.core.pipeline import _invoke_chat_impl

        context = {
            "llm": None,
            "retriever": object(),
            "reranker_active": False,
            "pubmed_query": "quit smoking",
            "retrieval_query": "quit smoking",
            "reframe_note": "",
            "docs_preview": [],
            "cache_hit": False,
            "cache_status": "miss",
            "context_top_k": 8,
            "retrieval_ms": 5.0,
        }

        with (
            patch("src.core.pipeline._get_llm_safe", return_value=None),
            patch("src.core.pipeline._prepare_chat_context", return_value=context) as mock_prepare_context,
            patch("src.core.pipeline._log_request_success"),
        ):
            payload = _invoke_chat_impl(
                query="how to quti smoking?",
                session_id="session-smoking",
                top_n=10,
                config=_config(),
                request_id="req-smoking",
                include_paper_links=True,
                start_time=0.0,
            )

        self.assertEqual(payload["status"], "answered")
        self.assertNotEqual(payload["intent_label"], "smalltalk")
        self.assertTrue(mock_prepare_context.called)

    def test_pipeline_routes_smokign_typo_query_to_retrieval(self) -> None:
        from src.core.pipeline import _invoke_chat_impl

        context = {
            "llm": None,
            "retriever": object(),
            "reranker_active": False,
            "pubmed_query": "quit smoking",
            "retrieval_query": "quit smoking",
            "reframe_note": "",
            "docs_preview": [],
            "cache_hit": False,
            "cache_status": "miss",
            "context_top_k": 8,
            "retrieval_ms": 5.0,
        }

        with (
            patch("src.core.pipeline._get_llm_safe", return_value=None),
            patch("src.core.pipeline._prepare_chat_context", return_value=context) as mock_prepare_context,
            patch("src.core.pipeline._log_request_success"),
        ):
            payload = _invoke_chat_impl(
                query="how to quit smokign",
                session_id="session-smoking-2",
                top_n=10,
                config=_config(),
                request_id="req-smoking-2",
                include_paper_links=True,
                start_time=0.0,
            )

        self.assertEqual(payload["status"], "answered")
        self.assertNotEqual(payload["intent_label"], "smalltalk")
        self.assertTrue(mock_prepare_context.called)

    def test_pipeline_typo_smoking_query_returns_pubmed_preview(self) -> None:
        from langchain_core.documents import Document
        from src.core.pipeline import _invoke_chat_impl
        from src.core.scope import ScopeResult

        records = [
            {
                "pmid": str(30000 + index),
                "title": f"Smoking cessation trial {index}",
                "journal": "Respiratory Journal",
                "year": "2025",
                "doi": f"10.1000/{30000 + index}",
                "fulltext_url": f"https://pubmed.ncbi.nlm.nih.gov/{30000 + index}/",
            }
            for index in range(10)
        ]
        docs = [
            Document(
                page_content=f"Smoking cessation trial {index}\n\nNicotine replacement evidence.",
                metadata={
                    "pmid": record["pmid"],
                    "title": record["title"],
                    "journal": record["journal"],
                    "year": record["year"],
                    "doi": record["doi"],
                    "fulltext_url": record["fulltext_url"],
                },
            )
            for index, record in enumerate(records)
        ]
        pmids = [record["pmid"] for record in records]
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)

        with (
            patch("src.core.pipeline._get_llm_safe", return_value=None),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline.get_query_cache_store", return_value=object()),
            patch("src.core.pipeline.get_abstract_store", return_value=object()),
            patch("src.core.pipeline._prepare_reranker_resources", return_value=None),
            patch("src.core.pipeline.lookup_query_result_cache", return_value=None),
            patch("src.core.pipeline.pubmed_esearch", return_value=pmids) as mock_esearch,
            patch("src.core.pipeline.pubmed_efetch", return_value=records),
            patch("src.core.pipeline.to_documents", return_value=docs),
            patch("src.core.pipeline.remember_query_result"),
            patch("src.core.pipeline.upsert_abstracts", return_value=10),
            patch("src.core.pipeline.build_contextual_retrieval_query", return_value="quit smoking"),
            patch("src.core.pipeline._build_retriever", return_value=(object(), False)),
            patch("src.core.pipeline._log_request_success"),
            patch("src.core.pipeline.start_span", side_effect=lambda *args, **kwargs: nullcontext()),
        ):
            payload = _invoke_chat_impl(
                query="how to quti smoking",
                session_id="session-smoking-3",
                top_n=10,
                config=_config(),
                request_id="req-smoking-3",
                include_paper_links=True,
                start_time=0.0,
            )

        self.assertEqual(payload["status"], "answered")
        self.assertNotEqual(payload["intent_label"], "smalltalk")
        self.assertIn("quit smoking", payload["pubmed_query"])
        self.assertEqual(len(payload["docs_preview"]), 10)
        self.assertGreaterEqual(mock_esearch.call_args.kwargs["retmax"], 10)
