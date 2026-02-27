from __future__ import annotations

from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

PIPELINE_DEPS_AVAILABLE = importlib.util.find_spec("langchain_core") is not None


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=Path("./data"),
        log_pipeline=False,
        use_reranker=True,
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


def _doc():
    from langchain_core.documents import Document

    return Document(
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


class _FakeRetriever:
    def __init__(self, docs: list) -> None:
        self._docs = list(docs)

    def invoke(self, _: str) -> list:
        return list(self._docs)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeStreamingChain:
    def stream(self, *_args, **_kwargs):
        yield _FakeMessage("First chunk")
        yield _FakeMessage(" second chunk")


class _FakeInvokeChain:
    def invoke(self, *_args, **_kwargs):
        return _FakeMessage("Answer with PMID 12345.")


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _RecordingExecutor:
    def __init__(self, submitted: list[object]) -> None:
        self.submitted = submitted

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        self.submitted.append(fn)
        return _ImmediateFuture(fn(*args, **kwargs))


def _collect_stream(generator) -> tuple[list[str], dict]:
    chunks: list[str] = []
    while True:
        try:
            chunks.append(next(generator))
        except StopIteration as stop:
            return chunks, dict(stop.value or {})


@skipUnless(PIPELINE_DEPS_AVAILABLE, "langchain_core is not installed")
class PipelineEnhancementTests(TestCase):
    def _invoke_with_patches(self, *, include_paper_links: bool) -> dict:
        from src.core.pipeline import _invoke_chat_impl
        from src.core.scope import ScopeResult

        docs = [_doc()]
        retriever = _FakeRetriever(docs)
        config = _config()
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
            "retrieval_ms": 12.5,
        }

        with (
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.8}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline._prepare_chat_context", return_value=context),
            patch("src.core.pipeline.build_rag_chain", return_value=object()),
            patch("src.core.pipeline.build_chat_chain", return_value=_FakeInvokeChain()),
            patch("src.core.pipeline.log_llm_usage", return_value={}),
            patch("src.core.pipeline._run_optional_validation", return_value={}),
            patch("src.core.pipeline._log_request_success"),
            patch("src.core.pipeline.start_span", side_effect=lambda *args, **kwargs: nullcontext()),
        ):
            return _invoke_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=4,
                config=config,
                request_id="req-links",
                include_paper_links=include_paper_links,
                start_time=0.0,
            )

    def test_stream_chat_impl_yields_multiple_chunks(self) -> None:
        from src.core.pipeline import _stream_chat_impl
        from src.core.scope import ScopeResult

        docs = [_doc()]
        retriever = _FakeRetriever(docs)
        config = _config()
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
            "retrieval_ms": 12.5,
        }

        with (
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.8}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline._prepare_chat_context", return_value=context),
            patch("src.core.pipeline.build_rag_chain", return_value=object()),
            patch("src.core.pipeline.build_chat_chain", return_value=_FakeStreamingChain()),
            patch("src.core.pipeline.log_llm_usage", return_value={}),
            patch("src.core.pipeline._run_optional_validation", return_value={}),
            patch("src.core.pipeline._log_request_success"),
            patch("src.core.pipeline.start_span", side_effect=lambda *args, **kwargs: nullcontext()),
        ):
            stream = _stream_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=4,
                config=config,
                request_id="req-stream",
                include_paper_links=False,
                start_time=0.0,
            )
            chunks, payload = _collect_stream(stream)

        self.assertEqual(chunks, ["First chunk", " second chunk"])
        self.assertEqual(payload["answer"], "First chunk second chunk")
        self.assertEqual(payload["request_id"], "req-stream")
        self.assertNotIn("doi", payload["sources"][0])
        self.assertNotIn("fulltext_url", payload["sources"][0])

    def test_prepare_chat_context_submits_parallel_retrieval_work(self) -> None:
        from src.core.pipeline import _prepare_chat_context
        from src.core.scope import ScopeResult

        config = _config()
        submitted: list[object] = []
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        docs = [_doc()]
        records = [
            {
                "pmid": "12345",
                "title": "Trial evidence",
                "journal": "Test Journal",
                "year": "2024",
            }
        ]
        fake_cache_store = object()
        fake_abstract_store = object()

        def executor_factory(**_kwargs):
            return _RecordingExecutor(submitted)

        with (
            patch("src.core.pipeline.get_query_cache_store", return_value=fake_cache_store) as mock_cache_store,
            patch("src.core.pipeline.get_abstract_store", return_value=fake_abstract_store) as mock_abstract_store,
            patch("src.core.pipeline._prepare_reranker_resources", return_value=object()) as mock_reranker,
            patch("src.core.pipeline.lookup_query_result_cache", return_value=None),
            patch("src.core.pipeline.rewrite_to_pubmed_query", return_value="trial evidence"),
            patch(
                "src.core.pipeline._fetch_pubmed_records",
                return_value=(["12345"], records, docs),
            ) as mock_fetch_records,
            patch("src.core.pipeline.remember_query_result"),
            patch("src.core.pipeline.upsert_abstracts", return_value=1),
            patch("src.core.pipeline.build_contextual_retrieval_query", return_value="trial evidence"),
            patch("src.core.pipeline._build_retriever", return_value=(object(), False)),
        ):
            result = _prepare_chat_context(
                query="sample question",
                session_id="session-1",
                top_n=4,
                llm=object(),
                scope=scope,
                config=config,
                request_id="req-prepare",
                executor_factory=executor_factory,
            )

        self.assertIn(mock_cache_store, submitted)
        self.assertIn(mock_abstract_store, submitted)
        self.assertIn(mock_reranker, submitted)
        self.assertIn(mock_fetch_records, submitted)
        self.assertEqual(result["cache_status"], "miss")
        self.assertEqual(result["abstracts_fetched"], 1)

    def test_invoke_chat_impl_respects_show_papers_toggle(self) -> None:
        with_links = self._invoke_with_patches(include_paper_links=True)
        without_links = self._invoke_with_patches(include_paper_links=False)

        self.assertIn("doi", with_links["sources"][0])
        self.assertIn("fulltext_url", with_links["sources"][0])
        self.assertNotIn("doi", without_links["sources"][0])
        self.assertNotIn("fulltext_url", without_links["sources"][0])
