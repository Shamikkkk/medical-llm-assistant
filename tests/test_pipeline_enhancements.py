from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

PIPELINE_DEPS_AVAILABLE = importlib.util.find_spec("langchain_core") is not None


def _config(
    *,
    use_reranker: bool = True,
    hybrid_retrieval: bool = False,
    max_abstracts: int = 8,
    max_context_abstracts: int | None = None,
    multi_strategy_retrieval: bool = True,
    retrieval_candidate_multiplier: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=Path("./data"),
        log_pipeline=False,
        use_reranker=use_reranker,
        max_abstracts=max_abstracts,
        max_context_abstracts=max_abstracts if max_context_abstracts is None else max_context_abstracts,
        max_context_tokens=2500,
        context_trim_strategy="truncate",
        multi_strategy_retrieval=multi_strategy_retrieval,
        retrieval_candidate_multiplier=retrieval_candidate_multiplier,
        hybrid_retrieval=hybrid_retrieval,
        hybrid_alpha=0.5,
        citation_alignment=False,
        alignment_mode="disclaim",
        validator_enabled=False,
        metrics_mode=False,
        metrics_store_path=Path("./data/metrics/events.jsonl"),
        nvidia_api_key="test-key",
    )


def _doc():
    return _make_doc(12345)


def _make_doc(pmid: int):
    from langchain_core.documents import Document

    return Document(
        page_content=(
            f"Trial evidence {pmid}\n\n"
            f"Trial evidence abstract showing reduced stroke risk with therapy for PMID {pmid}."
        ),
        metadata={
            "pmid": str(pmid),
            "title": f"Trial evidence {pmid}",
            "journal": "Test Journal",
            "year": "2024",
            "doi": f"10.1000/{pmid}",
            "fulltext_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        },
    )


def _make_docs(count: int) -> list:
    return [_make_doc(10000 + index) for index in range(count)]


def _make_records(count: int) -> list[dict]:
    return [
        {
            "pmid": str(10000 + index),
            "title": f"Trial evidence {10000 + index}",
            "journal": "Test Journal",
            "year": "2024",
            "doi": f"10.1000/{10000 + index}",
            "fulltext_url": f"https://pubmed.ncbi.nlm.nih.gov/{10000 + index}/",
        }
        for index in range(count)
    ]


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


class _StructuredInvokeChain:
    def invoke(self, *_args, **_kwargs):
        return _FakeMessage(
            "## Direct Answer\n"
            "Warfarin reduced stroke risk.\n\n"
            "## Evidence Summary\n"
            "- Stroke risk was lower with warfarin [PMID: 12345].\n"
            "- An unsupported citation slipped in [PMID: 99999999].\n\n"
            "## Evidence Quality\n"
            "Moderate because the evidence is observational and mixed.\n"
        )


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


class _FakeAbstractStore:
    def __init__(self, docs: list) -> None:
        self._docs = list(docs)

    def as_retriever(self, search_kwargs=None):
        del search_kwargs
        return _FakeRetriever(self._docs)


def _collect_stream(generator) -> tuple[list[str], dict]:
    chunks: list[str] = []
    while True:
        try:
            chunks.append(next(generator))
        except StopIteration as stop:
            return chunks, dict(stop.value or {})


@contextmanager
def _patch_multi_strategy_search(pmids: list[str]):
    with ExitStack() as stack:
        yield {
            "queries": stack.enter_context(
                patch("src.core.pipeline.build_multi_strategy_queries", return_value=["trial evidence"])
            ),
            "search": stack.enter_context(
                patch("src.core.pipeline.multi_strategy_esearch", return_value=pmids)
            ),
        }


@contextmanager
def _patch_generation_runtime(*, chat_chain):
    with ExitStack() as stack:
        yield {
            "build_rag_chain": stack.enter_context(
                patch("src.core.pipeline.build_rag_chain", return_value=object())
            ),
            "build_chat_chain": stack.enter_context(
                patch("src.core.pipeline.build_chat_chain", return_value=chat_chain)
            ),
            "log_llm_usage": stack.enter_context(
                patch("src.core.pipeline.log_llm_usage", return_value={})
            ),
            "store_answer_cache": stack.enter_context(
                patch("src.core.pipeline.store_answer_cache")
            ),
            "validation": stack.enter_context(
                patch("src.core.pipeline._run_optional_validation", return_value={})
            ),
            "request_success": stack.enter_context(
                patch("src.core.pipeline._log_request_success")
            ),
            "start_span": stack.enter_context(
                patch(
                    "src.core.pipeline.start_span",
                    side_effect=lambda *args, **kwargs: nullcontext(),
                )
            ),
        }


@skipUnless(PIPELINE_DEPS_AVAILABLE, "langchain_core is not installed")
class PipelineEnhancementTests(TestCase):
    def _invoke_with_patches(self, *, include_paper_links: bool, chat_chain: object | None = None) -> dict:
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
            "context_top_k": 4,
            "retrieval_ms": 12.5,
        }
        chat_chain = chat_chain or _FakeInvokeChain()

        with (
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.8}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline._prepare_chat_context", return_value=context),
            patch("src.core.pipeline.build_rag_chain", return_value=object()),
            patch("src.core.pipeline.build_chat_chain", return_value=chat_chain),
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
            "context_top_k": 4,
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
                return_value=(["12345"], ["12345"], records, docs),
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
        self.assertEqual(result["context_top_k"], 4)

    def test_build_docs_preview_returns_requested_unique_count(self) -> None:
        from src.core.pipeline import _build_docs_preview

        preview = _build_docs_preview(_make_records(10), top_n=10)

        self.assertEqual(len(preview), 10)
        self.assertEqual(len({item["pmid"] for item in preview}), 10)

    def test_prepare_chat_context_invalidates_undersized_cache_hit(self) -> None:
        from src.core.pipeline import _prepare_chat_context
        from src.core.scope import ScopeResult

        config = _config(
            use_reranker=False,
            max_abstracts=8,
            retrieval_candidate_multiplier=1,
        )
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        cached_pmids = [str(10000 + index) for index in range(8)]
        refreshed_records = _make_records(12)
        refreshed_pmids = [record["pmid"] for record in refreshed_records]
        refreshed_docs = _make_docs(12)
        abstract_store = _FakeAbstractStore(refreshed_docs)

        with (
            patch(
                "src.core.pipeline.lookup_query_result_cache",
                return_value={"pubmed_query": "trial evidence", "pmids": cached_pmids},
            ),
            patch("src.core.pipeline.get_query_cache_store", return_value=object()),
            patch("src.core.pipeline.get_abstract_store", return_value=abstract_store),
            patch("src.core.pipeline._prepare_reranker_resources", return_value=None),
            patch("src.core.pipeline.build_multi_strategy_queries", return_value=["trial evidence"]) as mock_queries,
            patch("src.core.pipeline.multi_strategy_esearch", return_value=refreshed_pmids) as mock_esearch,
            patch("src.core.pipeline.pubmed_efetch", return_value=refreshed_records),
            patch("src.core.pipeline.to_documents", return_value=refreshed_docs),
            patch("src.core.pipeline.remember_query_result") as mock_remember,
            patch("src.core.pipeline.upsert_abstracts", return_value=12),
            patch("src.core.pipeline.build_contextual_retrieval_query", return_value="trial evidence"),
            patch("src.core.pipeline._build_retriever", return_value=(object(), False)),
        ):
            result = _prepare_chat_context(
                query="sample question",
                session_id="session-1",
                top_n=10,
                llm=object(),
                scope=scope,
                config=config,
                request_id="req-cache-refresh",
                executor_factory=lambda **_kwargs: _RecordingExecutor([]),
            )

        self.assertEqual(result["cache_status"], "miss")
        self.assertFalse(result["cache_hit"])
        self.assertEqual(len(result["docs_preview"]), 10)
        self.assertEqual(result["abstracts_fetched"], 12)
        self.assertEqual(mock_queries.call_args.args[0], "sample question")
        self.assertGreaterEqual(mock_esearch.call_args.kwargs["retmax_each"], 10)
        mock_remember.assert_called_once()
        self.assertEqual(
            mock_remember.call_args.kwargs["pmids"],
            refreshed_pmids,
        )

    def test_prepare_chat_context_keeps_satisfied_cache_hit_without_refresh(self) -> None:
        from src.core.pipeline import _prepare_chat_context
        from src.core.scope import ScopeResult

        config = _config(
            use_reranker=False,
            max_abstracts=8,
            retrieval_candidate_multiplier=1,
        )
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        cached_records = _make_records(10)
        cached_pmids = [record["pmid"] for record in cached_records]
        cached_docs = _make_docs(10)
        abstract_store = _FakeAbstractStore(cached_docs)

        with (
            patch(
                "src.core.pipeline.lookup_query_result_cache",
                return_value={"pubmed_query": "trial evidence", "pmids": cached_pmids},
            ),
            patch("src.core.pipeline.get_query_cache_store", return_value=object()),
            patch("src.core.pipeline.get_abstract_store", return_value=abstract_store),
            patch("src.core.pipeline._prepare_reranker_resources", return_value=None),
            patch("src.core.pipeline.multi_strategy_esearch") as mock_esearch,
            patch("src.core.pipeline.pubmed_efetch", return_value=cached_records),
            patch("src.core.pipeline.to_documents", return_value=cached_docs),
            patch("src.core.pipeline.remember_query_result") as mock_remember,
            patch("src.core.pipeline.upsert_abstracts", return_value=10),
            patch("src.core.pipeline.build_contextual_retrieval_query", return_value="trial evidence"),
            patch("src.core.pipeline._build_retriever", return_value=(object(), False)),
        ):
            result = _prepare_chat_context(
                query="sample question",
                session_id="session-1",
                top_n=10,
                llm=object(),
                scope=scope,
                config=config,
                request_id="req-cache-hit",
                executor_factory=lambda **_kwargs: _RecordingExecutor([]),
            )

        self.assertEqual(result["cache_status"], "hit")
        self.assertTrue(result["cache_hit"])
        self.assertEqual(len(result["docs_preview"]), 10)
        mock_esearch.assert_not_called()
        mock_remember.assert_not_called()

    def test_invoke_chat_impl_respects_show_papers_toggle(self) -> None:
        with_links = self._invoke_with_patches(include_paper_links=True)
        without_links = self._invoke_with_patches(include_paper_links=False)

        self.assertIn("doi", with_links["sources"][0])
        self.assertIn("fulltext_url", with_links["sources"][0])
        self.assertNotIn("doi", without_links["sources"][0])
        self.assertNotIn("fulltext_url", without_links["sources"][0])

    def test_invoke_chat_impl_adds_source_count_note_when_sources_short(self) -> None:
        payload = self._invoke_with_patches(include_paper_links=True)

        self.assertEqual(
            payload.get("source_count_note"),
            "Only 1 unique papers were available for this query.",
        )

    def test_invoke_chat_impl_cleans_invalid_citations_and_extracts_evidence_quality(self) -> None:
        payload = self._invoke_with_patches(
            include_paper_links=True,
            chat_chain=_StructuredInvokeChain(),
        )

        self.assertEqual(payload.get("evidence_quality"), "Moderate")
        self.assertEqual(payload.get("invalid_citations"), ["99999999"])
        self.assertIn("[PMID: 12345]", payload["answer"])
        self.assertIn("[PMID: UNAVAILABLE]", payload["answer"])

    def test_top_n_10_is_not_capped_by_max_abstracts_in_invoke_path(self) -> None:
        from src.core.pipeline import _invoke_chat_impl
        from src.core.scope import ScopeResult

        docs = _make_docs(12)
        records = _make_records(12)
        pmids = [record["pmid"] for record in records]
        config = _config(use_reranker=False, max_abstracts=8)
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        abstract_store = _FakeAbstractStore(docs)

        with (
            patch("src.core.pipeline._get_llm_safe", return_value=object()),
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.8}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline.get_query_cache_store", return_value=object()),
            patch("src.core.pipeline.get_abstract_store", return_value=abstract_store),
            patch("src.core.pipeline._prepare_reranker_resources", return_value=None),
            patch("src.core.pipeline.lookup_query_result_cache", return_value=None),
            patch("src.core.pipeline.rewrite_to_pubmed_query", return_value="trial evidence"),
            _patch_multi_strategy_search(pmids) as search_patches,
            patch("src.core.pipeline.pubmed_efetch", return_value=records),
            patch("src.core.pipeline.to_documents", return_value=docs),
            patch("src.core.pipeline.remember_query_result"),
            patch("src.core.pipeline.upsert_abstracts", return_value=12),
            _patch_generation_runtime(chat_chain=_FakeInvokeChain()) as runtime_patches,
        ):
            payload = _invoke_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=10,
                config=config,
                request_id="req-top10",
                include_paper_links=True,
                start_time=0.0,
            )

        self.assertEqual(payload["status"], "answered")
        self.assertEqual(len(payload["sources"]), 10)
        self.assertEqual(len(payload["docs_preview"]), 10)
        self.assertEqual(len(payload["retrieved_contexts"]), 10)
        self.assertNotIn("source_count_note", payload)
        self.assertGreaterEqual(search_patches["search"].call_args.kwargs["retmax_each"], 10)
        self.assertEqual(runtime_patches["build_rag_chain"].call_args.kwargs["max_abstracts"], 8)

    def test_top_n_10_is_not_capped_by_max_abstracts_in_stream_path(self) -> None:
        from src.core.pipeline import _stream_chat_impl
        from src.core.scope import ScopeResult

        docs = _make_docs(12)
        records = _make_records(12)
        pmids = [record["pmid"] for record in records]
        config = _config(use_reranker=False, max_abstracts=8)
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        abstract_store = _FakeAbstractStore(docs)

        with (
            patch("src.core.pipeline._get_llm_safe", return_value=object()),
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.8}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline.get_query_cache_store", return_value=object()),
            patch("src.core.pipeline.get_abstract_store", return_value=abstract_store),
            patch("src.core.pipeline._prepare_reranker_resources", return_value=None),
            patch("src.core.pipeline.lookup_query_result_cache", return_value=None),
            patch("src.core.pipeline.rewrite_to_pubmed_query", return_value="trial evidence"),
            _patch_multi_strategy_search(pmids),
            patch("src.core.pipeline.pubmed_efetch", return_value=records),
            patch("src.core.pipeline.to_documents", return_value=docs),
            patch("src.core.pipeline.remember_query_result"),
            patch("src.core.pipeline.upsert_abstracts", return_value=12),
            _patch_generation_runtime(chat_chain=_FakeStreamingChain()) as runtime_patches,
        ):
            stream = _stream_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=10,
                config=config,
                request_id="req-top10-stream",
                include_paper_links=True,
                start_time=0.0,
            )
            chunks, payload = _collect_stream(stream)

        self.assertEqual(chunks, ["First chunk", " second chunk"])
        self.assertEqual(payload["status"], "answered")
        self.assertEqual(len(payload["sources"]), 10)
        self.assertEqual(len(payload["docs_preview"]), 10)
        self.assertEqual(len(payload["retrieved_contexts"]), 10)
        self.assertNotIn("source_count_note", payload)
        self.assertEqual(runtime_patches["build_rag_chain"].call_args.kwargs["max_abstracts"], 8)

    def test_top_n_shortfall_returns_available_unique_papers_and_note(self) -> None:
        from src.core.pipeline import _invoke_chat_impl
        from src.core.scope import ScopeResult

        docs = _make_docs(6)
        records = _make_records(6)
        pmids = [record["pmid"] for record in records]
        config = _config(use_reranker=False, max_abstracts=8)
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        abstract_store = _FakeAbstractStore(docs)

        with (
            patch("src.core.pipeline._get_llm_safe", return_value=object()),
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.8}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline.get_query_cache_store", return_value=object()),
            patch("src.core.pipeline.get_abstract_store", return_value=abstract_store),
            patch("src.core.pipeline._prepare_reranker_resources", return_value=None),
            patch("src.core.pipeline.lookup_query_result_cache", return_value=None),
            patch("src.core.pipeline.rewrite_to_pubmed_query", return_value="trial evidence"),
            _patch_multi_strategy_search(pmids),
            patch("src.core.pipeline.pubmed_efetch", return_value=records),
            patch("src.core.pipeline.to_documents", return_value=docs),
            patch("src.core.pipeline.remember_query_result"),
            patch("src.core.pipeline.upsert_abstracts", return_value=6),
            _patch_generation_runtime(chat_chain=_FakeInvokeChain()),
        ):
            payload = _invoke_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=10,
                config=config,
                request_id="req-top10-shortfall",
                include_paper_links=True,
                start_time=0.0,
            )

        self.assertEqual(payload["status"], "answered")
        self.assertEqual(len(payload["sources"]), 6)
        self.assertEqual(len(payload["docs_preview"]), 6)
        self.assertEqual(
            payload.get("source_count_note"),
            "PubMed returned fewer than 10 records.",
        )

    def test_cached_eight_pmids_do_not_cap_top_n_ten_sources(self) -> None:
        from src.core.pipeline import _invoke_chat_impl
        from src.core.scope import ScopeResult

        docs = _make_docs(10)
        records = _make_records(10)
        cached_pmids = [str(20000 + index) for index in range(8)]
        refreshed_pmids = [record["pmid"] for record in records]
        config = _config(use_reranker=False, max_abstracts=8)
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        abstract_store = _FakeAbstractStore(docs)

        with (
            patch("src.core.pipeline._get_llm_safe", return_value=object()),
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.8}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch(
                "src.core.pipeline.lookup_query_result_cache",
                return_value={"pubmed_query": "trial evidence", "pmids": cached_pmids},
            ),
            patch("src.core.pipeline.get_query_cache_store", return_value=object()),
            patch("src.core.pipeline.get_abstract_store", return_value=abstract_store),
            patch("src.core.pipeline._prepare_reranker_resources", return_value=None),
            _patch_multi_strategy_search(refreshed_pmids) as search_patches,
            patch("src.core.pipeline.pubmed_efetch", return_value=records),
            patch("src.core.pipeline.to_documents", return_value=docs),
            patch("src.core.pipeline.remember_query_result"),
            patch("src.core.pipeline.upsert_abstracts", return_value=10),
            _patch_generation_runtime(chat_chain=_FakeInvokeChain()),
        ):
            payload = _invoke_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=10,
                config=config,
                request_id="req-cache-top10",
                include_paper_links=True,
                start_time=0.0,
            )

        self.assertEqual(payload["status"], "answered")
        self.assertEqual(len(payload["docs_preview"]), 10)
        self.assertEqual(len(payload["sources"]), 10)
        self.assertGreaterEqual(search_patches["search"].call_args.kwargs["retmax_each"], 10)

    def test_sources_are_backfilled_from_docs_preview_when_retriever_returns_fewer_docs(self) -> None:
        from src.core.pipeline import _invoke_chat_impl
        from src.core.scope import ScopeResult

        retrieved_docs = _make_docs(8)
        fetched_records = _make_records(10)
        pmids = [record["pmid"] for record in fetched_records]
        config = _config(use_reranker=False, max_abstracts=8)
        scope = ScopeResult("BIOMEDICAL", True, "ok", None, None)
        abstract_store = _FakeAbstractStore(retrieved_docs)

        with (
            patch("src.core.pipeline._get_llm_safe", return_value=object()),
            patch("src.core.pipeline.classify_intent_details", return_value={"label": "medical", "confidence": 0.8}),
            patch("src.core.pipeline.classify_scope", return_value=scope),
            patch("src.core.pipeline.get_query_cache_store", return_value=object()),
            patch("src.core.pipeline.get_abstract_store", return_value=abstract_store),
            patch("src.core.pipeline._prepare_reranker_resources", return_value=None),
            patch("src.core.pipeline.lookup_query_result_cache", return_value=None),
            patch("src.core.pipeline.rewrite_to_pubmed_query", return_value="trial evidence"),
            _patch_multi_strategy_search(pmids),
            patch("src.core.pipeline.pubmed_efetch", return_value=fetched_records),
            patch("src.core.pipeline.to_documents", return_value=retrieved_docs),
            patch("src.core.pipeline.remember_query_result"),
            patch("src.core.pipeline.upsert_abstracts", return_value=8),
            _patch_generation_runtime(chat_chain=_FakeInvokeChain()),
        ):
            payload = _invoke_chat_impl(
                query="sample question",
                session_id="session-1",
                top_n=10,
                config=config,
                request_id="req-backfill",
                include_paper_links=True,
                start_time=0.0,
            )

        self.assertEqual(payload["status"], "answered")
        self.assertEqual(len(payload["docs_preview"]), 10)
        self.assertEqual(len(payload["sources"]), 10)
        self.assertEqual(payload["sources"][-1]["pmid"], fetched_records[-1]["pmid"])
