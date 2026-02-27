from __future__ import annotations

import importlib.util
import json
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch

LANGCHAIN_COMMUNITY_AVAILABLE = importlib.util.find_spec("langchain_community") is not None


@skipUnless(LANGCHAIN_COMMUNITY_AVAILABLE, "langchain_community is not installed")
class QueryCacheInsertTests(TestCase):
    def test_build_query_cache_document_omits_empty_pmids_metadata(self) -> None:
        from src.integrations.storage import build_query_cache_document

        doc = build_query_cache_document(
            "empty result query",
            "empty result query",
            [],
        )

        self.assertNotIn("pmids", doc.metadata)
        self.assertNotIn("pmids_json", doc.metadata)
        self.assertNotIn("pmids_csv", doc.metadata)

    def test_add_query_cache_entry_skips_empty_pmids(self) -> None:
        from src.integrations.storage import add_query_cache_entry

        store = Mock()

        with patch("src.integrations.storage._persist_if_supported") as mock_persist:
            inserted = add_query_cache_entry(
                store,
                query="glioblastoma therapy evidence",
                pubmed_query="glioblastoma AND temozolomide",
                pmids=[],
            )

        self.assertFalse(inserted)
        store.add_documents.assert_not_called()
        mock_persist.assert_not_called()

    def test_add_query_cache_entry_writes_non_empty_pmids(self) -> None:
        from src.integrations.storage import add_query_cache_entry

        store = Mock()

        with patch("src.integrations.storage._persist_if_supported") as mock_persist:
            inserted = add_query_cache_entry(
                store,
                query="ibs low fodmap evidence",
                pubmed_query="IBS AND low FODMAP",
                pmids=["123", " ", "456"],
            )

        self.assertTrue(inserted)
        store.add_documents.assert_called_once()
        mock_persist.assert_called_once_with(store)

        docs = store.add_documents.call_args.args[0]
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertNotIn("pmids", doc.metadata)
        self.assertEqual(json.loads(doc.metadata.get("pmids_json", "")), ["123", "456"])
        self.assertEqual(doc.metadata.get("pmids_csv"), "123,456")

    def test_sanitize_metadata_serializes_nested_values(self) -> None:
        from src.integrations.storage import sanitize_metadata

        sanitized = sanitize_metadata(
            {
                "pmid": "123",
                "authors": ["Jane Doe", "John Smith"],
                "details": {"phase": 3},
                "rank": 1,
                "optional": None,
            }
        )

        self.assertEqual(sanitized["pmid"], "123")
        self.assertEqual(json.loads(sanitized["authors"]), ["Jane Doe", "John Smith"])
        self.assertEqual(json.loads(sanitized["details"]), {"phase": 3})
        self.assertEqual(sanitized["rank"], 1)
        self.assertNotIn("optional", sanitized)

    def test_query_result_cache_ttl_expiry_returns_none(self) -> None:
        from src.integrations.storage import QueryResultCache

        current_time = [1000.0]
        cache = QueryResultCache(time_func=lambda: current_time[0])
        cache.set(
            "heart failure evidence",
            pubmed_query="heart failure evidence",
            pmids=["12345"],
        )

        self.assertIsNotNone(
            cache.get(
                "heart failure evidence",
                ttl_seconds=30,
                negative_ttl_seconds=5,
            )
        )
        current_time[0] = 1031.0
        self.assertIsNone(
            cache.get(
                "heart failure evidence",
                ttl_seconds=30,
                negative_ttl_seconds=5,
            )
        )

    def test_negative_cache_uses_negative_ttl(self) -> None:
        from src.integrations.storage import QueryResultCache

        current_time = [2000.0]
        cache = QueryResultCache(time_func=lambda: current_time[0])
        cache.set(
            "rare disease no hits",
            pubmed_query="rare disease no hits",
            pmids=[],
        )

        self.assertIsNotNone(
            cache.get(
                "rare disease no hits",
                ttl_seconds=60,
                negative_ttl_seconds=10,
            )
        )
        current_time[0] = 2011.0
        self.assertIsNone(
            cache.get(
                "rare disease no hits",
                ttl_seconds=60,
                negative_ttl_seconds=10,
            )
        )

    def test_extract_pmids_supports_serialized_metadata(self) -> None:
        from src.integrations.storage import _extract_pmids_from_payload

        pmids = _extract_pmids_from_payload(
            {
                "pmids_json": "[\"41758246\", \"12345678\"]",
                "pmids_csv": "41758246,12345678",
            }
        )

        self.assertEqual(pmids, ["41758246", "12345678"])
