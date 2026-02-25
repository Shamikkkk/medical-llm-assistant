from __future__ import annotations

import importlib.util
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch

LANGCHAIN_COMMUNITY_AVAILABLE = importlib.util.find_spec("langchain_community") is not None


@skipUnless(LANGCHAIN_COMMUNITY_AVAILABLE, "langchain_community is not installed")
class QueryCacheInsertTests(TestCase):
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
        self.assertEqual(doc.metadata.get("pmids"), ["123", "456"])
        self.assertEqual(doc.metadata.get("pmids_str"), "123,456")
