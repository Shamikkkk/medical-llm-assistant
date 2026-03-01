from __future__ import annotations

import importlib.util
from unittest import TestCase, skipUnless
from unittest.mock import patch

PUBMED_DEPS_AVAILABLE = (
    importlib.util.find_spec("requests") is not None
    and importlib.util.find_spec("langchain_core") is not None
)


PUBMED_XML = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Trial evidence title</ArticleTitle>
        <Abstract>
          <AbstractText Label="Background">Background text.</AbstractText>
          <AbstractText Label="Results">Results text.</AbstractText>
        </Abstract>
        <Journal>
          <JournalIssue>
            <PubDate>
              <MedlineDate>2024 Jan-Feb</MedlineDate>
            </PubDate>
          </JournalIssue>
          <Title>Clinical Journal</Title>
        </Journal>
        <AuthorList>
          <Author>
            <ForeName>Jane</ForeName>
            <LastName>Doe</LastName>
          </Author>
          <Author>
            <CollectiveName>Investigators Group</CollectiveName>
          </Author>
        </AuthorList>
        <ELocationID EIdType="doi">10.1000/example-doi</ELocationID>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345</ArticleId>
        <ArticleId IdType="pmc">1234567</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
""".strip()


@skipUnless(PUBMED_DEPS_AVAILABLE, "PubMed parsing dependencies are not installed")
class PubMedParsingTests(TestCase):
    @patch("src.integrations.pubmed._get_text", return_value=PUBMED_XML)
    def test_pubmed_efetch_parses_missing_year_doi_and_pmcid_edges(self, _mock_get_text) -> None:
        from src.integrations.pubmed import pubmed_efetch

        records = pubmed_efetch(["12345"])

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["pmid"], "12345")
        self.assertEqual(record["title"], "Trial evidence title")
        self.assertEqual(record["year"], "2024")
        self.assertEqual(record["journal"], "Clinical Journal")
        self.assertEqual(record["authors"], ["Jane Doe", "Investigators Group"])
        self.assertEqual(record["doi"], "10.1000/example-doi")
        self.assertEqual(record["pmcid"], "PMC1234567")
        self.assertEqual(
            record["fulltext_url"],
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        )
        self.assertIn("Background text.", record["abstract"])
        self.assertIn("Results text.", record["abstract"])

    def test_to_documents_preserves_pubmed_metadata(self) -> None:
        from src.integrations.pubmed import to_documents

        docs = to_documents(
            [
                {
                    "pmid": "12345",
                    "title": "Trial evidence title",
                    "abstract": "Background text.\nResults text.",
                    "journal": "Clinical Journal",
                    "year": "2024",
                    "authors": ["Jane Doe"],
                    "doi": "10.1000/example-doi",
                    "pmcid": "PMC1234567",
                    "fulltext_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
                }
            ]
        )

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.metadata["pmid"], "12345")
        self.assertEqual(doc.metadata["doi"], "10.1000/example-doi")
        self.assertIn("Results text.", doc.page_content)

    @patch("src.integrations.pubmed.log_llm_usage", return_value={})
    def test_build_multi_strategy_queries_returns_complementary_queries(self, _mock_log_usage) -> None:
        from src.integrations.pubmed import build_multi_strategy_queries

        class _FakeMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        class _FakeLLM:
            def invoke(self, prompt: str) -> _FakeMessage:
                if "MeSH terms only" in prompt:
                    return _FakeMessage("Atrial Fibrillation[MeSH] AND Warfarin")
                if "capture related concepts" in prompt:
                    return _FakeMessage("warfarin OR anticoagulants AND atrial fibrillation")
                return _FakeMessage("atrial fibrillation AND warfarin stroke")

        queries = build_multi_strategy_queries(
            "warfarin for stroke prevention in atrial fibrillation",
            _FakeLLM(),
        )

        self.assertEqual(
            queries[:3],
            [
                "atrial fibrillation AND warfarin stroke",
                "Atrial Fibrillation[MeSH] AND Warfarin",
                "warfarin OR anticoagulants AND atrial fibrillation",
            ],
        )
        self.assertIn(
            "warfarin for stroke prevention in atrial fibrillation",
            queries,
        )

    @patch("src.integrations.pubmed.pubmed_esearch")
    def test_multi_strategy_esearch_merges_unique_pmids_in_query_order(self, mock_esearch) -> None:
        from src.integrations.pubmed import multi_strategy_esearch

        mock_esearch.side_effect = [
            ["1", "2", "3"],
            ["3", "4"],
            ["2", "5"],
        ]

        pmids = multi_strategy_esearch(["q1", "q2", "q3"], retmax_each=3)

        self.assertEqual(pmids, ["1", "2", "3", "4", "5"])
