from __future__ import annotations

import importlib.util
from unittest import TestCase, skipUnless

LANGCHAIN_CORE_AVAILABLE = importlib.util.find_spec("langchain_core") is not None


def _doc(
    pmid: str,
    title: str,
    abstract: str,
    *,
    distance: float | None = None,
):
    from langchain_core.documents import Document

    metadata = {
        "pmid": pmid,
        "title": title,
        "journal": "Test Journal",
        "year": "2024",
        "doi": f"10.1000/{pmid}",
        "fulltext_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }
    if distance is not None:
        metadata["_distance"] = distance

    return Document(
        page_content=f"{title}\n\n{abstract}",
        metadata=metadata,
    )


@skipUnless(LANGCHAIN_CORE_AVAILABLE, "langchain_core is not installed")
class RetrievalEnhancementTests(TestCase):
    def test_context_budget_limits_abstract_count_and_tokens(self) -> None:
        from src.core.retrieval import build_context_text, count_tokens

        docs = [
            _doc(
                str(index),
                f"Study {index}",
                " ".join(["evidence"] * 400),
            )
            for index in range(1, 6)
        ]

        context_text = build_context_text(
            docs,
            max_abstracts=2,
            max_context_tokens=120,
            trim_strategy="truncate",
        )

        self.assertLessEqual(context_text.count("PMID:"), 2)
        self.assertLessEqual(count_tokens(context_text), 120)

    def test_hybrid_rerank_can_change_order_vs_semantic_rank(self) -> None:
        from src.core.retrieval import hybrid_rerank_documents

        docs = [
            _doc(
                "1",
                "General cardiovascular outcomes",
                "broad cardiovascular outcomes and registry follow-up",
                distance=0.05,
            ),
            _doc(
                "2",
                "Warfarin stroke prevention",
                "warfarin stroke prevention atrial fibrillation exact keyword match",
                distance=0.80,
            ),
        ]

        semantic_only = hybrid_rerank_documents(
            "warfarin stroke prevention",
            docs,
            alpha=1.0,
            limit=2,
        )
        hybrid = hybrid_rerank_documents(
            "warfarin stroke prevention",
            docs,
            alpha=0.0,
            limit=2,
        )

        self.assertEqual(semantic_only[0].metadata["pmid"], "1")
        self.assertEqual(hybrid[0].metadata["pmid"], "2")

    def test_semantic_deduplication_removes_near_duplicate_abstracts(self) -> None:
        from src.core.retrieval import deduplicate_by_semantic_similarity

        docs = [
            _doc(
                "1",
                "Warfarin outcomes trial",
                "warfarin reduced stroke risk in atrial fibrillation patients with consistent follow up",
            ),
            _doc(
                "2",
                "Warfarin outcomes trial duplicate",
                "warfarin reduced stroke risk in atrial fibrillation patients with consistent follow up",
            ),
            _doc(
                "3",
                "Bleeding outcomes study",
                "major bleeding outcomes were mixed across cohorts and dose strategies",
            ),
        ]

        deduplicated = deduplicate_by_semantic_similarity(docs, threshold=0.9)

        self.assertEqual([doc.metadata["pmid"] for doc in deduplicated], ["1", "3"])

    def test_citation_alignment_disclaims_unsupported_claims(self) -> None:
        from src.core.retrieval import align_answer_citations

        answer = "Warfarin reduced stroke risk in AF. Aspirin cured advanced heart failure."
        contexts = [
            {
                "pmid": "12345",
                "context": "Warfarin reduced stroke risk in atrial fibrillation cohorts.",
            }
        ]

        aligned, issues = align_answer_citations(answer, contexts=contexts, mode="disclaim")

        self.assertIn("Warfarin reduced stroke risk in AF.", aligned)
        self.assertIn("[No supporting PMID found in retrieved abstracts]", aligned)
        self.assertEqual(issues, ["Aspirin cured advanced heart failure."])

    def test_citation_alignment_can_remove_unsupported_claims(self) -> None:
        from src.core.retrieval import align_answer_citations

        answer = "Warfarin reduced stroke risk in AF. Aspirin cured advanced heart failure."
        contexts = [
            {
                "pmid": "12345",
                "context": "Warfarin reduced stroke risk in atrial fibrillation cohorts.",
            }
        ]

        aligned, issues = align_answer_citations(answer, contexts=contexts, mode="remove")

        self.assertIn("Warfarin reduced stroke risk in AF.", aligned)
        self.assertNotIn("Aspirin cured advanced heart failure.", aligned)
        self.assertEqual(issues, ["Aspirin cured advanced heart failure."])
