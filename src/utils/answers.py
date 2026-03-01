# Created by Codex - Section 4
from __future__ import annotations

import logging
import re

LOGGER = logging.getLogger("answer.postprocess")

_PMID_CITATION_RE = re.compile(r"\[PMID:\s*([^\]\s]+)\s*\]", flags=re.IGNORECASE)
_EVIDENCE_QUALITY_SECTION_RE = re.compile(
    r"(?ims)^##\s*Evidence Quality\s*(?P<body>.*?)(?=^##\s|\Z)"
)
_EVIDENCE_QUALITY_LABELS = ("Strong", "Moderate", "Preliminary", "Insufficient")
_INSUFFICIENT_EVIDENCE_ANSWER = (
    "The provided abstracts do not contain sufficient evidence to answer this question."
)


def validate_citations_in_answer(answer: str, source_pmids: list[str]) -> tuple[str, list[str]]:
    valid_pmids = {
        str(pmid or "").strip()
        for pmid in source_pmids
        if str(pmid or "").strip()
    }
    invalid_pmids: list[str] = []

    def replace(match: re.Match[str]) -> str:
        cited_pmid = str(match.group(1) or "").strip()
        if not cited_pmid or cited_pmid.upper() == "UNAVAILABLE":
            return "[PMID: UNAVAILABLE]"
        if cited_pmid in valid_pmids:
            return f"[PMID: {cited_pmid}]"
        if cited_pmid not in invalid_pmids:
            invalid_pmids.append(cited_pmid)
            LOGGER.warning(
                "Replacing invalid PMID citation. cited_pmid=%s valid_source_count=%s",
                cited_pmid,
                len(valid_pmids),
            )
        return "[PMID: UNAVAILABLE]"

    cleaned_answer = _PMID_CITATION_RE.sub(replace, str(answer or ""))
    return cleaned_answer, invalid_pmids


def extract_evidence_quality(answer: str) -> str:
    normalized_answer = str(answer or "").strip()
    if not normalized_answer:
        return ""
    if normalized_answer == _INSUFFICIENT_EVIDENCE_ANSWER:
        return "Insufficient"

    match = _EVIDENCE_QUALITY_SECTION_RE.search(normalized_answer)
    if not match:
        return ""

    section_body = " ".join(str(match.group("body") or "").split())
    for label in _EVIDENCE_QUALITY_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", section_body, flags=re.IGNORECASE):
            return label
    return ""


def annotate_answer_metadata(answer: str, source_pmids: list[str]) -> tuple[str, list[str], str]:
    cleaned_answer, invalid_citations = validate_citations_in_answer(answer, source_pmids)
    evidence_quality = extract_evidence_quality(cleaned_answer)
    return cleaned_answer, invalid_citations, evidence_quality
