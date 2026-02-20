from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import urlencode
import time
import xml.etree.ElementTree as ET

import requests
from langchain_core.documents import Document

from src.logging_utils import log_llm_usage
from src.papers.doi import build_doi_url, extract_doi

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_DELAY_SECONDS = 0.34


def build_esearch_url(term: str, retmax: int = 10) -> str:
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": retmax,
        "term": term,
    }
    return f"{EUTILS_BASE}/esearch.fcgi?{urlencode(params)}"


def pubmed_esearch(term: str, retmax: int = 10) -> List[str]:
    url = build_esearch_url(term, retmax=retmax)
    data = _get_json(url)
    if not data:
        return []
    id_list = data.get("esearchresult", {}).get("idlist", [])
    return [str(pmid) for pmid in id_list]


def pubmed_efetch(pmids: List[str]) -> List[Dict[str, Any]]:
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
    }
    url = f"{EUTILS_BASE}/efetch.fcgi?{urlencode(params)}"
    xml_text = _get_text(url)
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    records: List[Dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _get_text_from(article.find(".//PMID")) or ""
        title = _get_text_from(article.find(".//ArticleTitle")) or ""

        abstract_parts = []
        for abstract_node in article.findall(".//Abstract/AbstractText"):
            text = "".join(abstract_node.itertext()).strip()
            if text:
                abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        journal = (
            _get_text_from(article.find(".//Journal/Title"))
            or _get_text_from(article.find(".//Journal/ISOAbbreviation"))
            or ""
        )
        year = _extract_year(article)
        authors = _extract_authors(article)
        doi, pmcid, fulltext_url = _extract_identifier_info(article)

        records.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "authors": authors,
                "doi": doi,
                "pmcid": pmcid,
                "fulltext_url": fulltext_url,
            }
        )
    return records


def to_documents(records: List[Dict[str, Any]]) -> List[Document]:
    documents: List[Document] = []
    for record in records:
        title = record.get("title", "")
        abstract = record.get("abstract", "")
        page_content = title.strip()
        if abstract:
            page_content = f"{page_content}\n\n{abstract}".strip()
        metadata = {
            "pmid": record.get("pmid", ""),
            "title": title,
            "journal": record.get("journal", ""),
            "year": record.get("year", ""),
            "authors": record.get("authors", []),
            "doi": record.get("doi", ""),
            "pmcid": record.get("pmcid", ""),
            "fulltext_url": record.get("fulltext_url", ""),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


def rewrite_to_pubmed_query(user_query: str, llm: Any | None) -> str:
    """Rewrite a user query into a concise PubMed-ready boolean query."""
    if not llm:
        return user_query.strip()

    prompt = (
        "You are a medical search assistant. Rewrite the user's question into a "
        "PubMed-ready query.\n"
        "Rules:\n"
        "- Include key concepts using boolean operators (AND/OR).\n"
        "- Enforce a cardiovascular focus (heart/cardiac/cardiovascular terms).\n"
        "- Keep it short (<= 200 characters if possible).\n"
        "- Output only the final query string, no quotes or extra text.\n"
        f"User query: {user_query}\n"
    )

    rewritten = _invoke_llm(llm, prompt, usage_tag="rewrite")
    if not rewritten:
        return user_query.strip()

    cleaned = rewritten.strip().strip('"').strip("'")
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    if not cleaned:
        return user_query.strip()

    if not _contains_cardiovascular_term(cleaned):
        cleaned = f"({cleaned}) AND (cardiovascular OR heart OR cardiac)"
    if len(cleaned) > 200:
        cleaned = cleaned[:200].rstrip()
    return cleaned


def _get_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def _get_text(url: str, params: Dict[str, Any] | None = None) -> str | None:
    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None


def _invoke_llm(llm: Any, prompt: str, usage_tag: str | None = None) -> str | None:
    try:
        if hasattr(llm, "invoke"):
            result = llm.invoke(prompt)
            if usage_tag:
                log_llm_usage(usage_tag, result)
            return _coerce_to_text(result)
        if hasattr(llm, "predict"):
            result = llm.predict(prompt)
            return _coerce_to_text(result)
    except Exception:
        return None
    return None


def _coerce_to_text(result: Any) -> str:
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def _get_text_from(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = "".join(node.itertext()).strip()
    return text or None


def _extract_year(article: ET.Element) -> str:
    year = _get_text_from(article.find(".//JournalIssue/PubDate/Year"))
    if year:
        return year
    medline_date = _get_text_from(article.find(".//JournalIssue/PubDate/MedlineDate"))
    if medline_date:
        digits = "".join(ch for ch in medline_date if ch.isdigit())
        if len(digits) >= 4:
            return digits[:4]
    return ""


def _extract_authors(article: ET.Element) -> List[str]:
    authors: List[str] = []
    for author in article.findall(".//AuthorList/Author"):
        collective = _get_text_from(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        last_name = _get_text_from(author.find("LastName")) or ""
        fore_name = _get_text_from(author.find("ForeName")) or ""
        name = " ".join(part for part in [fore_name, last_name] if part).strip()
        if name:
            authors.append(name)
    return authors


def _extract_identifier_info(article: ET.Element) -> tuple[str, str, str]:
    doi = ""
    pmcid = ""
    for node in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = str(node.attrib.get("IdType", "")).strip().lower()
        value = _get_text_from(node) or ""
        if id_type == "doi" and not doi:
            doi = extract_doi(value)
        elif id_type == "pmc" and not pmcid:
            pmcid = _normalize_pmcid(value)

    if not doi:
        for node in article.findall(".//Article/ELocationID"):
            id_type = str(node.attrib.get("EIdType", "")).strip().lower()
            if id_type != "doi":
                continue
            doi = extract_doi(_get_text_from(node) or "")
            if doi:
                break

    fulltext_url = ""
    if pmcid:
        fulltext_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    elif doi:
        fulltext_url = build_doi_url(doi)
    return doi, pmcid, fulltext_url


def _normalize_pmcid(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.upper().startswith("PMC"):
        return text.upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    return f"PMC{digits}" if digits else ""


def _contains_cardiovascular_term(text: str) -> bool:
    lowered = text.lower()
    for term in ("cardiovascular", "cardiac", "heart", "coronary"):
        if term in lowered:
            return True
    return False
