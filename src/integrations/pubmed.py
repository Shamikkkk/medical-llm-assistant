from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from urllib.parse import urlencode
import time
import xml.etree.ElementTree as ET

import requests
from langchain_core.documents import Document

from src.intent import normalize_user_query
from src.logging_utils import log_llm_usage
from src.papers.doi import build_doi_url, extract_doi

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_DELAY_SECONDS = 0.34

PUBMED_REWRITE_PROMPT = """Convert this medical question into a concise PubMed search query.
Use MeSH terms where applicable. Use AND to combine concepts. Do not use quotes. Maximum 12 words.
Question: {query}
PubMed query:"""

_MESH_QUERY_PROMPT = "Rewrite this as a PubMed query using MeSH terms only: {query}"
_BROAD_QUERY_PROMPT = "Write a broader PubMed query using OR to capture related concepts: {query}"


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


def build_multi_strategy_queries(user_query: str, llm: Any | None) -> list[str]:
    normalized_query = normalize_user_query(user_query).strip() or str(user_query or "").strip()
    primary_query = rewrite_to_pubmed_query(normalized_query, llm)
    if not llm:
        return _unique_queries([primary_query or normalized_query])

    mesh_query = _clean_query(
        _invoke_llm(llm, _MESH_QUERY_PROMPT.format(query=normalized_query), usage_tag="rewrite.mesh")
    )
    broad_query = _clean_query(
        _invoke_llm(llm, _BROAD_QUERY_PROMPT.format(query=normalized_query), usage_tag="rewrite.broad")
    )
    return _unique_queries([primary_query, mesh_query, broad_query, normalized_query])


def multi_strategy_esearch(queries: list[str], retmax_each: int = 15) -> list[str]:
    normalized_queries = _unique_queries(queries)
    if not normalized_queries:
        return []

    effective_retmax = max(1, int(retmax_each))
    results_by_index: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(normalized_queries))) as executor:
        future_to_index = {
            executor.submit(pubmed_esearch, query, effective_retmax): index
            for index, query in enumerate(normalized_queries)
        }
        for future, index in future_to_index.items():
            try:
                results_by_index[index] = [str(pmid) for pmid in future.result() if str(pmid).strip()]
            except Exception:
                results_by_index[index] = []

    merged: list[str] = []
    seen: set[str] = set()
    cap = effective_retmax * 2
    for index in range(len(normalized_queries)):
        for pmid in results_by_index.get(index, []):
            if pmid in seen:
                continue
            seen.add(pmid)
            merged.append(pmid)
            if len(merged) >= cap:
                return merged
    return merged


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
    normalized_query = normalize_user_query(user_query).strip() or str(user_query or "").strip()
    if not llm:
        return normalized_query

    prompt = PUBMED_REWRITE_PROMPT.format(query=normalized_query)
    rewritten = _invoke_llm(llm, prompt, usage_tag="rewrite")
    cleaned = _clean_query(rewritten)
    return cleaned or normalized_query


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


def _clean_query(value: str | None) -> str:
    cleaned = str(value or "").strip().strip('"').strip("'")
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    if len(cleaned) > 200:
        cleaned = cleaned[:200].rstrip()
    return cleaned


def _unique_queries(queries: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = _clean_query(query)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return normalized


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
