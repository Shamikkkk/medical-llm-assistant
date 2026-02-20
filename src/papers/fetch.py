from __future__ import annotations

from io import BytesIO
import logging
import os
import xml.etree.ElementTree as ET

import requests

from src.integrations.pubmed import pubmed_efetch
from src.papers.doi import extract_doi
from src.papers.fetch_fulltext import (
    STATUS_OK_OA_HTML,
    STATUS_PAYWALLED_OR_BLOCKED,
    fetch_readable_text_from_url,
)
from src.papers.fulltext_discovery import discover_fulltext
from src.papers.store import (
    PAPER_TIER_ABSTRACT,
    PAPER_TIER_FULL_TEXT,
    PaperContent,
    build_pubmed_url,
)

LOGGER = logging.getLogger("papers.fetch")

ID_CONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
REQUEST_TIMEOUT_SECONDS = 25


def fetch_paper_content(pmid: str) -> PaperContent | None:
    normalized_pmid = "".join(ch for ch in str(pmid) if ch.isdigit())
    if not normalized_pmid:
        return None

    records = pubmed_efetch([normalized_pmid])
    if not records:
        return None
    record = records[0]
    doi = extract_doi(str(record.get("doi", "") or ""))
    pmcid = str(record.get("pmcid", "") or "").strip() or get_pmcid_from_pmid(normalized_pmid) or ""
    discovered = discover_fulltext(
        doi=doi,
        pmcid=pmcid,
        current_url=str(record.get("fulltext_url", "") or ""),
        unpaywall_email=os.getenv("UNPAYWALL_EMAIL"),
    )
    fulltext_url = discovered.fulltext_url
    full_text = ""
    source_label = "PUBMED_ABSTRACT"
    notes = ""

    if pmcid:
        full_text = fetch_pmc_full_text(pmcid)
        if full_text:
            source_label = "PMC"
            fulltext_url = fulltext_url or f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
        else:
            notes = "PMCID exists but PMC full text extraction failed; using abstract metadata only."

    if not full_text and fulltext_url and discovered.is_open_access:
        text, meta, status = fetch_readable_text_from_url(fulltext_url)
        if status == STATUS_OK_OA_HTML and text:
            full_text = text
            source_label = "OA_HTML"
            fulltext_url = str(meta.get("final_url") or fulltext_url)
            notes = "Full text extracted from a publicly accessible open-access page."
        elif status == STATUS_PAYWALLED_OR_BLOCKED:
            notes = (
                "Full text appears paywalled; I can't fetch it automatically. "
                "You can (a) paste sections, (b) upload the PDF, or (c) provide an "
                "accessible link that allows text extraction."
            )

    if not full_text and not notes:
        notes = (
            "Full text appears paywalled; I can't fetch it automatically. "
            "You can (a) paste sections, (b) upload the PDF, or (c) provide an "
            "accessible link that allows text extraction."
        )
        if discovered.note:
            notes = f"{notes} ({discovered.note})"

    content_tier = PAPER_TIER_FULL_TEXT if full_text else PAPER_TIER_ABSTRACT
    if not full_text:
        LOGGER.info("[PAPER] PMID=%s abstract-only mode (full text unavailable)", normalized_pmid)
    return PaperContent(
        pmid=normalized_pmid,
        doi=doi,
        fulltext_url=fulltext_url,
        title=str(record.get("title", "") or ""),
        authors=[str(item) for item in (record.get("authors") or [])],
        year=str(record.get("year", "") or ""),
        journal=str(record.get("journal", "") or ""),
        pubmed_url=build_pubmed_url(normalized_pmid),
        abstract=str(record.get("abstract", "") or ""),
        full_text=full_text,
        content_tier=content_tier,
        source_label=source_label,
        fetched_at=_utc_now_iso(),
        pmcid=pmcid,
        notes=notes,
    )


def get_pmcid_from_pmid(pmid: str) -> str | None:
    params = {"ids": pmid, "format": "json"}
    try:
        response = requests.get(ID_CONV_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None
    records = payload.get("records", []) if isinstance(payload, dict) else []
    if not records:
        return None
    pmcid = str(records[0].get("pmcid", "") or "").strip()
    return pmcid or None


def fetch_pmc_full_text(pmcid: str) -> str:
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
    try:
        response = requests.get(EUTILS_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        xml_text = response.text
    except requests.RequestException:
        return ""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""

    paragraphs: list[str] = []
    for node in root.findall(".//body//p"):
        text = " ".join("".join(node.itertext()).split())
        if text:
            paragraphs.append(text)
    if not paragraphs:
        return ""
    return "\n\n".join(paragraphs)


def extract_text_from_uploaded_pdf(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        LOGGER.warning("pypdf not installed; PDF upload parsing is unavailable.")
        return ""

    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception:
        return ""

    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        cleaned = " ".join(text.split())
        if cleaned:
            pages.append(cleaned)
    return "\n\n".join(pages)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
