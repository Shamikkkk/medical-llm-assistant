from __future__ import annotations

from functools import lru_cache
import logging
import re

import requests

LOGGER = logging.getLogger("papers.doi")

DOI_RESOLVER_BASE = "https://doi.org/"
DOI_TIMEOUT_SECONDS = 12
DOI_HEADERS = {
    "User-Agent": "medical-llm-assistant/1.0 (+https://pubmed.ncbi.nlm.nih.gov)",
    "Accept": "text/html,application/xhtml+xml",
}

DOI_REGEX = re.compile(
    r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    flags=re.IGNORECASE,
)


def extract_doi(raw_value: str | None) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    text = text.replace("doi:", "").replace("DOI:", "").strip()
    text = text.strip("<> ")
    match = DOI_REGEX.search(text)
    if not match:
        return ""
    doi = match.group(1).strip().rstrip(".;,)")
    return doi


def build_doi_url(doi: str) -> str:
    normalized = extract_doi(doi)
    if not normalized:
        return ""
    return f"{DOI_RESOLVER_BASE}{normalized}"


@lru_cache(maxsize=2048)
def resolve_doi_url(doi: str) -> str:
    """Resolve DOI redirect target (publisher landing URL)."""
    doi_url = build_doi_url(doi)
    if not doi_url:
        return ""

    try:
        head = requests.head(
            doi_url,
            timeout=DOI_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers=DOI_HEADERS,
        )
        if head.ok and str(head.url or "").strip():
            return str(head.url).strip()
    except requests.RequestException:
        pass

    try:
        response = requests.get(
            doi_url,
            timeout=DOI_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers=DOI_HEADERS,
        )
        if response.ok and str(response.url or "").strip():
            return str(response.url).strip()
    except requests.RequestException as exc:
        LOGGER.debug("DOI resolve failed for %s: %s", doi, exc)
        return ""

    return ""
