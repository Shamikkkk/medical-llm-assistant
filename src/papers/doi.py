from __future__ import annotations

import re

DOI_RESOLVER_BASE = "https://doi.org/"

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
