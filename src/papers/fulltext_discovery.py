from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from urllib.parse import quote

import requests

from src.papers.doi import build_doi_url, extract_doi, resolve_doi_url

LOGGER = logging.getLogger("papers.discovery")

UNPAYWALL_BASE = "https://api.unpaywall.org/v2/"
UNPAYWALL_TIMEOUT_SECONDS = 12


@dataclass(frozen=True)
class FullTextDiscoveryResult:
    doi: str
    pmcid: str
    fulltext_url: str
    source: str
    is_open_access: bool
    note: str


def discover_fulltext(
    *,
    doi: str,
    pmcid: str,
    current_url: str = "",
    unpaywall_email: str | None = None,
) -> FullTextDiscoveryResult:
    normalized_doi = extract_doi(doi)
    normalized_pmcid = _normalize_pmcid(pmcid)
    if normalized_pmcid:
        return FullTextDiscoveryResult(
            doi=normalized_doi,
            pmcid=normalized_pmcid,
            fulltext_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{normalized_pmcid}/",
            source="PMC",
            is_open_access=True,
            note="PMCID available; PMC full text path preferred.",
        )

    resolved = ""
    if normalized_doi:
        resolved = resolve_doi_url(normalized_doi) or build_doi_url(normalized_doi)
    elif current_url:
        resolved = str(current_url or "").strip()

    email = str(unpaywall_email or os.getenv("UNPAYWALL_EMAIL", "")).strip()
    if normalized_doi and email:
        oa_url = _lookup_unpaywall_oa_url(normalized_doi, email)
        if oa_url:
            return FullTextDiscoveryResult(
                doi=normalized_doi,
                pmcid=normalized_pmcid,
                fulltext_url=oa_url,
                source="UNPAYWALL_OA",
                is_open_access=True,
                note="Open-access location discovered via Unpaywall.",
            )

    if resolved:
        return FullTextDiscoveryResult(
            doi=normalized_doi,
            pmcid=normalized_pmcid,
            fulltext_url=resolved,
            source="DOI",
            is_open_access=False,
            note=(
                "Publisher landing page resolved from DOI; open-access full text not "
                "confirmed automatically."
            ),
        )

    return FullTextDiscoveryResult(
        doi=normalized_doi,
        pmcid=normalized_pmcid,
        fulltext_url="",
        source="NONE",
        is_open_access=False,
        note="No PMCID or resolvable DOI full-text URL found.",
    )


def _lookup_unpaywall_oa_url(doi: str, email: str) -> str:
    url = f"{UNPAYWALL_BASE}{quote(doi)}"
    try:
        response = requests.get(
            url,
            params={"email": email},
            timeout=UNPAYWALL_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        LOGGER.debug("Unpaywall lookup failed for %s: %s", doi, exc)
        return ""

    if not isinstance(payload, dict):
        return ""
    best = payload.get("best_oa_location") or {}
    if isinstance(best, dict):
        candidate = str(best.get("url_for_pdf") or best.get("url") or "").strip()
        if candidate:
            return candidate
    locations = payload.get("oa_locations") or []
    for item in locations:
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("url_for_pdf") or item.get("url") or "").strip()
        if candidate:
            return candidate
    return ""


def _normalize_pmcid(pmcid: str | None) -> str:
    value = str(pmcid or "").strip()
    if not value:
        return ""
    if value.upper().startswith("PMC"):
        return value.upper()
    digits = "".join(ch for ch in value if ch.isdigit())
    return f"PMC{digits}" if digits else ""
