from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


PAPER_TIER_FULL_TEXT = "full_text"
PAPER_TIER_ABSTRACT = "abstract_only"
PAPER_TIER_UPLOADED_PDF = "uploaded_pdf"


@dataclass(frozen=True)
class PaperContent:
    pmid: str
    doi: str
    fulltext_url: str
    title: str
    authors: list[str]
    year: str
    journal: str
    pubmed_url: str
    abstract: str
    full_text: str
    content_tier: str
    source_label: str
    fetched_at: str
    pmcid: str = ""
    notes: str = ""

    @property
    def primary_text(self) -> str:
        if self.full_text.strip():
            return self.full_text
        return self.abstract


class PaperStore:
    """Disk-backed paper cache for abstract/full-text/PDF content tiers."""

    def __init__(self, root_dir: str | Path = "./data/papers") -> None:
        self.root = Path(root_dir).expanduser().absolute()
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, pmid: str) -> PaperContent | None:
        path = self._paper_path(pmid)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return _paper_content_from_dict(payload)

    def save(self, paper: PaperContent) -> None:
        path = self._paper_path(paper.pmid)
        payload = asdict(paper)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_with_pdf(
        self,
        *,
        pmid: str,
        pdf_text: str,
        file_name: str,
    ) -> PaperContent | None:
        current = self.load(pmid)
        if current is None:
            return None
        merged = PaperContent(
            pmid=current.pmid,
            doi=current.doi,
            fulltext_url=current.fulltext_url,
            title=current.title,
            authors=current.authors,
            year=current.year,
            journal=current.journal,
            pubmed_url=current.pubmed_url,
            abstract=current.abstract,
            full_text=str(pdf_text or ""),
            content_tier=PAPER_TIER_UPLOADED_PDF,
            source_label="USER_PDF",
            fetched_at=_utc_now_iso(),
            pmcid=current.pmcid,
            notes=f"Content loaded from uploaded PDF: {file_name}",
        )
        self.save(merged)
        return merged

    def update_with_link_text(
        self,
        *,
        pmid: str,
        link_url: str,
        link_text: str,
        source_label: str = "OA_HTML",
        notes: str = "",
    ) -> PaperContent | None:
        current = self.load(pmid)
        if current is None:
            return None
        merged = PaperContent(
            pmid=current.pmid,
            doi=current.doi,
            fulltext_url=str(link_url or current.fulltext_url or ""),
            title=current.title,
            authors=current.authors,
            year=current.year,
            journal=current.journal,
            pubmed_url=current.pubmed_url,
            abstract=current.abstract,
            full_text=str(link_text or ""),
            content_tier=PAPER_TIER_FULL_TEXT,
            source_label=source_label,
            fetched_at=_utc_now_iso(),
            pmcid=current.pmcid,
            notes=str(notes or ""),
        )
        self.save(merged)
        return merged

    def has_cached(self, pmid: str) -> bool:
        return self._paper_path(pmid).exists()

    def _paper_path(self, pmid: str) -> Path:
        safe_pmid = "".join(ch for ch in str(pmid) if ch.isdigit()) or "unknown"
        return self.cache_dir / f"{safe_pmid}.json"


def _paper_content_from_dict(payload: dict[str, Any]) -> PaperContent:
    return PaperContent(
        pmid=str(payload.get("pmid", "") or ""),
        doi=str(payload.get("doi", "") or ""),
        fulltext_url=str(payload.get("fulltext_url", "") or ""),
        title=str(payload.get("title", "") or ""),
        authors=[str(item) for item in (payload.get("authors") or [])],
        year=str(payload.get("year", "") or ""),
        journal=str(payload.get("journal", "") or ""),
        pubmed_url=str(payload.get("pubmed_url", "") or ""),
        abstract=str(payload.get("abstract", "") or ""),
        full_text=str(payload.get("full_text", "") or ""),
        content_tier=str(payload.get("content_tier", PAPER_TIER_ABSTRACT) or PAPER_TIER_ABSTRACT),
        source_label=str(payload.get("source_label", "PUBMED") or "PUBMED"),
        fetched_at=str(payload.get("fetched_at", _utc_now_iso()) or _utc_now_iso()),
        pmcid=str(payload.get("pmcid", "") or ""),
        notes=str(payload.get("notes", "") or ""),
    )


def build_pubmed_url(pmid: str) -> str:
    normalized = str(pmid).strip()
    return f"https://pubmed.ncbi.nlm.nih.gov/{normalized}/" if normalized else ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
