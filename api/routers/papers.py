# Created by Codex - Section 1

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_config
from src.core.config import AppConfig
from src.papers.fetch import fetch_paper_content
from src.papers.store import PaperStore

router = APIRouter(prefix="/api/papers", tags=["papers"])


@router.get("/{pmid}")
def get_paper(pmid: str, config: AppConfig = Depends(get_config)) -> dict:
    store = PaperStore(config.data_dir / "papers")
    paper = store.load(pmid)
    if paper is None:
        paper = fetch_paper_content(pmid)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found.")
        store.save(paper)
    return asdict(paper)
