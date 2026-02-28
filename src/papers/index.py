from __future__ import annotations

from hashlib import sha1
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from src.papers.store import PaperContent

LOGGER = logging.getLogger("papers.index")


class PaperIndexer:
    """Per-paper Chroma index for focused follow-up retrieval."""

    def __init__(self, root_dir: str | Path = "./data/papers") -> None:
        self.root_dir = Path(root_dir).expanduser().absolute()
        self.persist_dir = self.root_dir / "chroma"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    def index_paper(self, paper: PaperContent) -> int:
        text = paper.primary_text.strip()
        if not text:
            return 0
        chunks = _chunk_text(text, chunk_size=1800, chunk_overlap=300)
        if not chunks:
            return 0
        source_hash = sha1(text.encode("utf-8")).hexdigest()
        store = self._get_store(paper.pmid)
        existing_hash = _get_existing_source_hashes(store)
        if source_hash in existing_hash:
            return 0

        docs: list[Document] = []
        ids: list[str] = []
        source_type = _source_type_from_label(paper.source_label)
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "pmid": paper.pmid,
                        "title": paper.title,
                        "journal": paper.journal,
                        "year": paper.year,
                        "doi": paper.doi,
                        "fulltext_url": paper.fulltext_url,
                        "content_tier": paper.content_tier,
                        "source_label": paper.source_label,
                        "source_type": source_type,
                        "source_hash": source_hash,
                        "chunk_index": idx,
                    },
                )
            )
            ids.append(f"{paper.pmid}::{source_hash[:10]}::{idx}")

        store.add_documents(docs, ids=ids)
        _persist_if_supported(store)
        LOGGER.info(
            "[PAPER] Indexed paper pmid=%s chunks=%s tier=%s",
            paper.pmid,
            len(docs),
            paper.content_tier,
        )
        return len(docs)

    def retrieve(self, pmid: str, query: str, k: int = 6) -> list[Document]:
        store = self._get_store(pmid)
        try:
            docs = store.similarity_search(query, k=max(1, min(10, int(k))))
        except Exception:
            docs = []
        return docs

    def has_index(self, pmid: str) -> bool:
        store = self._get_store(pmid)
        collection = getattr(store, "_collection", None)
        if collection is None:
            return False
        try:
            return int(collection.count()) > 0
        except Exception:
            return False

    def _get_store(self, pmid: str) -> Chroma:
        collection_name = _collection_for_pmid(pmid)
        return Chroma(
            collection_name=collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=self._embeddings,
        )


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        return [chunk for chunk in splitter.split_text(cleaned) if chunk.strip()]
    except Exception:
        chunks: list[str] = []
        step = max(200, chunk_size - chunk_overlap)
        for idx in range(0, len(cleaned), step):
            part = cleaned[idx : idx + chunk_size].strip()
            if part:
                chunks.append(part)
        return chunks


def _collection_for_pmid(pmid: str) -> str:
    digits = "".join(ch for ch in str(pmid) if ch.isdigit()) or "unknown"
    return f"paper_{digits}"


def _get_existing_source_hashes(store: Chroma) -> set[str]:
    hashes: set[str] = set()
    try:
        payload = store.get(include=["metadatas"])
    except Exception:
        return hashes
    for metadata in payload.get("metadatas", []) or []:
        if not isinstance(metadata, dict):
            continue
        source_hash = str(metadata.get("source_hash", "") or "").strip()
        if source_hash:
            hashes.add(source_hash)
    return hashes


def _persist_if_supported(store: Chroma) -> None:
    if hasattr(store, "persist"):
        store.persist()


def _source_type_from_label(source_label: str) -> str:
    lowered = str(source_label or "").lower()
    if "pdf" in lowered:
        return "pdf"
    if "pmc" in lowered:
        return "pmc"
    if "html" in lowered or "oa" in lowered:
        return "html"
    return "unknown"
