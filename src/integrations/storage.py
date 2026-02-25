from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.core.config import AppConfig

QUERY_CACHE_COLLECTION = "query_cache"
ABSTRACT_COLLECTION = "pubmed_abstracts"
LOGGER = logging.getLogger("pipeline.storage")


def get_vectorstore(config: AppConfig) -> Any:
    """Placeholder for future general-purpose vector store setup."""
    raise NotImplementedError("ChromaDB storage not implemented yet.")


def get_persist_directory(config: AppConfig) -> Path:
    return config.data_dir


def get_query_cache_store(persist_dir: str) -> Chroma:
    return _build_store(persist_dir, QUERY_CACHE_COLLECTION)


def get_abstract_store(persist_dir: str) -> Chroma:
    return _build_store(persist_dir, ABSTRACT_COLLECTION)


def _build_store(persist_dir: str, collection_name: str) -> Chroma:
    persist_path = Path(persist_dir).expanduser().absolute()
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_path),
        embedding_function=embeddings,
    )


def build_query_cache_document(
    query: str,
    pubmed_query: str,
    pmids: list[str] | None = None,
) -> Document:
    normalized_pmids = _normalize_pmids(pmids)
    metadata = {
        "normalized_query": _normalize_query(query),
        "pubmed_query": pubmed_query,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if normalized_pmids:
        metadata["pmids"] = normalized_pmids
        metadata["pmids_str"] = ",".join(normalized_pmids)
    return Document(
        page_content=query,
        metadata=metadata,
    )


def add_query_cache_entry(
    store: Chroma,
    query: str,
    pubmed_query: str,
    pmids: list[str] | None = None,
) -> bool:
    normalized_pmids = _normalize_pmids(pmids)
    if not normalized_pmids:
        LOGGER.info(
            "[PIPELINE] Skipping query cache insert: empty pmids | query='%s' pubmed_query='%s'",
            _trim(query),
            _trim(pubmed_query),
        )
        return False

    doc = build_query_cache_document(query, pubmed_query, normalized_pmids)
    store.add_documents([doc])
    _persist_if_supported(store)
    return True


def upsert_abstracts(
    store: Chroma,
    docs: list[Document],
    *,
    query_text: str = "",
    pubmed_query: str = "",
    log_pipeline: bool = False,
) -> int:
    """Upsert abstract documents by PMID and return count of newly embedded docs.

    Notes:
    - "abstracts fetched from PubMed this query" maps to len(records) in pipeline.
    - "abstracts embedded this query" maps to the returned value from this function.
    - "abstracts used in answer context" maps to retriever k/top_n.
    """
    if not docs:
        if log_pipeline:
            LOGGER.info(
                "[PIPELINE] Embedding skipped: no documents to upsert | query='%s' pubmed_query='%s'",
                _trim(query_text),
                _trim(pubmed_query),
            )
        return 0

    id_to_doc: dict[str, Document] = {}
    for doc in docs:
        pmid = str(doc.metadata.get("pmid", "")).strip()
        if not pmid:
            continue
        id_to_doc[pmid] = doc

    if not id_to_doc:
        if log_pipeline:
            LOGGER.info(
                "[PIPELINE] Embedding skipped: documents had no PMIDs | query='%s' pubmed_query='%s'",
                _trim(query_text),
                _trim(pubmed_query),
            )
        return 0

    ids = list(id_to_doc.keys())
    existing_ids: set[str] = set()
    try:
        existing = store.get(ids=ids, include=["metadatas"])
        existing_ids = set(existing.get("ids", []) or [])
    except Exception:
        existing_ids = set()

    new_ids: list[str] = []
    new_docs: list[Document] = []
    for pmid, doc in id_to_doc.items():
        if pmid in existing_ids:
            continue
        new_ids.append(pmid)
        new_docs.append(doc)

    if not new_docs:
        if log_pipeline:
            LOGGER.info(
                "[PIPELINE] Embedding skipped: all abstracts already present in Chroma | existing=%s requested=%s",
                len(existing_ids),
                len(ids),
            )
        return 0

    if log_pipeline:
        preview_pmids = new_ids[:10]
        suffix = "..." if len(new_ids) > 10 else ""
        before_count = _safe_collection_count(store)
        LOGGER.info(
            "[PIPELINE] Embedding %s abstracts into abstract_store | pmids=%s%s",
            len(new_docs),
            preview_pmids,
            suffix,
        )
        LOGGER.info(
            "[PIPELINE] Embedding is executed by Chroma.add_documents using SentenceTransformerEmbeddings on page_content."
        )
        if before_count is not None:
            LOGGER.info("[PIPELINE] abstract_store count before upsert=%s", before_count)
    else:
        before_count = None

    store.add_documents(new_docs, ids=new_ids)
    _persist_if_supported(store)
    if log_pipeline:
        after_count = _safe_collection_count(store)
        if after_count is None:
            LOGGER.info("[PIPELINE] Abstract upsert complete.")
        else:
            if before_count is None:
                LOGGER.info("[PIPELINE] Abstract upsert complete. collection_count=%s", after_count)
            else:
                delta = after_count - before_count
                LOGGER.info(
                    "[PIPELINE] Abstract upsert complete. before=%s after=%s delta=%s",
                    before_count,
                    after_count,
                    delta,
                )
    return len(new_docs)


def retrieve_relevant_abstracts(
    store: Chroma, query: str, k: int = 6
) -> list[Document]:
    try:
        return store.similarity_search(query, k=k)
    except Exception:
        return []


def lookup_cached_query(
    store: Chroma,
    query: str,
    k: int = 1,
    threshold: float = 0.15,
) -> Dict[str, Any] | None:
    """Return cached metadata + matched query if similarity is above threshold.

    Note: Chroma returns a distance score (lower is more similar).
    """
    results = store.similarity_search_with_score(query, k=k)
    if not results:
        return None

    doc, score = results[0]
    if score is None or score > threshold:
        return None

    payload: Dict[str, Any] = dict(doc.metadata or {})
    payload["matched_query"] = doc.page_content
    payload["score"] = score
    return payload


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def _normalize_pmids(pmids: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for raw in pmids or []:
        pmid = str(raw).strip()
        if pmid:
            normalized.append(pmid)
    return normalized


def _persist_if_supported(store: Chroma) -> None:
    if hasattr(store, "persist"):
        store.persist()


def _safe_collection_count(store: Chroma) -> int | None:
    collection = getattr(store, "_collection", None)
    if collection is None or not hasattr(collection, "count"):
        return None
    try:
        return int(collection.count())
    except Exception:
        return None


def _trim(text: str, limit: int = 120) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."
