from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
from pathlib import Path
from threading import Lock, RLock
import time
from typing import Any, Callable, Dict

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.core.config import AppConfig

QUERY_CACHE_COLLECTION = "query_cache"
ABSTRACT_COLLECTION = "pubmed_abstracts"
DEFAULT_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
LOGGER = logging.getLogger("pipeline.storage")

_STORE_BUILD_LOCK = Lock()
_QUERY_RESULT_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class QueryCacheEntry:
    normalized_query: str
    pubmed_query: str
    pmids: tuple[str, ...]
    created_at: float
    negative: bool = False


class QueryResultCache:
    """Process-local LRU cache for PubMed query results."""

    def __init__(
        self,
        *,
        maxsize: int = 512,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.maxsize = max(32, int(maxsize))
        self._time_func = time_func or time.time
        self._lock = RLock()
        self._entries: OrderedDict[str, QueryCacheEntry] = OrderedDict()

    def get(
        self,
        query: str,
        *,
        ttl_seconds: int,
        negative_ttl_seconds: int,
    ) -> dict[str, Any] | None:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            return None
        with self._lock:
            entry = self._entries.get(normalized_query)
            if entry is None:
                return None
            ttl = max(0, negative_ttl_seconds if entry.negative else ttl_seconds)
            age_seconds = self._time_func() - float(entry.created_at)
            if ttl and age_seconds > ttl:
                self._entries.pop(normalized_query, None)
                return None
            self._entries.move_to_end(normalized_query)
            return {
                "normalized_query": entry.normalized_query,
                "pubmed_query": entry.pubmed_query,
                "pmids": list(entry.pmids),
                "created_at_epoch": entry.created_at,
                "negative": entry.negative,
                "cache_layer": "memory",
            }

    def set(self, query: str, *, pubmed_query: str, pmids: list[str] | None = None) -> QueryCacheEntry:
        normalized_query = _normalize_query(query)
        normalized_pmids = tuple(_normalize_pmids(pmids))
        entry = QueryCacheEntry(
            normalized_query=normalized_query,
            pubmed_query=str(pubmed_query or query or "").strip(),
            pmids=normalized_pmids,
            created_at=self._time_func(),
            negative=not bool(normalized_pmids),
        )
        with self._lock:
            self._entries[normalized_query] = entry
            self._entries.move_to_end(normalized_query)
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)
        return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def get_vectorstore(config: AppConfig) -> Any:
    """Placeholder for future general-purpose vector store setup."""
    raise NotImplementedError("ChromaDB storage not implemented yet.")


def get_persist_directory(config: AppConfig) -> Path:
    return config.data_dir


def get_embeddings(model_name: str = DEFAULT_EMBEDDINGS_MODEL) -> SentenceTransformerEmbeddings:
    return _get_embeddings_cached(str(model_name or DEFAULT_EMBEDDINGS_MODEL).strip() or DEFAULT_EMBEDDINGS_MODEL)


def get_query_cache_store(
    persist_dir: str,
    *,
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL,
) -> Chroma:
    return _build_store_cached(
        str(Path(persist_dir).expanduser().absolute()),
        QUERY_CACHE_COLLECTION,
        embeddings_model_name,
    )


def get_abstract_store(
    persist_dir: str,
    *,
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL,
) -> Chroma:
    return _build_store_cached(
        str(Path(persist_dir).expanduser().absolute()),
        ABSTRACT_COLLECTION,
        embeddings_model_name,
    )


def get_query_result_cache(maxsize: int = 512) -> QueryResultCache:
    with _QUERY_RESULT_CACHE_LOCK:
        return _get_query_result_cache_cached(max(32, int(maxsize)))


def lookup_query_result_cache(
    query: str,
    *,
    store: Chroma | None,
    ttl_seconds: int,
    negative_ttl_seconds: int,
    threshold: float = 0.15,
    in_memory_cache: QueryResultCache | None = None,
) -> dict[str, Any] | None:
    cache = in_memory_cache or get_query_result_cache()
    cached = cache.get(
        query,
        ttl_seconds=ttl_seconds,
        negative_ttl_seconds=negative_ttl_seconds,
    )
    if cached is not None:
        return cached
    if store is None:
        return None
    persistent = lookup_cached_query(store, query, threshold=threshold)
    if persistent is None:
        return None
    if _cache_payload_expired(
        persistent,
        ttl_seconds=ttl_seconds,
        negative_ttl_seconds=negative_ttl_seconds,
        now_epoch=time.time(),
    ):
        return None
    entry = cache.set(
        query,
        pubmed_query=str(persistent.get("pubmed_query") or query),
        pmids=_normalize_pmids(persistent.get("pmids")),
    )
    payload = {
        "normalized_query": entry.normalized_query,
        "pubmed_query": entry.pubmed_query,
        "pmids": list(entry.pmids),
        "created_at_epoch": entry.created_at,
        "negative": entry.negative,
        "cache_layer": "persistent",
    }
    return payload


def remember_query_result(
    query: str,
    *,
    pubmed_query: str,
    pmids: list[str] | None,
    store: Chroma | None = None,
    in_memory_cache: QueryResultCache | None = None,
) -> bool:
    normalized_pmids = _normalize_pmids(pmids)
    cache = in_memory_cache or get_query_result_cache()
    cache.set(query, pubmed_query=pubmed_query, pmids=normalized_pmids)
    if not normalized_pmids or store is None:
        return False
    return add_query_cache_entry(
        store,
        query,
        pubmed_query=pubmed_query,
        pmids=normalized_pmids,
    )


@lru_cache(maxsize=8)
def _get_embeddings_cached(model_name: str) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name=model_name)


@lru_cache(maxsize=64)
def _build_store_cached(
    persist_dir: str,
    collection_name: str,
    embeddings_model_name: str,
) -> Chroma:
    persist_path = Path(persist_dir).expanduser().absolute()
    with _STORE_BUILD_LOCK:
        # Local dev note: if upgrading Chroma versions, delete the persist directory to reinitialize it.
        persist_path.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(embeddings_model_name)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_path),
        embedding_function=embeddings,
    )


@lru_cache(maxsize=4)
def _get_query_result_cache_cached(maxsize: int) -> QueryResultCache:
    return QueryResultCache(maxsize=maxsize)


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
        metadata["pmids_json"] = json.dumps(normalized_pmids)
        metadata["pmids_csv"] = ",".join(normalized_pmids)
    return Document(page_content=query, metadata=sanitize_metadata(metadata))


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
    """Upsert abstract documents by PMID and return count of newly embedded docs."""
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
        pmid = str((doc.metadata or {}).get("pmid", "")).strip()
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
        new_docs.append(_sanitize_document_metadata(doc))

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
        elif before_count is None:
            LOGGER.info("[PIPELINE] Abstract upsert complete. collection_count=%s", after_count)
        else:
            LOGGER.info(
                "[PIPELINE] Abstract upsert complete. before=%s after=%s delta=%s",
                before_count,
                after_count,
                after_count - before_count,
            )
    return len(new_docs)


def retrieve_relevant_abstracts(store: Chroma, query: str, k: int = 6) -> list[Document]:
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
    """Return cached metadata + matched query if similarity is above threshold."""
    try:
        results = store.similarity_search_with_score(query, k=k)
    except Exception:
        return None
    if not results:
        return None

    doc, score = results[0]
    if score is None or score > threshold:
        return None

    payload: Dict[str, Any] = dict(doc.metadata or {})
    payload["matched_query"] = doc.page_content
    payload["score"] = score
    payload["pmids"] = _extract_pmids_from_payload(payload)
    return payload


def _cache_payload_expired(
    payload: dict[str, Any],
    *,
    ttl_seconds: int,
    negative_ttl_seconds: int,
    now_epoch: float,
) -> bool:
    created_at_epoch = _coerce_created_at_epoch(payload.get("created_at"))
    if created_at_epoch is None:
        return False
    pmids = _normalize_pmids(payload.get("pmids"))
    ttl = negative_ttl_seconds if not pmids else ttl_seconds
    return bool(ttl and (now_epoch - created_at_epoch) > ttl)


def _extract_pmids_from_payload(payload: dict[str, Any]) -> list[str]:
    pmids = payload.get("pmids")
    normalized = _normalize_pmids(pmids if isinstance(pmids, list) else None)
    if normalized:
        return normalized
    pmids_json = str(payload.get("pmids_json", "") or "").strip()
    if pmids_json:
        try:
            parsed = json.loads(pmids_json)
        except json.JSONDecodeError:
            parsed = None
        normalized = _normalize_pmids(parsed if isinstance(parsed, list) else None)
        if normalized:
            return normalized
    for key in ("pmids_csv", "pmids_str", "pmids"):
        pmids_value = payload.get(key)
        if isinstance(pmids_value, str):
            normalized = _normalize_pmids(pmids_value.split(","))
            if normalized:
                return normalized
    return []


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool):
            sanitized[str(key)] = value
            continue
        if isinstance(value, (str, int, float)):
            sanitized[str(key)] = value
            continue
        if isinstance(value, (list, tuple, dict)):
            sanitized[str(key)] = json.dumps(value)
            continue
        sanitized[str(key)] = str(value)
    return sanitized


def _sanitize_document_metadata(doc: Document) -> Document:
    return Document(
        page_content=str(getattr(doc, "page_content", "") or ""),
        metadata=sanitize_metadata(dict(getattr(doc, "metadata", {}) or {})),
    )


def _coerce_created_at_epoch(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").lower().strip().split())


def _normalize_pmids(pmids: list[str] | tuple[str, ...] | None) -> list[str]:
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
