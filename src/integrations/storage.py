from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
from threading import Lock, RLock
import time
from typing import Any, Callable, Dict, Iterable, Mapping

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.core.config import AppConfig

QUERY_CACHE_COLLECTION = "query_cache"
ABSTRACT_COLLECTION = "pubmed_abstracts"
ANSWER_CACHE_COLLECTION = "answer_cache"
DEFAULT_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
QUERY_CACHE_PMID_STORE_LIMIT = 50
LOGGER = logging.getLogger("pipeline.storage")
ANSWER_CACHE_FINGERPRINT_VERSION = "display-top-n-v2"

_STORE_BUILD_LOCK = Lock()
_QUERY_RESULT_CACHE_LOCK = Lock()
_VALID_COMPUTE_DEVICE_PREFERENCES = {"auto", "cpu", "gpu", "cuda"}


@dataclass(frozen=True)
class QueryCacheEntry:
    normalized_query: str
    pubmed_query: str
    pmids: tuple[str, ...]
    created_at: float
    requested_retmax: int
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
                "requested_retmax": entry.requested_retmax,
                "negative": entry.negative,
                "cache_layer": "memory",
            }

    def set(
        self,
        query: str,
        *,
        pubmed_query: str,
        pmids: list[str] | None = None,
        requested_retmax: int | None = None,
    ) -> QueryCacheEntry:
        normalized_query = _normalize_query(query)
        normalized_pmids = tuple(_normalize_pmids(pmids))
        resolved_retmax = max(
            len(normalized_pmids),
            int(requested_retmax if requested_retmax is not None else len(normalized_pmids)),
        )
        entry = QueryCacheEntry(
            normalized_query=normalized_query,
            pubmed_query=str(pubmed_query or query or "").strip(),
            pmids=normalized_pmids,
            created_at=self._time_func(),
            requested_retmax=resolved_retmax,
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


def normalize_compute_device_preference(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower() or "auto"
    if normalized not in _VALID_COMPUTE_DEVICE_PREFERENCES:
        return "auto"
    if normalized == "cuda":
        return "gpu"
    return normalized


def resolve_compute_device(preference: str | None = "auto") -> tuple[str, str | None]:
    normalized = normalize_compute_device_preference(preference)
    if normalized == "cpu":
        return "cpu", None

    try:
        import torch
    except Exception:
        if normalized == "gpu":
            return "cpu", "GPU requested, but torch/CUDA is unavailable. Falling back to CPU."
        return "cpu", None

    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    if normalized == "gpu":
        if cuda_available:
            return "cuda", None
        return "cpu", "GPU requested, but CUDA is unavailable. Falling back to CPU."
    return ("cuda" if cuda_available else "cpu"), None


def get_embeddings(
    model_name: str = DEFAULT_EMBEDDINGS_MODEL,
    *,
    device: str | None = None,
) -> SentenceTransformerEmbeddings:
    resolved_device, _ = resolve_compute_device(device)
    normalized_model = str(model_name or DEFAULT_EMBEDDINGS_MODEL).strip() or DEFAULT_EMBEDDINGS_MODEL
    return _get_embeddings_cached(normalized_model, resolved_device)


def get_query_cache_store(
    persist_dir: str,
    *,
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL,
    embeddings_device: str | None = None,
) -> Chroma:
    return _build_store_cached(
        str(Path(persist_dir).expanduser().absolute()),
        QUERY_CACHE_COLLECTION,
        embeddings_model_name,
        resolve_compute_device(embeddings_device)[0],
    )


def get_abstract_store(
    persist_dir: str,
    *,
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL,
    embeddings_device: str | None = None,
) -> Chroma:
    return _build_store_cached(
        str(Path(persist_dir).expanduser().absolute()),
        ABSTRACT_COLLECTION,
        embeddings_model_name,
        resolve_compute_device(embeddings_device)[0],
    )


def get_answer_cache_store(
    persist_dir: str,
    *,
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL,
    embeddings_device: str | None = None,
) -> Chroma:
    return _build_store_cached(
        str(Path(persist_dir).expanduser().absolute()),
        ANSWER_CACHE_COLLECTION,
        embeddings_model_name,
        resolve_compute_device(embeddings_device)[0],
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
    persistent_pmids = _normalize_pmids(persistent.get("pmids"))[:QUERY_CACHE_PMID_STORE_LIMIT]
    entry = cache.set(
        query,
        pubmed_query=str(persistent.get("pubmed_query") or query),
        pmids=persistent_pmids,
        requested_retmax=int(
            persistent.get("requested_retmax", len(persistent_pmids)) or len(persistent_pmids)
        ),
    )
    return {
        "normalized_query": entry.normalized_query,
        "pubmed_query": entry.pubmed_query,
        "pmids": list(entry.pmids),
        "created_at_epoch": entry.created_at,
        "requested_retmax": entry.requested_retmax,
        "negative": entry.negative,
        "cache_layer": "persistent",
    }


def remember_query_result(
    query: str,
    *,
    pubmed_query: str,
    pmids: list[str] | None,
    requested_retmax: int | None = None,
    store: Chroma | None = None,
    in_memory_cache: QueryResultCache | None = None,
) -> bool:
    normalized_pmids = _normalize_pmids(pmids)[:QUERY_CACHE_PMID_STORE_LIMIT]
    cache = in_memory_cache or get_query_result_cache()
    cache.set(
        query,
        pubmed_query=pubmed_query,
        pmids=normalized_pmids,
        requested_retmax=requested_retmax,
    )
    if not normalized_pmids or store is None:
        return False
    return add_query_cache_entry(
        store,
        query,
        pubmed_query=pubmed_query,
        pmids=normalized_pmids,
        requested_retmax=requested_retmax,
    )


def build_answer_cache_fingerprint(
    *,
    config: Any,
    top_n: int,
    include_paper_links: bool,
    backend: str,
) -> str:
    fingerprint_payload = {
        "fingerprint_version": ANSWER_CACHE_FINGERPRINT_VERSION,
        "backend": str(backend or "baseline"),
        "nvidia_model": str(getattr(config, "nvidia_model", "") or ""),
        "use_reranker": bool(getattr(config, "use_reranker", False)),
        "hybrid_retrieval": bool(getattr(config, "hybrid_retrieval", False)),
        "hybrid_alpha": float(getattr(config, "hybrid_alpha", 0.5) or 0.5),
        "max_abstracts": int(getattr(config, "max_abstracts", 8) or 8),
        "max_context_abstracts": int(
            getattr(config, "max_context_abstracts", getattr(config, "max_abstracts", 8)) or 8
        ),
        "max_context_tokens": int(getattr(config, "max_context_tokens", 2500) or 2500),
        "context_trim_strategy": str(getattr(config, "context_trim_strategy", "truncate") or "truncate"),
        "citation_alignment": bool(getattr(config, "citation_alignment", False)),
        "alignment_mode": str(getattr(config, "alignment_mode", "disclaim") or "disclaim"),
        "validator_enabled": bool(getattr(config, "validator_enabled", False)),
        "validator_model_name": str(getattr(config, "validator_model_name", "") or ""),
        "validator_threshold": float(getattr(config, "validator_threshold", 0.7) or 0.7),
        "validator_margin": float(getattr(config, "validator_margin", 0.2) or 0.2),
        "top_n": int(top_n),
        "include_paper_links": bool(include_paper_links),
    }
    serialized = json.dumps(fingerprint_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def lookup_answer_cache(
    query_text: str,
    *,
    store: Chroma | None,
    config_fingerprint: str,
    ttl_seconds: int,
    min_similarity: float = 0.9,
    top_k: int = 3,
    strict_fingerprint: bool = True,
    now_epoch: float | None = None,
) -> dict[str, Any] | None:
    if store is None:
        return None
    normalized_query = _normalize_query(query_text)
    if not normalized_query:
        return None

    now_epoch = now_epoch if now_epoch is not None else time.time()
    candidates = _lookup_answer_cache_exact(store, normalized_query)
    if not candidates:
        candidates = _lookup_answer_cache_similar(
            store,
            query_text,
            min_similarity=min_similarity,
            top_k=top_k,
        )

    valid_candidates: list[dict[str, Any]] = []
    loose_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = _deserialize_answer_cache_payload(candidate)
        if payload is None:
            continue
        created_at_epoch = _coerce_created_at_epoch(candidate.get("created_at"))
        if created_at_epoch is not None and ttl_seconds and (now_epoch - created_at_epoch) > ttl_seconds:
            continue
        config_match = str(candidate.get("config_fingerprint", "") or "") == str(config_fingerprint or "")
        result = {
            "response_payload": payload,
            "matched_query": str(candidate.get("matched_query", "") or ""),
            "created_at": str(candidate.get("created_at", "") or ""),
            "similarity": float(candidate.get("similarity", 0.0) or 0.0),
            "match_type": str(candidate.get("match_type", "similar") or "similar"),
            "config_match": config_match,
        }
        if config_match:
            valid_candidates.append(result)
        elif not strict_fingerprint:
            loose_candidates.append(result)

    if valid_candidates:
        return sorted(valid_candidates, key=_answer_cache_sort_key, reverse=True)[0]
    if loose_candidates:
        best = sorted(loose_candidates, key=_answer_cache_sort_key, reverse=True)[0]
        best["note"] = "Cached answer reused with a different runtime fingerprint."
        return best
    return None


def store_answer_cache(
    query_text: str,
    *,
    response_payload: Mapping[str, Any],
    config_fingerprint: str,
    store: Chroma | None,
    model_id: str = "",
    backend: str = "baseline",
) -> bool:
    if store is None:
        return False
    payload = _sanitize_cached_response(response_payload)
    if str(payload.get("status", "") or "") != "answered":
        return False
    if not str(payload.get("answer", "") or "").strip():
        return False

    created_at = datetime.now(timezone.utc).isoformat()
    normalized_query = _normalize_query(query_text)
    sources = payload.get("sources", []) or []
    source_pmids = [
        str(item.get("pmid", "")).strip()
        for item in sources
        if isinstance(item, Mapping) and str(item.get("pmid", "")).strip()
    ]
    response_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    metadata = sanitize_metadata(
        {
            "normalized_query": normalized_query,
            "config_fingerprint": str(config_fingerprint or ""),
            "model_id": str(model_id or ""),
            "backend": str(backend or "baseline"),
            "created_at": created_at,
            "response_payload_json": response_json,
            "source_pmids_json": json.dumps(source_pmids),
            "status": str(payload.get("status", "") or ""),
        }
    )
    doc = Document(page_content=str(query_text or ""), metadata=metadata)
    cache_id = hashlib.sha256(
        f"{normalized_query}:{config_fingerprint}:{created_at}".encode("utf-8")
    ).hexdigest()
    store.add_documents([doc], ids=[cache_id])
    _persist_if_supported(store)
    return True


def clear_query_result_caches(
    persist_dir: str,
    *,
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL,
    embeddings_device: str | None = None,
) -> int:
    _get_query_result_cache_cached.cache_clear()
    store = get_query_cache_store(
        persist_dir,
        embeddings_model_name=embeddings_model_name,
        embeddings_device=embeddings_device,
    )
    return clear_store_documents(store)


def clear_answer_cache(
    persist_dir: str,
    *,
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL,
    embeddings_device: str | None = None,
) -> int:
    store = get_answer_cache_store(
        persist_dir,
        embeddings_model_name=embeddings_model_name,
        embeddings_device=embeddings_device,
    )
    return clear_store_documents(store)


@lru_cache(maxsize=16)
def _get_embeddings_cached(model_name: str, device: str) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )


@lru_cache(maxsize=96)
def _build_store_cached(
    persist_dir: str,
    collection_name: str,
    embeddings_model_name: str,
    embeddings_device: str,
) -> Chroma:
    persist_path = Path(persist_dir).expanduser().absolute()
    with _STORE_BUILD_LOCK:
        # Local dev note: if upgrading Chroma versions, delete the persist directory to reinitialize it.
        persist_path.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(
        embeddings_model_name,
        device=embeddings_device,
    )
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
    requested_retmax: int | None = None,
) -> Document:
    normalized_pmids = _normalize_pmids(pmids)[:QUERY_CACHE_PMID_STORE_LIMIT]
    resolved_retmax = max(
        len(normalized_pmids),
        int(requested_retmax if requested_retmax is not None else len(normalized_pmids)),
    )
    metadata = {
        "normalized_query": _normalize_query(query),
        "pubmed_query": pubmed_query,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_retmax": resolved_retmax,
        "pmid_count": len(normalized_pmids),
    }
    if normalized_pmids:
        metadata["pmids_json"] = json.dumps(normalized_pmids)
        metadata["pmids_csv"] = ",".join(normalized_pmids)
        metadata["pmids_up_to_50_json"] = json.dumps(normalized_pmids)
        metadata["pmids_up_to_50_csv"] = ",".join(normalized_pmids)
    return Document(page_content=query, metadata=sanitize_metadata(metadata))


def add_query_cache_entry(
    store: Chroma,
    query: str,
    pubmed_query: str,
    pmids: list[str] | None = None,
    requested_retmax: int | None = None,
) -> bool:
    normalized_pmids = _normalize_pmids(pmids)
    if not normalized_pmids:
        LOGGER.info(
            "[PIPELINE] Skipping query cache insert: empty pmids | query='%s' pubmed_query='%s'",
            _trim(query),
            _trim(pubmed_query),
        )
        return False

    doc = build_query_cache_document(
        query,
        pubmed_query,
        normalized_pmids,
        requested_retmax=requested_retmax,
    )
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


def clear_store_documents(store: Chroma) -> int:
    try:
        payload = store.get(include=[])
    except Exception:
        payload = {}
    ids = [str(item).strip() for item in (payload.get("ids", []) or []) if str(item).strip()]
    if not ids:
        return 0
    try:
        store.delete(ids=ids)
    except Exception:
        collection = getattr(store, "_collection", None)
        if collection is None:
            return 0
        try:
            collection.delete(ids=ids)
        except Exception:
            return 0
    _persist_if_supported(store)
    return len(ids)


def _lookup_answer_cache_exact(store: Chroma, normalized_query: str) -> list[dict[str, Any]]:
    try:
        payload = store.get(
            where={"normalized_query": normalized_query},
            include=["metadatas", "documents"],
        )
    except Exception:
        return []

    ids = payload.get("ids", []) or []
    documents = payload.get("documents", []) or []
    metadatas = payload.get("metadatas", []) or []
    candidates: list[dict[str, Any]] = []
    for idx, cache_id in enumerate(ids):
        metadata = dict(metadatas[idx] or {}) if idx < len(metadatas) else {}
        matched_query = documents[idx] if idx < len(documents) else normalized_query
        metadata.update(
            {
                "cache_id": cache_id,
                "matched_query": matched_query,
                "similarity": 1.0,
                "match_type": "exact",
            }
        )
        candidates.append(metadata)
    return candidates


def _lookup_answer_cache_similar(
    store: Chroma,
    query_text: str,
    *,
    min_similarity: float,
    top_k: int,
) -> list[dict[str, Any]]:
    try:
        results = store.similarity_search_with_relevance_scores(
            query_text,
            k=max(1, int(top_k)),
            score_threshold=max(0.0, min(1.0, float(min_similarity))),
        )
    except Exception:
        return []

    candidates: list[dict[str, Any]] = []
    for doc, similarity in results:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        metadata.update(
            {
                "matched_query": str(getattr(doc, "page_content", "") or ""),
                "similarity": float(similarity or 0.0),
                "match_type": "similar",
            }
        )
        candidates.append(metadata)
    return candidates


def _deserialize_answer_cache_payload(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = str(candidate.get("response_payload_json", "") or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("status", "") or "") != "answered":
        return None
    if not str(payload.get("answer", "") or "").strip():
        return None
    return payload


def _answer_cache_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float]:
    created_at_epoch = _coerce_created_at_epoch(candidate.get("created_at")) or 0.0
    return (
        float(candidate.get("similarity", 0.0) or 0.0),
        created_at_epoch,
    )


def _sanitize_cached_response(response_payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(response_payload or {}), ensure_ascii=False, sort_keys=True))


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
    for json_key in ("pmids_up_to_50_json", "pmids_json"):
        pmids_json = str(payload.get(json_key, "") or "").strip()
        if not pmids_json:
            continue
        try:
            parsed = json.loads(pmids_json)
        except json.JSONDecodeError:
            parsed = None
        normalized = _normalize_pmids(parsed if isinstance(parsed, list) else None)
        if normalized:
            return normalized[:QUERY_CACHE_PMID_STORE_LIMIT]
    for key in ("pmids_up_to_50_csv", "pmids_csv", "pmids_str", "pmids"):
        pmids_value = payload.get(key)
        if isinstance(pmids_value, str):
            normalized = _normalize_pmids(pmids_value.split(","))
            if normalized:
                return normalized[:QUERY_CACHE_PMID_STORE_LIMIT]
    pmids = payload.get("pmids")
    normalized = _normalize_pmids(pmids if isinstance(pmids, list) else None)
    if normalized:
        return normalized[:QUERY_CACHE_PMID_STORE_LIMIT]
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


def _normalize_pmids(pmids: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in pmids or []:
        pmid = str(raw).strip()
        if pmid and pmid not in seen:
            seen.add(pmid)
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
