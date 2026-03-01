from __future__ import annotations

from functools import lru_cache
import logging
import math
import re

from langchain_core.documents import Document

LOGGER = logging.getLogger("pipeline.retrieval")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CLAIM_HINT_RE = re.compile(
    r"\b(reduce|reduces|reduced|increase|increases|increased|risk|mortality|bleeding|benefit|associated|significant|trial|odds|hazard|dose|dosage|cure|cured|improve|improved)\b",
    flags=re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def count_tokens(text: str) -> int:
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(str(text or "")))
        except Exception:
            pass
    compact = str(text or "").strip()
    if not compact:
        return 0
    return max(1, math.ceil(len(compact) / 4))


def select_context_documents(docs: list[Document], *, max_abstracts: int) -> list[Document]:
    limited: list[Document] = []
    deduplicated_docs = deduplicate_by_semantic_similarity(docs, threshold=0.92)
    seen_pmids: set[str] = set()
    for doc in deduplicated_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        pmid = str(metadata.get("pmid", "") or "").strip()
        if pmid and pmid in seen_pmids:
            continue
        if pmid:
            seen_pmids.add(pmid)
        limited.append(doc)
        if len(limited) >= max(1, int(max_abstracts)):
            break
    return limited


def build_context_rows(
    docs: list[Document],
    *,
    max_abstracts: int,
    max_context_tokens: int,
    trim_strategy: str,
) -> list[dict[str, str]]:
    selected_docs = select_context_documents(docs, max_abstracts=max_abstracts)
    if not selected_docs:
        return []
    rows: list[dict[str, str]] = []
    token_budget = max(1, int(max_context_tokens))
    remaining_budget = token_budget
    remaining_docs = len(selected_docs)

    for doc in selected_docs:
        separator_budget = 0 if not rows else count_tokens("\n\n---\n\n")
        remaining_budget = max(0, remaining_budget - separator_budget)
        if remaining_budget <= 0:
            break
        row_budget = max(16, remaining_budget // max(1, remaining_docs))
        row = _build_context_row(doc, token_budget=row_budget, trim_strategy=trim_strategy)
        row_text = row_to_context_text(row)
        row_tokens = count_tokens(row_text)
        if row_tokens > remaining_budget:
            row = _build_context_row(
                doc,
                token_budget=max(8, remaining_budget),
                trim_strategy="truncate",
            )
            row_text = row_to_context_text(row)
            row_tokens = count_tokens(row_text)
        if row_tokens > remaining_budget and rows:
            break
        rows.append(row)
        remaining_budget = max(0, remaining_budget - row_tokens)
        remaining_docs -= 1
        if remaining_budget <= 8:
            break
    return rows


def build_context_text(
    docs: list[Document],
    *,
    max_abstracts: int,
    max_context_tokens: int,
    trim_strategy: str,
) -> str:
    rows = build_context_rows(
        docs,
        max_abstracts=max_abstracts,
        max_context_tokens=max_context_tokens,
        trim_strategy=trim_strategy,
    )
    text = rows_to_context_text(rows)
    if count_tokens(text) <= max_context_tokens:
        return text
    return _clip_text_to_token_budget(text, token_budget=max(1, int(max_context_tokens)))


def rows_to_context_text(rows: list[dict[str, str]]) -> str:
    sections = [row_to_context_text(row) for row in rows if row_to_context_text(row)]
    return "\n\n---\n\n".join(sections)


def row_to_context_text(row: dict[str, str]) -> str:
    parts = [
        f"PMID: {str(row.get('pmid', '') or '').strip()}",
        f"Title: {str(row.get('title', '') or '').strip()}",
        f"Journal: {str(row.get('journal', '') or '').strip()}",
        f"Year: {str(row.get('year', '') or '').strip()}",
        f"Abstract: {str(row.get('context', '') or '').strip()}",
    ]
    return "\n".join(parts).strip()


def hybrid_rerank_documents(
    query: str,
    docs: list[Document],
    *,
    alpha: float,
    limit: int,
    explain_reranking: bool = False,
) -> list[Document]:
    if not docs:
        return []

    clipped_alpha = max(0.0, min(1.0, float(alpha)))
    query_tokens = set(_tokenize(query))
    bm25_scores = _normalize_scores(_bm25_document_scores(query, docs))
    semantic_scores = _normalize_scores(
        [_semantic_document_score(doc, query_tokens) for doc in docs]
    )

    scored_docs: list[tuple[float, float, float, int, Document]] = []
    for index, doc in enumerate(docs):
        final_score = (
            clipped_alpha * semantic_scores[index]
            + (1.0 - clipped_alpha) * bm25_scores[index]
        )
        scored_docs.append(
            (final_score, semantic_scores[index], bm25_scores[index], index, doc)
        )

    ranked = sorted(
        scored_docs,
        key=lambda item: (-item[0], -item[1], -item[2], item[3]),
    )
    ranked_docs = [doc for _, _, _, _, doc in ranked[: max(1, int(limit))]]

    if explain_reranking:
        _log_reranking_changes(docs, ranked_docs, ranked, clipped_alpha)

    return ranked_docs


def deduplicate_by_semantic_similarity(
    docs: list[Document],
    threshold: float = 0.92,
) -> list[Document]:
    deduplicated: list[Document] = []
    seen_token_sets: list[set[str]] = []
    clipped_threshold = max(0.0, min(1.0, float(threshold)))

    for doc in docs:
        tokens = set(_tokenize(_document_abstract_text(doc)))
        if not tokens:
            deduplicated.append(doc)
            continue
        duplicate = False
        for seen_tokens in seen_token_sets:
            similarity = _jaccard_similarity(tokens, seen_tokens)
            if similarity >= clipped_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        deduplicated.append(doc)
        seen_token_sets.append(tokens)
    return deduplicated


def align_answer_citations(
    answer: str,
    *,
    contexts: list[dict[str, str]] | list[str],
    mode: str = "disclaim",
) -> tuple[str, list[str]]:
    text = str(answer or "").strip()
    if not text:
        return text, []
    raw_contexts = [
        str(item.get("context", "") or "").strip() if isinstance(item, dict) else str(item or "").strip()
        for item in contexts
    ]
    normalized_contexts = [item for item in raw_contexts if item]
    sentences = _split_sentences(text)
    if not sentences or not normalized_contexts:
        return text, []

    issues: list[str] = []
    output_sentences: list[str] = []
    for sentence in sentences:
        if not _looks_like_medical_claim(sentence):
            output_sentences.append(sentence)
            continue
        if _sentence_supported(sentence, normalized_contexts):
            output_sentences.append(sentence)
            continue
        issues.append(sentence)
        if mode == "remove":
            continue
        output_sentences.append(f"{sentence} [No supporting PMID found in retrieved abstracts]")
    return " ".join(output_sentences).strip(), issues


def _build_context_row(
    doc: Document,
    *,
    token_budget: int,
    trim_strategy: str,
) -> dict[str, str]:
    metadata = getattr(doc, "metadata", {}) or {}
    abstract = _document_abstract_text(doc)
    title = str(metadata.get("title", "") or "").strip()
    if trim_strategy == "compress":
        abstract = _compress_text(abstract, max_sentences=2)
    abstract = _clip_text_to_token_budget(abstract, token_budget=max(24, token_budget))
    return {
        "pmid": str(metadata.get("pmid", "") or "").strip(),
        "title": title,
        "journal": str(metadata.get("journal", "") or "").strip(),
        "year": str(metadata.get("year", "") or "").strip(),
        "context": abstract,
    }


def _compress_text(text: str, *, max_sentences: int) -> str:
    sentences = _split_sentences(str(text or "").strip())
    if not sentences:
        return str(text or "").strip()
    return " ".join(sentences[: max(1, int(max_sentences))]).strip()


def _clip_text_to_token_budget(text: str, *, token_budget: int) -> str:
    compact = str(text or "").strip()
    if not compact:
        return compact
    if token_budget <= 0:
        return ""
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        try:
            ids = encoder.encode(compact)
            if len(ids) <= token_budget:
                return compact
            clipped = encoder.decode(ids[:token_budget]).strip()
            return clipped
        except Exception:
            pass
    approx_chars = max(16, int(token_budget) * 4)
    return compact[:approx_chars].rstrip()


def _bm25_document_scores(query: str, docs: list[Document]) -> list[float]:
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return [0.0 for _ in docs]

    tokenized_docs = [_tokenize(getattr(doc, "page_content", "")) for doc in docs]
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(1, len(tokenized_docs))
    doc_freqs: dict[str, int] = {}
    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freqs[token] = doc_freqs.get(token, 0) + 1

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens in tokenized_docs:
        score = 0.0
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        doc_len = max(1, len(tokens))
        length_norm = 1 - b + b * (doc_len / max(1.0, avg_doc_len))
        for token in tokenized_query:
            if token not in frequencies:
                continue
            df = doc_freqs.get(token, 0)
            idf = math.log(1 + ((len(docs) - df + 0.5) / (df + 0.5)))
            numerator = frequencies[token] * (k1 + 1)
            denominator = frequencies[token] + (k1 * length_norm)
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores


def _bm25_rank_documents(query: str, docs: list[Document]) -> list[Document]:
    scores = _bm25_document_scores(query, docs)
    ranked = sorted(
        enumerate(docs),
        key=lambda item: (-scores[item[0]], item[0]),
    )
    return [doc for _, doc in ranked]


def _semantic_document_score(doc: Document, query_tokens: set[str]) -> float:
    metadata = getattr(doc, "metadata", {}) or {}
    if metadata.get("score") is not None:
        try:
            return max(0.0, float(metadata.get("score")))
        except (TypeError, ValueError):
            pass
    if metadata.get("_distance") is not None:
        try:
            distance = float(metadata.get("_distance"))
            return 1.0 / (1.0 + max(0.0, distance))
        except (TypeError, ValueError):
            pass
    doc_tokens = set(_tokenize(getattr(doc, "page_content", "")))
    if not query_tokens or not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / max(1, len(query_tokens))


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        if maximum <= 0:
            return [0.0 for _ in values]
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _log_reranking_changes(
    original_docs: list[Document],
    ranked_docs: list[Document],
    ranked_scores: list[tuple[float, float, float, int, Document]],
    alpha: float,
) -> None:
    original_order = {id(doc): index for index, doc in enumerate(original_docs, start=1)}
    reranked_order = {id(doc): index for index, doc in enumerate(ranked_docs, start=1)}
    movements: list[str] = []
    for final_score, semantic_score, bm25_score, _, doc in ranked_scores[: min(6, len(ranked_scores))]:
        before = original_order.get(id(doc), 0)
        after = reranked_order.get(id(doc), before)
        if before == after:
            continue
        pmid = str((getattr(doc, "metadata", {}) or {}).get("pmid", "") or "?")
        direction = "promoted" if after < before else "demoted"
        movements.append(
            f"PMID {pmid} {direction} {before}->{after} "
            f"(final={final_score:.3f}, semantic={semantic_score:.3f}, bm25={bm25_score:.3f}, alpha={alpha:.2f})"
        )
    if movements:
        LOGGER.info("[RERANK] %s", " | ".join(movements))


def _document_abstract_text(doc: Document) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    text = str(getattr(doc, "page_content", "") or "").strip()
    title = str(metadata.get("title", "") or "").strip()
    if title and text.lower().startswith(title.lower()):
        return text[len(title) :].lstrip()
    return text


def _looks_like_medical_claim(sentence: str) -> bool:
    lowered = str(sentence or "").strip().lower()
    if not lowered:
        return False
    return bool(_CLAIM_HINT_RE.search(lowered) or _NUMERIC_RE.search(lowered))


def _sentence_supported(sentence: str, contexts: list[str]) -> bool:
    sentence_tokens = set(_tokenize(sentence))
    if not sentence_tokens:
        return True
    for context in contexts:
        context_tokens = set(_tokenize(context))
        if not context_tokens:
            continue
        overlap = len(sentence_tokens & context_tokens)
        if overlap >= max(3, math.ceil(len(sentence_tokens) * 0.35)):
            return True
    return False


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _tokenize(text: str) -> list[str]:
    normalized = str(text or "").lower()
    normalized = re.sub(r"\baf\b", "atrial fibrillation", normalized)
    return _TOKEN_RE.findall(normalized)


def _split_sentences(text: str) -> list[str]:
    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(str(text or "").strip())]
    return [sentence for sentence in sentences if sentence]


@lru_cache(maxsize=1)
def _get_tiktoken_encoder():
    try:
        import tiktoken
    except Exception:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None
