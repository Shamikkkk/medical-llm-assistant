from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Any

from langchain_core.documents import Document

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
    seen_pmids: set[str] = set()
    for doc in docs:
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
) -> list[Document]:
    if not docs:
        return []
    alpha = max(0.0, min(1.0, float(alpha)))
    semantic_ranks = {id(doc): idx + 1 for idx, doc in enumerate(docs)}
    bm25_ranked = _bm25_rank_documents(query, docs)
    bm25_ranks = {id(doc): idx + 1 for idx, doc in enumerate(bm25_ranked)}

    def _score(doc: Document) -> float:
        semantic_rank = semantic_ranks.get(id(doc), len(docs) + 1)
        bm25_rank = bm25_ranks.get(id(doc), len(docs) + 1)
        return (alpha * (1.0 / (60 + semantic_rank))) + ((1.0 - alpha) * (1.0 / (60 + bm25_rank)))

    ranked = sorted(docs, key=_score, reverse=True)
    return ranked[: max(1, int(limit))]


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
    abstract = str(getattr(doc, "page_content", "") or "").strip()
    title = str(metadata.get("title", "") or "").strip()
    if title and abstract.lower().startswith(title.lower()):
        abstract = abstract[len(title) :].lstrip()
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


def _bm25_rank_documents(query: str, docs: list[Document]) -> list[Document]:
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return list(docs)
    tokenized_docs = [_tokenize(getattr(doc, "page_content", "")) for doc in docs]
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(1, len(tokenized_docs))
    doc_freqs: dict[str, int] = {}
    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freqs[token] = doc_freqs.get(token, 0) + 1

    k1 = 1.5
    b = 0.75
    ranked: list[tuple[float, Document]] = []
    for idx, tokens in enumerate(tokenized_docs):
        score = 0.0
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        doc_len = max(1, len(tokens))
        for token in tokenized_query:
            if token not in frequencies:
                continue
            df = doc_freqs.get(token, 0)
            idf = math.log(1 + ((len(docs) - df + 0.5) / (df + 0.5)))
            numerator = frequencies[token] * (k1 + 1)
            denominator = frequencies[token] + (k1 * (1 - b + b * (doc_len / max(1.0, avg_doc_len))))
            score += idf * (numerator / denominator)
        ranked.append((score, docs[idx]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked]


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
