from __future__ import annotations

import re
from typing import Any

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def build_claim_premise(
    *,
    claim: str,
    evidence_chunks: list[str],
    tokenizer: Any,
    top_n_chunks: int = 4,
    top_k_sentences: int = 2,
    max_premise_tokens: int = 384,
) -> dict[str, Any]:
    cleaned_chunks = [str(chunk).strip() for chunk in evidence_chunks if str(chunk).strip()]
    if not cleaned_chunks:
        return {
            "premise": "",
            "tokens_used": 0,
            "truncated": False,
            "selected_chunk_count": 0,
            "selected_sentence_count": 0,
        }

    ranked_chunks = _rank_by_claim_similarity(claim, cleaned_chunks)[: max(1, int(top_n_chunks))]
    selected_sentences: list[str] = []
    for _, chunk in ranked_chunks:
        sentences = _split_sentences(chunk)
        if not sentences:
            continue
        ranked_sentences = _rank_by_claim_similarity(claim, sentences)
        take_n = max(1, int(top_k_sentences))
        best = [item for _, item in ranked_sentences[:take_n]]
        selected_sentences.extend(best)

    unique_sentences = _dedupe_preserve_order(selected_sentences)
    premise, token_count, truncated = _pack_with_budget(
        sentences=unique_sentences,
        tokenizer=tokenizer,
        max_tokens=max(32, int(max_premise_tokens)),
    )

    return {
        "premise": premise,
        "tokens_used": token_count,
        "truncated": truncated,
        "selected_chunk_count": len(ranked_chunks),
        "selected_sentence_count": len(unique_sentences),
    }


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    sentences = [part.strip() for part in parts if part and part.strip()]
    return sentences


def _rank_by_claim_similarity(claim: str, candidates: list[str]) -> list[tuple[float, str]]:
    claim_tokens = _tokens(claim)
    ranked: list[tuple[float, str]] = []
    for candidate in candidates:
        candidate_tokens = _tokens(candidate)
        if not candidate_tokens:
            ranked.append((0.0, candidate))
            continue
        overlap = len(claim_tokens & candidate_tokens)
        denom = len(claim_tokens | candidate_tokens) or 1
        jaccard = overlap / denom
        ranked.append((jaccard, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = " ".join(item.split()).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(item.strip())
    return output


def _pack_with_budget(*, sentences: list[str], tokenizer: Any, max_tokens: int) -> tuple[str, int, bool]:
    if not sentences:
        return "", 0, False

    assembled: list[str] = []
    truncated = False
    for sentence in sentences:
        candidate = "\n".join(assembled + [sentence]) if assembled else sentence
        token_count = _count_tokens(tokenizer, candidate)
        if token_count > max_tokens:
            truncated = True
            break
        assembled.append(sentence)

    if not assembled:
        first = sentences[0]
        clipped, token_count, clipped_flag = _clip_to_token_budget(first, tokenizer, max_tokens)
        return clipped, token_count, clipped_flag

    premise = "\n".join(assembled).strip()
    token_count = _count_tokens(tokenizer, premise)
    return premise, token_count, truncated


def _clip_to_token_budget(text: str, tokenizer: Any, max_tokens: int) -> tuple[str, int, bool]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text, len(ids), False
        clipped_ids = ids[:max_tokens]
        clipped = tokenizer.decode(
            clipped_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        return clipped, len(clipped_ids), True
    except Exception:
        words = text.split()
        if len(words) <= max_tokens:
            return text, len(words), False
        clipped = " ".join(words[:max_tokens]).strip()
        return clipped, max_tokens, True


def _count_tokens(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())
