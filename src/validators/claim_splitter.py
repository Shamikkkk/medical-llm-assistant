from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(r"^\s*[-*+]\s*")
_NUMBERED_BULLET_RE = re.compile(r"^\s*\d+[\).\s]+")


def split_into_claims(answer: str, *, max_claims: int = 12) -> list[str]:
    text = _normalize_answer(answer)
    if not text:
        return []

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_sentences: list[str] = []
    for line in raw_lines:
        cleaned_line = _strip_bullet_prefix(line)
        if not cleaned_line:
            continue
        segments = _SENTENCE_SPLIT_RE.split(cleaned_line)
        for segment in segments:
            normalized = _normalize_sentence(segment)
            if normalized:
                candidate_sentences.append(normalized)

    if not candidate_sentences:
        candidate_sentences = [_normalize_sentence(text)]

    claims: list[str] = []
    for sentence in candidate_sentences:
        if not sentence:
            continue
        claims.append(sentence)
        if len(claims) >= max_claims:
            break
    return claims


def _normalize_answer(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    return normalized


def _strip_bullet_prefix(line: str) -> str:
    without_symbol = _BULLET_RE.sub("", line)
    without_number = _NUMBERED_BULLET_RE.sub("", without_symbol)
    return without_number.strip()


def _normalize_sentence(sentence: str) -> str:
    compact = _WHITESPACE_RE.sub(" ", sentence).strip()
    return compact
