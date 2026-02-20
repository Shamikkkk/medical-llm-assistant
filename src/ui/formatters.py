from __future__ import annotations

import re


def strip_reframe_block(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # Remove a leading "Reframe:" block up to first blank line or "Direct answer:".
    cleaned = re.sub(
        r"^\s*reframe:\s*.*?(?=\n\s*\n|\n\s*direct answer\s*:|$)\n?",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Remove standalone "Reframe:" lines anywhere.
    cleaned = re.sub(r"(?im)^\s*reframe:\s*.*\n?", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def beautify_text(text: str) -> str:
    if text is None:
        return ""
    cleaned = strip_reframe_block(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return ""

    # Keep citation clusters readable, e.g. "...[123][456]" -> "... [123][456]"
    cleaned = re.sub(r"(?<=\S)(\[\d)", r" \1", cleaned)

    lines = cleaned.split("\n")
    non_empty = [line for line in lines if line.strip()]
    avg_len = (
        sum(len(line) for line in non_empty) / len(non_empty)
        if non_empty
        else len(cleaned)
    )

    if "\n" not in cleaned or avg_len > 180:
        cleaned = re.sub(r"([.!?])\s+", r"\1\n\n", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def pubmed_url(pmid: str) -> str:
    normalized = str(pmid or "").strip()
    return f"https://pubmed.ncbi.nlm.nih.gov/{normalized}/"


def doi_url(doi: str) -> str:
    normalized = str(doi or "").strip()
    if not normalized:
        return ""
    return f"https://doi.org/{normalized}"
