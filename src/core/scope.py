from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re

from src.history import get_session_history

REFUSAL_MESSAGE = (
    "I'm focused on cardiovascular topics (heart and blood vessel conditions). "
    "Please rephrase your question to a cardiovascular topic."
)

CARDIO_PHRASES: tuple[str, ...] = (
    "cardiovascular",
    "heart failure",
    "congestive heart failure",
    "myocardial infarction",
    "atrial fibrillation",
    "cardiac arrest",
    "coronary artery",
    "blood pressure",
    "heart valve",
    "aortic valve",
    "mitral valve",
    "tricuspid valve",
    "pulmonary valve",
    "peripheral artery disease",
    "hfpef",
    "hfref",
)

CARDIO_WORDS: tuple[str, ...] = (
    "heart",
    "cardiac",
    "cardiology",
    "coronary",
    "atrial",
    "ventricular",
    "hypertension",
    "hypotension",
    "angina",
    "arrhythmia",
    "arrhythmias",
    "cardiomyopathy",
    "ischemia",
    "ischemic",
    "atherosclerosis",
    "aneurysm",
    "aorta",
    "myocardial",
    "pericarditis",
    "endocarditis",
    "myocarditis",
    "stroke",
    "cerebrovascular",
    "stemi",
    "nstemi",
    "ecg",
    "ekg",
    "echocardiogram",
    "statin",
    "sglt2",
    "antiplatelet",
    "anticoagulant",
)

CARDIO_ABBREV_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmi\b", flags=re.IGNORECASE),
    re.compile(r"\bafib\b", flags=re.IGNORECASE),
    re.compile(r"\ba[\-\s]?fib\b", flags=re.IGNORECASE),
    re.compile(r"\bchf\b", flags=re.IGNORECASE),
    re.compile(r"\bpad\b", flags=re.IGNORECASE),
    re.compile(r"\bhfpef\b", flags=re.IGNORECASE),
    re.compile(r"\bhfref\b", flags=re.IGNORECASE),
)

OVERLAP_PHRASES: tuple[str, ...] = (
    "chronic obstructive pulmonary disease",
    "pulmonary hypertension",
    "pulmonary arterial hypertension",
    "cor pulmonale",
    "sleep apnea",
    "obstructive sleep apnea",
    "pulmonary embolism",
)

OVERLAP_WORDS: tuple[str, ...] = (
    "copd",
    "dyspnea",
    "hypoxemia",
    "hypercapnia",
    "respiratory failure",
    "pulmonary",
)

OVERLAP_ABBREV_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcopd\b", flags=re.IGNORECASE),
    re.compile(r"\bosa\b", flags=re.IGNORECASE),
    re.compile(r"\bpah\b", flags=re.IGNORECASE),
)

FOLLOWUP_PHRASES: tuple[str, ...] = (
    "tell me more",
    "more about",
    "what about",
    "what else",
    "follow up",
    "follow-up",
    "expand on",
    "elaborate",
    "go deeper",
    "what is its",
    "what is their",
    "what about dosage",
    "dosage",
    "dose",
    "safety",
    "side effects",
    "adverse",
    "tolerability",
)

PRONOUN_PATTERN = re.compile(
    r"\b(it|they|this|that|those|these|them|its|their|here|there)\b",
    flags=re.IGNORECASE,
)

STOPWORDS = {
    "the",
    "and",
    "or",
    "but",
    "with",
    "for",
    "from",
    "about",
    "more",
    "tell",
    "what",
    "which",
    "when",
    "where",
    "who",
    "why",
    "how",
    "me",
    "my",
    "we",
    "us",
    "you",
    "your",
    "their",
    "them",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
}


@dataclass(frozen=True)
class ScopeResult:
    label: str
    allow: bool
    user_message: str
    reframed_query: str | None = None
    reason: str | None = None


def classify_scope(query: str, session_id: str, llm: Any | None = None) -> ScopeResult:
    normalized = _normalize(query)
    if not normalized:
        return ScopeResult(
            label="OUT_OF_SCOPE",
            allow=False,
            user_message=REFUSAL_MESSAGE,
            reason="empty_query",
        )

    if _matches_cardio(normalized):
        return ScopeResult(
            label="CARDIOVASCULAR",
            allow=True,
            user_message="ok",
            reason="cardio_heuristic",
        )

    history_text = _get_history_context(session_id)
    ambiguous = _is_ambiguous_followup(normalized)

    if history_text and ambiguous:
        history_norm = _normalize(history_text)
        history_cardio = _matches_cardio(history_norm)
        history_overlap = _matches_overlap(history_norm)
        if history_cardio:
            return ScopeResult(
                label="CARDIOVASCULAR",
                allow=True,
                user_message="ok",
                reason="history_cardio",
            )
        if history_overlap:
            return ScopeResult(
                label="CARDIOPULMONARY_OVERLAP",
                allow=True,
                user_message="ok",
                reason="history_overlap",
            )
        if _query_refers_to_history(normalized, history_norm):
            label = "CARDIOPULMONARY_OVERLAP" if history_overlap else "CARDIOVASCULAR"
            return ScopeResult(
                label=label,
                allow=True,
                user_message="ok",
                reason="history_reference",
            )

    if _matches_overlap(normalized):
        needs_reframe = _needs_overlap_reframe(normalized)
        reframed_query = None
        user_message = "ok"
        if needs_reframe:
            reframed_query = _build_overlap_reframe_query(query, llm)
            user_message = (
                "This is primarily a pulmonary topic. I can answer from a cardiovascular "
                "lens (e.g., effects on heart failure, pulmonary hypertension, arrhythmias, "
                "or cardiovascular risk). I will proceed with that cardiopulmonary overlap view."
            )
        return ScopeResult(
            label="CARDIOPULMONARY_OVERLAP",
            allow=True,
            user_message=user_message,
            reframed_query=reframed_query,
            reason="overlap_heuristic",
        )

    if llm is not None:
        llm_result = _classify_with_llm_json(query, history_text, llm)
        if llm_result:
            label = llm_result.get("label")
            needs_reframe = llm_result.get("needs_reframe", False)
            reframe = llm_result.get("reframe") or None
            reason = llm_result.get("reason")
            if label == "CARDIOVASCULAR":
                return ScopeResult(
                    label="CARDIOVASCULAR",
                    allow=True,
                    user_message="ok",
                    reason=reason,
                )
            if label == "CARDIOPULMONARY_OVERLAP":
                reframed_query = reframe if needs_reframe else None
                if needs_reframe and not reframed_query:
                    reframed_query = _build_overlap_reframe_query(query, llm)
                user_message = "ok"
                if needs_reframe:
                    user_message = (
                        "This is primarily a pulmonary topic. I can answer from a cardiovascular "
                        "lens (e.g., effects on heart failure, pulmonary hypertension, arrhythmias, "
                        "or cardiovascular risk). I will proceed with that cardiopulmonary overlap view."
                    )
                return ScopeResult(
                    label="CARDIOPULMONARY_OVERLAP",
                    allow=True,
                    user_message=user_message,
                    reframed_query=reframed_query,
                    reason=reason,
                )
            if label == "OUT_OF_SCOPE":
                return ScopeResult(
                    label="OUT_OF_SCOPE",
                    allow=False,
                    user_message=_build_refusal_message(query),
                    reason=reason,
                )

    return ScopeResult(
        label="OUT_OF_SCOPE",
        allow=False,
        user_message=_build_refusal_message(query),
        reason="no_match",
    )


def is_cardiovascular_query(query: str, llm: Any | None = None) -> tuple[bool, str]:
    normalized = _normalize(query)
    if not normalized:
        return False, REFUSAL_MESSAGE

    if _matches_cardio(normalized):
        return True, "ok"

    if llm is None:
        return False, REFUSAL_MESSAGE

    classification = _classify_with_llm(query, llm)
    if classification == "CARDIOVASCULAR":
        return True, "ok"
    return False, REFUSAL_MESSAGE


def is_cardiovascular_query_with_history(
    query: str, session_id: str, llm: Any | None = None
) -> tuple[bool, str]:
    result = classify_scope(query, session_id, llm=llm)
    return result.allow, result.user_message


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _matches_cardio(normalized_query: str) -> bool:
    for phrase in CARDIO_PHRASES:
        if phrase in normalized_query:
            return True
    for word in CARDIO_WORDS:
        if _contains_word(normalized_query, word):
            return True
    for pattern in CARDIO_ABBREV_PATTERNS:
        if pattern.search(normalized_query):
            return True
    return False


def _matches_overlap(normalized_query: str) -> bool:
    for phrase in OVERLAP_PHRASES:
        if phrase in normalized_query:
            return True
    for word in OVERLAP_WORDS:
        if _contains_word(normalized_query, word):
            return True
    for pattern in OVERLAP_ABBREV_PATTERNS:
        if pattern.search(normalized_query):
            return True
    if _contains_word(normalized_query, "pulmonary") and _contains_word(
        normalized_query, "hypertension"
    ):
        return True
    if _contains_word(normalized_query, "pulmonary") and _contains_word(
        normalized_query, "embolism"
    ):
        return True
    return False


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _is_ambiguous_followup(normalized_query: str) -> bool:
    tokens = normalized_query.split()
    if len(tokens) < 8:
        return True
    if any(phrase in normalized_query for phrase in FOLLOWUP_PHRASES):
        return True
    if PRONOUN_PATTERN.search(normalized_query):
        return True
    return False


def _needs_overlap_reframe(normalized_query: str) -> bool:
    if _matches_cardio(normalized_query):
        return False
    return True


def _query_refers_to_history(normalized_query: str, history_text: str) -> bool:
    keywords = _extract_keywords(normalized_query)
    if not keywords:
        return False
    for keyword in keywords:
        if keyword in history_text:
            return True
    return False


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-']+", text)
    keywords = []
    for token in tokens:
        lower = token.lower()
        if len(lower) < 4:
            continue
        if lower in STOPWORDS:
            continue
        keywords.append(lower)
    return keywords


def _get_history_context(session_id: str, max_messages: int = 6, max_chars: int = 1200) -> str:
    try:
        history = get_session_history(session_id)
    except Exception:
        return ""

    messages = getattr(history, "messages", [])[-max_messages:]
    lines = []
    for message in messages:
        role = getattr(message, "type", "") or ""
        content = getattr(message, "content", "") or ""
        if not content:
            continue
        label = "assistant" if role in {"ai", "assistant"} else "user"
        lines.append(f"{label}: {content}")

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _classify_with_llm(query: str, llm: Any) -> str | None:
    prompt = (
        "You are a strict classifier for a cardiovascular PubMed assistant.\n"
        "Classify the user query as exactly one of: CARDIOVASCULAR or "
        "NON_CARDIOVASCULAR.\n"
        "Return exactly two lines:\n"
        "CLASSIFICATION: <CARDIOVASCULAR|NON_CARDIOVASCULAR>\n"
        "RATIONALE: <short reason>\n"
        f"Query: {query}\n"
    )

    response_text = _invoke_llm(llm, prompt)
    if not response_text:
        return None
    return _parse_classification(response_text)


def _classify_with_llm_json(
    query: str, history_text: str, llm: Any
) -> dict[str, Any] | None:
    history_snippet = history_text or "(no prior history)"
    prompt = (
        "You are a strict classifier for a cardiovascular PubMed assistant.\n"
        "You handle cardiovascular and cardiopulmonary overlap topics ONLY from a "
        "cardiovascular relevance perspective.\n"
        "If the query is respiratory-only (e.g., 'What is COPD?'), choose "
        "CARDIOPULMONARY_OVERLAP with needs_reframe=true.\n"
        "If the query is cancer/skin/etc with no cardio angle, choose OUT_OF_SCOPE.\n"
        "Return ONLY valid JSON with keys label, needs_reframe, reframe, reason.\n"
        "Valid labels: CARDIOVASCULAR, CARDIOPULMONARY_OVERLAP, OUT_OF_SCOPE.\n"
        f"History: {history_snippet}\n"
        f"Query: {query}\n"
        "JSON:"
    )

    response_text = _invoke_llm(llm, prompt)
    if not response_text:
        return None
    return _parse_scope_json(response_text)


def _build_overlap_reframe_query(query: str, llm: Any | None = None) -> str:
    fallback = (
        f"{query} AND (cardiovascular OR heart OR cardiac OR heart failure OR "
        "myocardial infarction OR arrhythmia OR stroke OR pulmonary hypertension)"
    )
    if llm is None:
        return fallback

    prompt = (
        "Rewrite this pulmonology question into a PubMed-ready query focused on "
        "cardiovascular relevance. Use boolean operators. Return only the query.\n"
        f"Question: {query}\n"
    )
    rewritten = _invoke_llm(llm, prompt)
    if not rewritten:
        return fallback
    return _sanitize_query(rewritten, fallback)


def _build_refusal_message(query: str) -> str:
    return (
        "I'm specialized in cardiovascular and cardiopulmonary overlap topics. "
        "If you'd like, ask how this topic affects cardiovascular outcomes, "
        "heart function, arrhythmias, or vascular risk."
    )


def out_of_scope_message(query: str) -> str:
    return _build_refusal_message(query)


def _invoke_llm(llm: Any, prompt: str) -> str | None:
    try:
        if hasattr(llm, "invoke"):
            result = llm.invoke(prompt)
            return _coerce_to_text(result)
        if hasattr(llm, "predict"):
            result = llm.predict(prompt)
            return _coerce_to_text(result)
    except Exception:
        return None
    return None


def _coerce_to_text(result: Any) -> str:
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def _parse_classification(text: str) -> str | None:
    upper = text.upper()
    for line in upper.splitlines():
        if "CLASSIFICATION" in line:
            if "NON_CARDIOVASCULAR" in line:
                return "NON_CARDIOVASCULAR"
            if "CARDIOVASCULAR" in line:
                return "CARDIOVASCULAR"

    if "NON_CARDIOVASCULAR" in upper:
        return "NON_CARDIOVASCULAR"
    if "CARDIOVASCULAR" in upper:
        return "CARDIOVASCULAR"
    return None


def _parse_scope_json(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    try:
        payload = json.loads(raw)
        return payload
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
            return payload
        except Exception:
            return None


def _sanitize_query(text: str, fallback: str) -> str:
    cleaned = str(text).strip().strip('"').strip("'")
    if not cleaned:
        return fallback
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    if len(cleaned) > 240:
        cleaned = cleaned[:240].rstrip()
    return cleaned or fallback
