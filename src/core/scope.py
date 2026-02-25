from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re

from src.history import get_session_history

REFUSAL_MESSAGE = (
    "I focus on biomedical and health literature questions. "
    "Please ask a medical, clinical, or public-health question."
)

BIOMEDICAL_PHRASES: tuple[str, ...] = (
    "clinical trial",
    "randomized trial",
    "systematic review",
    "meta analysis",
    "meta-analysis",
    "public health",
    "case control",
    "cohort study",
    "adverse event",
    "side effect",
    "quality of life",
    "disease progression",
    "overall survival",
    "progression free survival",
    "heart failure",
    "atrial fibrillation",
    "myocardial infarction",
    "pulmonary embolism",
    "chronic kidney disease",
    "type 2 diabetes",
    "inflammatory bowel disease",
    "low fodmap",
    "glioblastoma",
)

BIOMEDICAL_WORDS: tuple[str, ...] = (
    "medicine",
    "medical",
    "clinical",
    "patient",
    "patients",
    "trial",
    "therapy",
    "treatment",
    "drug",
    "medication",
    "dose",
    "diagnosis",
    "prognosis",
    "mortality",
    "morbidity",
    "biomarker",
    "pathophysiology",
    "infection",
    "vaccine",
    "antibiotic",
    "oncology",
    "cancer",
    "tumor",
    "carcinoma",
    "chemotherapy",
    "radiotherapy",
    "neurology",
    "brain",
    "stroke",
    "seizure",
    "epilepsy",
    "dementia",
    "parkinson",
    "alzheimer",
    "migraine",
    "gastroenterology",
    "gastrointestinal",
    "gut",
    "ibs",
    "ibd",
    "gerd",
    "microbiome",
    "ulcer",
    "colitis",
    "crohn",
    "hepatitis",
    "liver",
    "renal",
    "kidney",
    "ckd",
    "dialysis",
    "nephrology",
    "pulmonary",
    "respiratory",
    "copd",
    "asthma",
    "pneumonia",
    "cardio",
    "cardiac",
    "heart",
    "hypertension",
    "myocardial",
    "arrhythmia",
    "endocrine",
    "diabetes",
    "thyroid",
    "cholesterol",
    "lipid",
    "blood pressure",
)

NON_BIOMEDICAL_WORDS: tuple[str, ...] = (
    "stock",
    "stocks",
    "crypto",
    "bitcoin",
    "forex",
    "finance",
    "investment",
    "javascript",
    "python code",
    "programming",
    "debug my code",
    "sports",
    "nba",
    "nfl",
    "football",
    "soccer",
    "weather",
    "travel",
    "restaurant",
    "recipe",
    "movie",
    "song",
    "lyrics",
    "politics",
    "election",
)

SYSTEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cardio": ("cardio", "cardiac", "heart", "myocardial", "arrhythmia", "hypertension"),
    "pulmonary": ("pulmonary", "respiratory", "copd", "asthma", "pneumonia"),
    "neuro": ("brain", "stroke", "seizure", "dementia", "parkinson", "neurology"),
    "gi": ("gastro", "gut", "ibs", "ibd", "gerd", "liver", "hepatitis"),
    "renal": ("renal", "kidney", "ckd", "dialysis", "nephrology"),
    "endocrine": ("diabetes", "thyroid", "endocrine", "insulin"),
    "oncology": ("cancer", "tumor", "carcinoma", "metastasis", "chemotherapy", "oncology"),
}

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

MEDICAL_SUFFIX_PATTERN = re.compile(
    r"\b[a-z]{4,}(itis|osis|emia|opathy|plasty|ectomy|oma|algia|genic)\b",
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

    if _matches_biomedical(normalized):
        label = "MULTI_SYSTEM_OVERLAP" if _matches_multisystem(normalized) else "BIOMEDICAL"
        return ScopeResult(
            label=label,
            allow=True,
            user_message="ok",
            reason="biomedical_heuristic",
        )

    history_text = _get_history_context(session_id)
    ambiguous = _is_ambiguous_followup(normalized)
    if history_text and ambiguous:
        history_norm = _normalize(history_text)
        if _matches_biomedical(history_norm):
            label = "MULTI_SYSTEM_OVERLAP" if _matches_multisystem(history_norm) else "BIOMEDICAL"
            return ScopeResult(
                label=label,
                allow=True,
                user_message="ok",
                reason="history_biomedical",
            )
        if _query_refers_to_history(normalized, history_norm):
            return ScopeResult(
                label="BIOMEDICAL",
                allow=True,
                user_message="ok",
                reason="history_reference",
            )

    if llm is not None:
        llm_result = _classify_with_llm_json(query, history_text, llm)
        if llm_result:
            label = str(llm_result.get("label", "") or "").strip().upper()
            reason = str(llm_result.get("reason", "") or "").strip() or None
            needs_reframe = bool(llm_result.get("needs_reframe", False))
            reframe = str(llm_result.get("reframe", "") or "").strip()
            if label in {"BIOMEDICAL", "MULTI_SYSTEM_OVERLAP"}:
                reframed_query = None
                user_message = "ok"
                if needs_reframe:
                    reframed_query = _sanitize_query(reframe, fallback=query)
                    user_message = (
                        "I interpreted this as a biomedical question and reframed the retrieval "
                        "query for clearer evidence search."
                    )
                return ScopeResult(
                    label=label,
                    allow=True,
                    user_message=user_message,
                    reframed_query=reframed_query,
                    reason=reason or "llm_allow",
                )
            if label == "OUT_OF_SCOPE":
                return ScopeResult(
                    label="OUT_OF_SCOPE",
                    allow=False,
                    user_message=_build_refusal_message(query),
                    reason=reason or "llm_out_of_scope",
                )

    if _looks_strongly_non_biomedical(normalized):
        return ScopeResult(
            label="OUT_OF_SCOPE",
            allow=False,
            user_message=_build_refusal_message(query),
            reason="non_biomedical_heuristic",
        )

    if _looks_medical_morphology(normalized):
        label = "MULTI_SYSTEM_OVERLAP" if _matches_multisystem(normalized) else "BIOMEDICAL"
        return ScopeResult(
            label=label,
            allow=True,
            user_message="ok",
            reason="medical_morphology_fallback",
        )

    return ScopeResult(
        label="OUT_OF_SCOPE",
        allow=False,
        user_message=_build_refusal_message(query),
        reason="no_match",
    )


def is_cardiovascular_query(query: str, llm: Any | None = None) -> tuple[bool, str]:
    result = classify_scope(query, session_id="default", llm=llm)
    return result.allow, "ok" if result.allow else result.user_message


def is_cardiovascular_query_with_history(
    query: str, session_id: str, llm: Any | None = None
) -> tuple[bool, str]:
    result = classify_scope(query, session_id, llm=llm)
    return result.allow, result.user_message


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().strip().split())


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _matches_biomedical(normalized_query: str) -> bool:
    for phrase in BIOMEDICAL_PHRASES:
        if phrase in normalized_query:
            return True
    for word in BIOMEDICAL_WORDS:
        if _contains_word(normalized_query, word):
            return True
    return _looks_medical_morphology(normalized_query)


def _looks_medical_morphology(normalized_query: str) -> bool:
    if MEDICAL_SUFFIX_PATTERN.search(normalized_query):
        return True
    # Common clinical phrasing patterns.
    if "evidence for" in normalized_query:
        return True
    if "risk of" in normalized_query:
        return True
    if "survival" in normalized_query or "outcome" in normalized_query:
        return True
    return False


def _matches_multisystem(normalized_query: str) -> bool:
    matched_systems = 0
    for terms in SYSTEM_KEYWORDS.values():
        if any(_contains_word(normalized_query, term) for term in terms):
            matched_systems += 1
            if matched_systems >= 2:
                return True
    return False


def _looks_strongly_non_biomedical(normalized_query: str) -> bool:
    if _matches_biomedical(normalized_query):
        return False
    for term in NON_BIOMEDICAL_WORDS:
        if _contains_word(normalized_query, term):
            return True
    return False


def _is_ambiguous_followup(normalized_query: str) -> bool:
    tokens = normalized_query.split()
    if len(tokens) < 8:
        return True
    if any(phrase in normalized_query for phrase in FOLLOWUP_PHRASES):
        return True
    if PRONOUN_PATTERN.search(normalized_query):
        return True
    return False


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
        "You are a strict classifier for a biomedical literature assistant.\n"
        "Classify the user query as exactly one of: BIOMEDICAL or "
        "NON_BIOMEDICAL.\n"
        "Return exactly two lines:\n"
        "CLASSIFICATION: <BIOMEDICAL|NON_BIOMEDICAL>\n"
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
        "You are a strict classifier for a biomedical PubMed assistant.\n"
        "Return ONLY valid JSON with keys: label, needs_reframe, reframe, reason.\n"
        "Valid labels: BIOMEDICAL, MULTI_SYSTEM_OVERLAP, OUT_OF_SCOPE.\n"
        "Rules:\n"
        "- BIOMEDICAL: medical, clinical, translational, epidemiology, or public-health questions.\n"
        "- MULTI_SYSTEM_OVERLAP: cross-specialty/multi-organ biomedical questions.\n"
        "- OUT_OF_SCOPE: non-biomedical topics (finance, coding help, sports, entertainment).\n"
        "- If query is a follow-up and history shows biomedical context, classify as BIOMEDICAL.\n"
        f"History: {history_snippet}\n"
        f"Query: {query}\n"
        "JSON:"
    )

    response_text = _invoke_llm(llm, prompt)
    if not response_text:
        return None
    return _parse_scope_json(response_text)


def _build_refusal_message(query: str) -> str:
    del query
    return (
        "I can help with biomedical and health literature questions. "
        "Try reframing this as a medical, clinical, or public-health question."
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
            if "NON_BIOMEDICAL" in line:
                return "NON_BIOMEDICAL"
            if "BIOMEDICAL" in line:
                return "BIOMEDICAL"

    if "NON_BIOMEDICAL" in upper:
        return "NON_BIOMEDICAL"
    if "BIOMEDICAL" in upper:
        return "BIOMEDICAL"
    return None


def _parse_scope_json(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else None
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
