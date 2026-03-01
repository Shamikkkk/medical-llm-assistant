# Created by Codex - Section 1

from __future__ import annotations

import re

_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gi": (
        "gut",
        "gastro",
        "gastric",
        "stomach",
        "ibs",
        "ibd",
        "gerd",
        "microbiome",
        "diarrhea",
        "constipation",
        "ulcer",
        "colitis",
        "crohn",
        "hepatitis",
        "liver",
    ),
    "cardio": (
        "heart",
        "cardiac",
        "cardio",
        "afib",
        "atrial fibrillation",
        "mi",
        "myocardial",
        "hypertension",
        "lipid",
        "cholesterol",
        "arrhythmia",
        "heart failure",
    ),
    "pulmonary": (
        "copd",
        "asthma",
        "lung",
        "pneumonia",
        "pulmonary",
        "respiratory",
        "oxygenation",
        "ventilation",
    ),
    "neuro": (
        "brain",
        "stroke",
        "seizure",
        "epilepsy",
        "dementia",
        "parkinson",
        "alzheimer",
        "migraine",
        "neurology",
    ),
    "oncology": (
        "cancer",
        "tumor",
        "carcinoma",
        "chemo",
        "chemotherapy",
        "radiotherapy",
        "metastasis",
        "oncology",
        "glioblastoma",
    ),
    "renal": (
        "kidney",
        "renal",
        "ckd",
        "dialysis",
        "nephrology",
        "creatinine",
    ),
    "endo": (
        "diabetes",
        "thyroid",
        "endocrine",
        "insulin",
        "glucose",
        "hormone",
    ),
}

_TOPIC_MESSAGES: dict[str, str] = {
    "gi": "Digesting the literature...",
    "cardio": "Checking the evidence pulse...",
    "pulmonary": "Taking a deep breath and searching...",
    "neuro": "Firing up neurons and scanning evidence...",
    "oncology": "Scanning tumor biology and trial evidence...",
    "renal": "Filtering the nephrology evidence...",
    "endo": "Balancing endocrine evidence...",
    "general": "Thinking...",
}


def detect_topic(query: str) -> str:
    normalized = _normalize(query)
    if not normalized:
        return "general"
    for topic in ("oncology", "neuro", "gi", "pulmonary", "cardio", "renal", "endo"):
        if _contains_any(normalized, _TOPIC_KEYWORDS[topic]):
            return topic
    return "general"


def pick_loading_message(topic: str, query: str) -> str:
    del query
    normalized_topic = str(topic or "").strip().lower()
    return _TOPIC_MESSAGES.get(normalized_topic, _TOPIC_MESSAGES["general"])


def _normalize(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[\W_]+", " ", lowered)
    return " ".join(lowered.split())


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return True
    return False
