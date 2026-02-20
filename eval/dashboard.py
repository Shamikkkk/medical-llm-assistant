from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

from eval.store import EvalStore


def render_evaluation_dashboard(store_path: str) -> None:
    st.subheader("Evaluation Dashboard")
    store = EvalStore(store_path)
    rows = store.read_all()
    if not rows:
        st.info("No evaluation runs recorded yet.")
        return

    date_filter_days = st.selectbox("Window", options=[1, 7, 30, 90, 365], index=2)
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(date_filter_days))
    filtered = [row for row in rows if _parse_ts(row.get("timestamp")) >= cutoff]
    if not filtered:
        st.info("No records in selected time window.")
        return

    metrics = [
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
        "citation_alignment",
        "safety_compliance",
    ]
    st.markdown("### Rolling Averages", unsafe_allow_html=False)
    columns = st.columns(3)
    for idx, metric in enumerate(metrics):
        value = _avg_metric(filtered, metric)
        columns[idx % 3].metric(metric.replace("_", " ").title(), f"{value:.3f}")

    st.markdown("### Trend", unsafe_allow_html=False)
    trend_metric = st.selectbox("Metric", options=metrics, index=0)
    trend_data = _build_trend_data(filtered, trend_metric)
    st.line_chart(trend_data)

    st.markdown("### Failure Examples", unsafe_allow_html=False)
    low_rows = sorted(
        filtered,
        key=lambda item: float((item.get("metrics") or {}).get("faithfulness", 0.0)),
    )[:10]
    for row in low_rows:
        query = str(row.get("query", "") or "")
        metrics_row = row.get("metrics") or {}
        with st.expander(f"Query: {query[:120]}", expanded=False):
            st.markdown(
                f"- Faithfulness: {float(metrics_row.get('faithfulness', 0.0)):.3f}\n"
                f"- Relevance: {float(metrics_row.get('answer_relevance', 0.0)):.3f}\n"
                f"- Citation alignment: {float(metrics_row.get('citation_alignment', 0.0)):.3f}\n"
                f"- Safety: {float(metrics_row.get('safety_compliance', 0.0)):.3f}",
                unsafe_allow_html=False,
            )
            st.markdown("**Answer**", unsafe_allow_html=False)
            st.markdown(str(row.get("answer", "") or ""), unsafe_allow_html=False)

    st.markdown("### Raw Records", unsafe_allow_html=False)
    st.dataframe(filtered, use_container_width=True)


def _avg_metric(rows: list[dict[str, Any]], metric: str) -> float:
    values = []
    for row in rows:
        metrics = row.get("metrics") or {}
        value = metrics.get(metric)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_trend_data(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    points = sorted(rows, key=lambda item: str(item.get("timestamp", "")))
    labels: list[str] = []
    values: list[float] = []
    for row in points:
        labels.append(str(row.get("timestamp", "")))
        metrics = row.get("metrics") or {}
        try:
            values.append(float(metrics.get(metric, 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return {"timestamp": labels, metric: values}


def _parse_ts(value: Any) -> datetime:
    raw = str(value or "")
    if not raw:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
