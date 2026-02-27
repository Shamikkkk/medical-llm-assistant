from __future__ import annotations

import streamlit as st

from src.observability.metrics import read_metric_events, summarize_metric_events


def render_metrics_dashboard(store_path: str) -> None:
    st.subheader("Metrics Dashboard")
    events = read_metric_events(store_path)
    if not events:
        st.info("No metrics events recorded yet.")
        return
    summary = summarize_metric_events(events)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Requests", int(summary["total_requests"]))
    col2.metric("Error Rate", f"{float(summary['error_rate']):.2%}")
    col3.metric("Cache Hit Rate", f"{float(summary['cache_hit_rate']):.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Latency P50 (ms)", f"{float(summary['latency_p50_ms']):.1f}")
    col5.metric("Latency P95 (ms)", f"{float(summary['latency_p95_ms']):.1f}")
    col6.metric("Avg PMID Count", f"{float(summary['avg_pmid_count']):.2f}")

    st.markdown("### Raw Events", unsafe_allow_html=False)
    st.dataframe(events, use_container_width=True)
