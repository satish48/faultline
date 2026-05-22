"""
Faultline — AgentWatch Streamlit Dashboard
===========================================
Autonomous Agent Observability UI.

Usage:
    streamlit run agentwatch/dashboard/app.py

API contract (confirmed from route inspection):
  POST /audit                       → AuditReport (direct object, not wrapped)
  POST /conflicts/scan/{run_id}     → List[ConflictSummary]  (raw list, no conflicting_outputs)
  GET  /conflicts/patterns          → List[ConflictPattern]  (raw list)
  GET  /conflicts/recommendations   → List[str]              (raw list)
  GET  /health                      → health check
"""

from __future__ import annotations

import os

import streamlit as st
import plotly.graph_objects as go
import httpx

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Faultline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helper functions ──────────────────────────────────────────────────────────

def api_get(url: str, timeout: int = 30):
    """
    Synchronous GET. Returns (data, error_str).
    data is parsed JSON on success, None on failure.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json(), None
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return None, detail
    except Exception as exc:
        return None, str(exc)


def api_post(url: str, body: dict, timeout: int = 30):
    """
    Synchronous POST with JSON body. Returns (data, error_str).
    data is parsed JSON on success, None on failure.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            return resp.json(), None
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return None, detail
    except Exception as exc:
        return None, str(exc)


def unwrap_list(data) -> list:
    """
    Accept raw list OR common wrapped shapes:
      {"data": [...]}  {"patterns": [...]}  {"recommendations": [...]}
    Returns a plain list in all cases.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "patterns", "recommendations", "results", "items", "conflicts"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def safe_get(d, key: str, default=None):
    """Safely get a key from a dict; return default if missing or d is not a dict."""
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def risk_badge(level: str) -> str:
    """Colored HTML badge for a risk level string."""
    level = (level or "").lower()
    color_map = {
        "high":   "#E8341A",
        "medium": "#F5A623",
        "low":    "#2ECC71",
    }
    label_map = {
        "high":   "🔴 HIGH",
        "medium": "🟡 MEDIUM",
        "low":    "🟢 LOW",
    }
    color = color_map.get(level, "#888888")
    label = label_map.get(level, level.upper() if level else "UNKNOWN")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-weight:bold;font-size:0.85em;">{label}</span>'
    )


def bounded_float(v, default: float = 0.0) -> float:
    """Clamp a value to [0.0, 1.0]; return default if value is non-numeric."""
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _humanize(s: str) -> str:
    """Replace underscores with spaces and title-case a string."""
    return str(s).replace("_", " ").title() if s else "—"


# ── Sidebar ───────────────────────────────────────────────────────────────────

default_url = os.getenv(
    "API_BASE_URL",
    "https://faultline-production-e04c.up.railway.app"
)

with st.sidebar:
    st.markdown("# ⚡ Faultline")
    st.caption("Autonomous Agent Observability")
    st.divider()

    # Advanced Settings expander — collapsed by default so it stays out of the way
    with st.expander("⚙️ Advanced Settings", expanded=False):
        api_url = st.text_input("API URL", value=default_url)
    base_url = api_url.rstrip("/")

    # Connection status — always visible, outside the expander
    try:
        with httpx.Client(timeout=5) as _client:
            _resp = _client.get(f"{base_url}/health")
        _connected = _resp.status_code == 200
    except Exception:
        _connected = False

    if _connected:
        st.markdown(
            '<span style="color:#2ECC71;font-weight:bold;">● Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#E8341A;font-weight:bold;">● Disconnected</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Model:** Groq llama-3.3-70b")


# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="padding:1.6rem 0 0.4rem;">
      <h1 style="font-size:2.6rem;font-weight:800;margin-bottom:0.25rem;letter-spacing:-0.5px;">
        ⚡ Faultline
      </h1>
      <p style="font-size:1.15rem;color:#9ca3af;margin-bottom:0.6rem;font-weight:500;">
        Autonomous Agent Observability &amp; Governance
      </p>
      <p style="font-size:0.95rem;color:#6b7280;max-width:720px;line-height:1.6;margin:0;">
        Faultline instruments multi-agent pipelines to audit every decision, detect
        inter-agent conflicts in real time — semantic, procedural, and temporal — and
        surface specific architectural remediation recommendations before failures reach
        production.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── System status cards ────────────────────────────────────────────────────────

_status_label    = "● Online" if _connected else "● Offline"
_backend_display = base_url.replace("https://", "").replace("http://", "")

sc1, sc2, sc3 = st.columns(3)
sc1.metric("System Status",   _status_label)
sc2.metric("Model Provider",  "Groq · llama-3.3-70b")
sc3.metric("API Backend",     _backend_display)

st.divider()

# ── Main tabs ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔍 Decision Audit",
    "⚡ Conflict Monitor",
    "🧠 Memory Intelligence",
])


# ═══════════════════════════════════════════════════════════
# TAB 1 — Decision Audit
# ═══════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Decision Audit")
    st.caption(
        "Paste any Run ID to generate a full grounded audit report — "
        "risk classification, decision trail, counterfactual sensitivity, and risk factors."
    )

    run_id_audit = st.text_input(
        "Run ID",
        key="audit_run_id",
        value="demo-session-001",
    )
    st.caption("Use **demo-session-001** to load the pre-recorded demo run with no setup required.")

    if st.button("Run Audit", key="btn_audit"):
        if not run_id_audit.strip():
            st.warning("Please enter a Run ID.")
        else:
            with st.spinner("Running audit pipeline…"):
                audit_data, audit_err = api_post(
                    f"{base_url}/audit",
                    {
                        "run_id": run_id_audit.strip(),
                        "include_counterfactuals": True,
                        "max_counterfactuals": 5,
                    },
                )

            if audit_err:
                st.error(f"Audit failed: {audit_err}")

            elif audit_data is not None:
                # Pull fields defensively
                overall_risk    = safe_get(audit_data, "overall_risk", "unknown")
                steps           = unwrap_list(safe_get(audit_data, "steps", []))
                critical_nodes  = unwrap_list(safe_get(audit_data, "critical_decisions", []))
                counterfactuals = unwrap_list(safe_get(audit_data, "counterfactuals", []))
                key_finding     = safe_get(audit_data, "key_finding", "No key finding available.")
                risk_factors    = unwrap_list(safe_get(audit_data, "risk_factors", []))

                # ── Metrics row ──────────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall Risk",       str(overall_risk).upper() if overall_risk else "—")
                m2.metric("Audit Steps",        len(steps))
                m3.metric("Critical Decisions", len(critical_nodes))
                m4.metric("Counterfactuals",    len(counterfactuals))

                # Colored risk badge — st.metric can't custom-color
                st.markdown(risk_badge(overall_risk), unsafe_allow_html=True)
                st.divider()

                # ── Key Finding ──────────────────────────────────────────────
                st.subheader("Key Finding")
                st.info(key_finding)

                # ── Audit Trail ──────────────────────────────────────────────
                if steps:
                    st.subheader("Audit Trail")
                    for step in steps:
                        step_no    = safe_get(step, "step_no", "?")
                        agent_name = safe_get(step, "agent_name", "unknown")
                        risk_level = safe_get(step, "risk_level", "low")
                        label      = f"Step {step_no} · {agent_name} · {str(risk_level).upper()}"

                        with st.expander(label):
                            action    = safe_get(step, "action",    "—")
                            reasoning = safe_get(step, "reasoning", "—")
                            outcome   = safe_get(step, "outcome",   "—")
                            src_ids   = unwrap_list(safe_get(step, "source_event_ids", []))

                            st.markdown(f"**Action:** {action}")
                            st.markdown(f"**Reasoning:** {reasoning}")
                            st.markdown(f"**Outcome:** {outcome}")
                            st.markdown(
                                f"**Risk:** {risk_badge(risk_level)}",
                                unsafe_allow_html=True,
                            )
                            if src_ids:
                                st.markdown(
                                    "**Source Events:** " + ", ".join(str(s) for s in src_ids)
                                )

                # ── Counterfactual Analysis ──────────────────────────────────
                if counterfactuals:
                    st.subheader("Counterfactual Analysis")
                    for cf in counterfactuals:
                        div_point       = safe_get(cf, "divergence_point",  "unknown")
                        outcome_changed = safe_get(cf, "outcome_changed",   False)
                        sensitivity     = bounded_float(safe_get(cf, "sensitivity_score", 0.0))
                        explanation     = safe_get(cf, "explanation",       "")

                        c1, c2, c3 = st.columns([2, 1, 2])
                        c1.markdown(f"**Divergence point:** `{div_point}`")

                        if outcome_changed:
                            c2.markdown(
                                '<span style="color:#2ECC71;font-weight:bold;">✓ Changed</span>',
                                unsafe_allow_html=True,
                            )
                        else:
                            c2.markdown(
                                '<span style="color:#E8341A;">✗ Unchanged</span>',
                                unsafe_allow_html=True,
                            )

                        with c3:
                            st.markdown(f"**Sensitivity:** {sensitivity:.2f}")
                            st.progress(sensitivity)

                        if explanation:
                            st.caption(explanation)
                        st.divider()

                # ── Risk Factors ─────────────────────────────────────────────
                if risk_factors:
                    st.subheader("Risk Factors")
                    for factor in risk_factors:
                        st.warning(str(factor))


# ═══════════════════════════════════════════════════════════
# TAB 2 — Conflict Monitor
# ═══════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Conflict Monitor")
    st.caption(
        "Scan any agent run for inter-agent conflicts. "
        "Detects semantic contradictions, procedural collisions, and temporal staleness — "
        "with root cause attribution and autonomous resolution strategy."
    )

    run_id_conflict = st.text_input(
        "Run ID",
        key="conflict_run_id",
        value="demo-session-001",
    )
    st.caption("Use **demo-session-001** to scan the pre-recorded 3-conflict demo run.")

    if st.button("Scan for Conflicts", key="btn_conflict"):
        if not run_id_conflict.strip():
            st.warning("Please enter a Run ID.")
        else:
            with st.spinner("Scanning for conflicts…"):
                # POST /conflicts/scan/{run_id} — no request body
                conflict_data, conflict_err = api_post(
                    f"{base_url}/conflicts/scan/{run_id_conflict.strip()}",
                    {},
                )

            if conflict_err:
                st.error(f"Conflict scan failed: {conflict_err}")

            elif conflict_data is not None:
                conflicts = unwrap_list(conflict_data)

                # ── Type breakdown ────────────────────────────────────────────
                def _count_type(clist: list, ctype: str) -> int:
                    return sum(
                        1 for c in clist
                        if str(safe_get(c, "conflict_type", "")).lower() == ctype
                    )

                n_semantic   = _count_type(conflicts, "semantic")
                n_procedural = _count_type(conflicts, "procedural")
                n_temporal   = _count_type(conflicts, "temporal")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Conflicts", len(conflicts))
                m2.metric("Semantic",        n_semantic)
                m3.metric("Procedural",      n_procedural)
                m4.metric("Temporal",        n_temporal)

                # ── Plotly bar chart ──────────────────────────────────────────
                fig = go.Figure(
                    go.Bar(
                        x=["Semantic", "Procedural", "Temporal"],
                        y=[n_semantic, n_procedural, n_temporal],
                        marker_color=["#E8341A", "#F5A623", "#F0C040"],
                    )
                )
                fig.update_layout(
                    plot_bgcolor="#0a0a0a",
                    paper_bgcolor="#0a0a0a",
                    font=dict(color="white"),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Per-conflict detail ───────────────────────────────────────
                # Note: scan returns ConflictSummary — no conflicting_outputs or
                # root_cause_confidence. safe_get handles missing fields gracefully.
                for conflict in conflicts:
                    ctype   = safe_get(conflict, "conflict_type", "unknown")
                    agents  = unwrap_list(safe_get(conflict, "agents_involved", []))
                    label   = f"[{str(ctype).upper()}] {' vs '.join(agents) or '—'}"

                    with st.expander(label):
                        rc         = safe_get(conflict, "root_cause_type",    None) or "unknown"
                        confidence = bounded_float(safe_get(conflict, "root_cause_confidence", None))
                        strategy   = safe_get(conflict, "resolution_strategy", None)
                        status     = safe_get(conflict, "status",              "detected")

                        rc_col, conf_col = st.columns(2)
                        rc_col.markdown(f"**Root Cause:** {_humanize(rc)}")
                        with conf_col:
                            st.markdown(f"**Confidence:** {confidence:.0%}")
                            st.progress(confidence)

                        st.markdown(f"**Resolution Strategy:** {_humanize(strategy) if strategy else '—'}")
                        st.markdown(f"**Status:** {_humanize(status)}")

                        # conflicting_outputs only present on full Conflict object
                        conflicting_outputs = safe_get(conflict, "conflicting_outputs", {}) or {}
                        if conflicting_outputs:
                            st.subheader("Conflicting Outputs")
                            out_agents = list(conflicting_outputs.keys())
                            out_cols   = st.columns(max(len(out_agents), 1))
                            for idx, agent_key in enumerate(out_agents):
                                out_cols[idx].markdown(f"**{agent_key}**")
                                out_cols[idx].code(str(conflicting_outputs[agent_key]))

                if not conflicts:
                    st.info("No conflicts detected for this run.")


# ═══════════════════════════════════════════════════════════
# TAB 3 — Memory & Recommendations
# ═══════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Memory Intelligence")
    st.caption(
        "Accumulates conflict patterns across runs. Surfaces specific architectural "
        "fix recommendations when the same pattern recurs — the system gets smarter "
        "with every conflict it observes."
    )

    # Auto-load on tab render
    patterns_data, _pat_err = api_get(f"{base_url}/conflicts/patterns")
    recs_data,     _rec_err = api_get(f"{base_url}/conflicts/recommendations")

    patterns = unwrap_list(patterns_data) if patterns_data is not None else []
    recs     = unwrap_list(recs_data)     if recs_data     is not None else []

    # ── Summary metrics ───────────────────────────────────────────────────────
    most_common_type = "—"
    if patterns:
        top_pattern      = max(patterns, key=lambda p: safe_get(p, "occurrence_count", 0))
        most_common_type = _humanize(safe_get(top_pattern, "conflict_type", "—"))

    m1, m2, m3 = st.columns(3)
    m1.metric("Patterns Detected", len(patterns))
    m2.metric("Recommendations",   len(recs))
    m3.metric("Most Common Type",  most_common_type)

    st.divider()

    # ── Recommendations ───────────────────────────────────────────────────────
    st.subheader("Recommendations")
    if not recs:
        st.info("Run the demo twice to accumulate conflict patterns.")
    else:
        for rec in recs:
            st.warning("🔧 " + str(rec))

    # ── Conflict Patterns ─────────────────────────────────────────────────────
    st.subheader("Conflict Patterns")
    if not patterns:
        st.info("No patterns recorded yet.")
    else:
        # Build list of dicts — no pandas
        rows = [
            {
                "conflict_type":    safe_get(p, "conflict_type",    "—"),
                "agents_involved":  " ↔ ".join(
                    unwrap_list(safe_get(p, "agents_involved", []))
                ),
                "occurrence_count": safe_get(p, "occurrence_count", 0),
                "root_cause_type":  safe_get(p, "root_cause_type",  "—") or "—",
                "fix_specificity":  safe_get(p, "fix_specificity",  "—"),
            }
            for p in patterns
        ]
        st.dataframe(rows, use_container_width=True)

    st.caption("To reset patterns: rm demo_conflict_memory.json from project root")
