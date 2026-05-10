"""
AgentWatch — Audit Routes
===========================
POST /audit          — run full Decision Audit on a stored trace
GET  /audit/{id}     — retrieve cached audit report
POST /audit/{id}/export — export report as markdown
"""

from __future__ import annotations

import time
from typing import Dict

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from agentwatch.api.dependencies import (
    get_audit_cache,
    get_cf_explorer,
    get_decomposer,
    get_explainer,
    get_reporter,
    get_store,
)
from agentwatch.agents.auditor.compliance_reporter import ComplianceReporter
from agentwatch.agents.auditor.counterfactual_explorer import CounterfactualExplorer
from agentwatch.agents.auditor.decision_decomposer import DecisionDecomposer
from agentwatch.agents.auditor.nl_explainer import NLExplainer
from agentwatch.core.models.schemas import AuditReport, AuditRequest
from agentwatch.core.store.trace_store import TraceStore

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("", response_model=AuditReport)
async def run_audit(
    body: AuditRequest,
    store: TraceStore = Depends(get_store),
    decomposer: DecisionDecomposer = Depends(get_decomposer),
    explainer: NLExplainer = Depends(get_explainer),
    cf_explorer: CounterfactualExplorer = Depends(get_cf_explorer),
    reporter: ComplianceReporter = Depends(get_reporter),
    cache: Dict = Depends(get_audit_cache),
) -> AuditReport:
    """
    Run a full Decision Audit pipeline on a stored trace.

    Pipeline stages (timed individually):
    1. Fetch trace from store
    2. DecisionDecomposer — extract decision nodes
    3. NLExplainer — generate audit steps
    4. CounterfactualExplorer — find sensitive decisions (optional)
    5. ComplianceReporter — assemble final report
    """
    t_start = time.monotonic()

    # 1. Fetch trace
    graph = await store.get_trace(body.run_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Trace {body.run_id} not found")

    t_fetch = time.monotonic()

    # 2. Decompose decisions
    decision_nodes = await decomposer.decompose(graph)
    t_decompose = time.monotonic()

    # 3. Generate audit steps
    steps = await explainer.explain(graph, decision_nodes)
    t_explain = time.monotonic()

    # 4. Counterfactual analysis (optional)
    counterfactuals = []
    if body.include_counterfactuals and decision_nodes:
        sensitive = await cf_explorer.find_sensitive_decisions(
            graph,
            decision_nodes[:body.max_counterfactuals],
        )
        if sensitive:
            mods = [cf_explorer._auto_modify(n.context_snapshot) for n in sensitive]
            counterfactuals = await cf_explorer.batch_explore(graph, sensitive, mods)

    t_cf = time.monotonic()

    # 5. Assemble report
    report = await reporter.generate_report(graph, steps, decision_nodes, counterfactuals)
    t_report = time.monotonic()

    # Cache report
    cache[report.report_id] = report

    # Log pipeline timing breakdown
    logger.info(
        "audit_pipeline_complete",
        run_id=body.run_id,
        report_id=report.report_id,
        overall_risk=report.overall_risk.value,
        latency_fetch_ms=round((t_fetch - t_start) * 1000, 1),
        latency_decompose_ms=round((t_decompose - t_fetch) * 1000, 1),
        latency_explain_ms=round((t_explain - t_decompose) * 1000, 1),
        latency_cf_ms=round((t_cf - t_explain) * 1000, 1),
        latency_report_ms=round((t_report - t_cf) * 1000, 1),
        latency_total_ms=round((t_report - t_start) * 1000, 1),
    )

    return report


@router.get("/{report_id}", response_model=AuditReport)
async def get_report(
    report_id: str,
    cache: Dict = Depends(get_audit_cache),
) -> AuditReport:
    """Retrieve a cached audit report by report_id."""
    report = cache.get(report_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report {report_id} not found. Reports are cached per session — re-run audit."
        )
    return report


@router.post("/{run_id}/export")
async def export_report(
    run_id: str,
    store: TraceStore = Depends(get_store),
    decomposer: DecisionDecomposer = Depends(get_decomposer),
    explainer: NLExplainer = Depends(get_explainer),
    reporter: ComplianceReporter = Depends(get_reporter),
) -> Dict[str, str]:
    """
    Export a markdown audit report for a run.
    Re-runs decomposition + explanation (no counterfactuals for speed).
    """
    graph = await store.get_trace(run_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Trace {run_id} not found")

    nodes = await decomposer.decompose(graph)
    steps = await explainer.explain(graph, nodes)
    report = await reporter.generate_report(graph, steps, nodes, [])
    markdown = await reporter.to_markdown(report)

    return {"run_id": run_id, "format": "markdown", "content": markdown}
