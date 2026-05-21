"""
AgentWatch — Demo Runner
==========================
Orchestrates a full end-to-end demo run:

1.  Creates a deliberately conflicting 3-agent pipeline
2.  Instruments it with ExecutionTracer
3.  Runs all three agents with handoffs
4.  Detects all three conflict types
5.  Classifies root causes
6.  Resolves conflicts
7.  Generates a Decision Audit report
8.  Prints a structured summary

Usage:
    python -m agentwatch.demo.run_demo
    python -m agentwatch.demo.run_demo --pretty      # full JSON output
    python -m agentwatch.demo.run_demo --export      # saves markdown report

This is the live demo you run in interviews.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()

# Silence noisy logs during demo — only show our structured output
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        int(os.getenv("LOG_LEVEL_INT", "30"))  # WARNING by default
    )
)

logger = structlog.get_logger(__name__)


async def run_demo(pretty: bool = False, export: bool = False) -> dict:
    """
    Full end-to-end AgentWatch demo.
    Returns a structured summary dict.
    """
    from agentwatch.core.llm.client import LLMClient
    from agentwatch.core.tracer.execution_tracer import ExecutionTracer
    from agentwatch.core.store.trace_store import TraceStore
    from agentwatch.agents.auditor.decision_decomposer import DecisionDecomposer
    from agentwatch.agents.auditor.nl_explainer import NLExplainer
    from agentwatch.agents.auditor.counterfactual_explorer import CounterfactualExplorer
    from agentwatch.agents.auditor.compliance_reporter import ComplianceReporter
    from agentwatch.agents.conflict.conflict_sensor import ConflictSensor
    from agentwatch.agents.conflict.root_cause_classifier import RootCauseClassifier
    from agentwatch.agents.conflict.resolution_strategist import ResolutionStrategist
    from agentwatch.agents.conflict.conflict_memory import ConflictMemoryStore
    from agentwatch.demo.demo_agents import CustomerServiceAgent, ComplianceAgent, BillingAgent, OrderContext
    from agentwatch.core.models.schemas import AgentEventType

    t_start = time.monotonic()

    # ── Setup ─────────────────────────────────────────
    _print_header("AgentWatch — Live Demo")

    llm    = LLMClient.from_env()
    store  = TraceStore()
    await store.init_db()
    memory = ConflictMemoryStore(store_path="demo_conflict_memory.json")

    tracer = ExecutionTracer(
        session_id="demo-session-001",
        store=store,
        loop_detection_threshold=3,
    )

    # ── Create order that triggers all conflict types ──
    order = OrderContext(
        order_id="ORD-8821",
        customer_id="CUST-4492",
        amount=150.0,          # > compliance threshold (100) → SEMANTIC conflict
        tier="premium",        # → CS agent approves
    )

    _print_section("Order Under Review")
    _print_kv("Order ID",    order.order_id)
    _print_kv("Customer",    order.customer_id)
    _print_kv("Amount",      f"${order.amount:.2f}")
    _print_kv("Tier",        order.tier)
    _print_kv("Data freshness", f"STALE — last updated {order.updated_at}")

    # ── Run agents ────────────────────────────────────
    _print_section("Running Agent Pipeline")

    run_id = await tracer.start_run()
    _print_kv("Run ID", run_id)

    cs_agent         = CustomerServiceAgent(tracer)
    compliance_agent = ComplianceAgent(tracer)
    billing_agent    = BillingAgent(tracer)

    # Agent 1: Customer Service
    _print_step("1", "CustomerServiceAgent", "evaluating refund request...")
    cs_result = await cs_agent.handle_refund_request(order)
    _print_kv("  Decision", cs_result["decision"], color="green" if cs_result["approved"] else "red")

    # Handoff CS → Compliance
    await tracer.record_handoff(
        from_agent="customer_service_agent",
        to_agent="compliance_agent",
        payload=cs_result,
        context={"reason": "compliance_review_required"},
    )

    # Agent 2: Compliance (runs independently — doesn't know CS approved)
    _print_step("2", "ComplianceAgent", "reviewing for risk...")
    comp_result = await compliance_agent.review_action(order, cs_result)
    _print_kv("  Decision", comp_result["decision"], color="red" if comp_result["blocked"] else "green")
    if comp_result["blocked"]:
        _print_kv("  Reason",   comp_result["reason"])

    # Handoff CS → Billing (without waiting for Compliance → PROCEDURAL)
    await tracer.record_handoff(
        from_agent="customer_service_agent",
        to_agent="billing_agent",
        payload={"action": cs_result["decision"], "order_id": order.order_id},
        context={"note": "parallel handoff — compliance review ongoing"},
    )

    # Agent 3: Billing (uses stale context → TEMPORAL)
    _print_step("3", "BillingAgent", "executing billing action...")
    bill_result = await billing_agent.process(order, cs_result["decision"])
    _print_kv("  Executed", str(bill_result["executed"]))
    _print_kv("  Stale data used", bill_result["stale_context_used"], color="yellow")

    graph = await tracer.end_run(success=True)
    await store.save_trace(graph)

    _print_kv("Events captured", str(len(graph.events)))
    _print_kv("Agents involved", ", ".join(graph.agents_involved))
    _print_kv("Total cost", f"${graph.total_cost_usd or 0:.4f}")
    if graph.total_latency_ms and graph.total_latency_ms > 10:
        _print_kv("Agent execution latency (excl. LLM)", f"{graph.total_latency_ms:.0f}ms")

    # ── Conflict Detection ────────────────────────────
    _print_section("Conflict Detection & Root Cause")

    sensor     = ConflictSensor(client=llm)
    classifier = RootCauseClassifier(client=llm)
    strategist = ResolutionStrategist(client=llm)

    _print_step("→", "ConflictSensor", "scanning for inter-agent conflicts...")
    conflicts = await sensor.scan(graph)

    if not conflicts:
        _print_kv("Result", "No conflicts detected")
    else:
        _print_kv("Conflicts found", str(len(conflicts)))
        for i, c in enumerate(conflicts, 1):
            _print_kv(f"  [{i}] Type", c.conflict_type.value.upper(), color="red")
            _print_kv(f"      Agents", " vs ".join(c.agents_involved))

    # Classify root causes
    _print_step("→", "RootCauseClassifier", "attributing causes...")
    conflicts = await classifier.batch_classify(conflicts, graph)
    for c in conflicts:
        _print_kv(f"  {c.conflict_type.value}", f"root cause → {c.root_cause_type.value if c.root_cause_type else 'unknown'}")
        _print_kv(f"  confidence", f"{c.root_cause_confidence:.0%}" if c.root_cause_confidence else "N/A")

    # Resolve
    _print_step("→", "ResolutionStrategist", "resolving conflicts...")
    resolved_conflicts = []
    for c in conflicts:
        resolved = await strategist.resolve(c, graph)
        resolved_conflicts.append(resolved)
        _print_kv(
            f"  {c.conflict_type.value}",
            f"strategy → {resolved.resolution_strategy.value if resolved.resolution_strategy else 'none'}"
        )

    # Update memory
    for c in resolved_conflicts:
        await memory.record(c)
        await memory.upsert_pattern(c)

    patterns = await memory.get_all_patterns()
    recs     = await memory.get_recommendations()
    _print_kv("Patterns in memory", str(len(patterns)))
    _print_kv("Recommendations", str(len(recs)))

    # ── Decision Audit ────────────────────────────────
    _print_section("Decision Audit & Compliance Report")

    decomposer = DecisionDecomposer(client=llm)
    explainer  = NLExplainer(client=llm)
    cf_explorer = CounterfactualExplorer(client=llm)
    reporter   = ComplianceReporter(client=llm)

    _print_step("→", "DecisionDecomposer", "extracting decision nodes...")
    nodes = await decomposer.decompose(graph)
    _print_kv("Decision nodes", str(len(nodes)))
    _print_kv("Critical", str(sum(1 for n in nodes if n.is_critical)))

    _print_step("→", "NLExplainer", "generating audit trail...")
    steps = await explainer.explain(graph, nodes)
    _print_kv("Audit steps", str(len(steps)))

    _print_step("→", "CounterfactualExplorer", "finding sensitive decisions...")
    sensitive = await cf_explorer.find_sensitive_decisions(graph, nodes)
    _print_kv("Sensitive decisions", str(len(sensitive)))

    _print_step("→", "ComplianceReporter", "assembling audit report...")
    mods = [cf_explorer._auto_modify(n.context_snapshot) for n in sensitive]
    cfs  = await cf_explorer.batch_explore(graph, sensitive, mods) if sensitive else []
    report = await reporter.generate_report(graph, steps, nodes, cfs)

    _print_kv("Overall risk",    report.overall_risk.value.upper(),
              color="red" if report.overall_risk.value == "high" else "yellow")
    _print_kv("Key finding",     _clean_finding(report.key_finding))
    _print_kv("Report ID",       report.report_id)

    # ── Final Summary ─────────────────────────────────
    total_ms = (time.monotonic() - t_start) * 1000
    _print_section("Summary")

    summary = {
        "run_id":                run_id,
        "total_latency_ms":      round(total_ms, 1),
        "total_cost_usd":        round(graph.total_cost_usd or 0, 6),
        "total_tokens":          graph.total_tokens or 0,
        "events_captured":       len(graph.events),
        "agents_involved":       graph.agents_involved,
        "conflicts_detected":    len(conflicts),
        "conflict_types":        [c.conflict_type.value for c in conflicts],
        "resolutions_applied":   [
            c.resolution_strategy.value
            for c in resolved_conflicts
            if c.resolution_strategy
        ],
        "decision_nodes":        len(nodes),
        "critical_decisions":    sum(1 for n in nodes if n.is_critical),
        "sensitive_decisions":   len(sensitive),
        "audit_overall_risk":    report.overall_risk.value,
        "audit_key_finding":     report.key_finding,
        "patterns_in_memory":    len(patterns),
        "recommendations":       recs,
    }

    for k, v in summary.items():
        if k in ("recommendations", "patterns_in_memory"):
            continue  # recommendations printed below; patterns are internal
        elif k == "total_latency_ms":
            _print_kv(
                "End-to-end pipeline latency (incl. LLM calls)",
                f"{v / 1000:.1f}s",
            )
        elif k == "total_cost_usd":
            _print_kv("Total Cost (USD)", str(v))
        elif isinstance(v, list):
            _print_kv(k.replace("_", " ").title(), ", ".join(str(x) for x in v) or "none")
        else:
            _print_kv(k.replace("_", " ").title(), str(v))

    _print_recommendations(patterns)

    # ── Executive Summary ─────────────────────────────
    deduped_recs = _dedup_recommendations(patterns)
    top_rec = _clean_rec(deduped_recs[0][0]) if deduped_recs else (recs[0] if recs else "No recommendations yet")
    n_conflicts = len(resolved_conflicts)
    outcome = (
        f"BLOCKED — {n_conflicts} inter-agent conflict{'s' if n_conflicts != 1 else ''} detected"
        if n_conflicts else "COMPLETED — no conflicts detected"
    )

    _print_section("Executive Summary")
    _print_kv("Outcome",            outcome, color="red" if n_conflicts else "green")
    _print_kv("Audit Risk",         report.overall_risk.value.upper(),
              color="red" if report.overall_risk.value == "high" else "yellow")
    _print_kv("Key Finding",        _clean_finding(report.key_finding))
    _print_kv("Top Recommendation", top_rec)
    _print_kv("Run ID",             run_id)

    if pretty:
        print("\n" + "─" * 60)
        print("FULL JSON OUTPUT")
        print("─" * 60)
        print(json.dumps(
            {
                "summary": summary,
                "audit_report": json.loads(report.model_dump_json()),
                "conflicts": [json.loads(c.model_dump_json()) for c in resolved_conflicts],
            },
            indent=2,
            default=str,
        ))

    if export:
        md = await reporter.to_markdown(report)
        path = Path("demo_audit_report.md")
        path.write_text(md)
        print(f"\n  📄 Report exported → {path.absolute()}")

    print("\n" + "═" * 60 + "\n")
    return summary


# ── Pretty print helpers ──────────────────────────────

COLORS = {
    "red":    "\033[91m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "blue":   "\033[94m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def _clean_finding(text: str) -> str:
    """Ensure Key Finding starts with a capital letter and ends with a period."""
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


def _c(text: str, color: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def _print_header(title: str):
    print(f"\n{_c('═' * 60, 'bold')}")
    print(f"{_c(f'  {title}', 'bold')}")
    print(f"{_c('═' * 60, 'bold')}\n")

def _print_section(title: str):
    print(f"\n  {_c('▸ ' + title, 'blue')}")
    print(f"  {'─' * 50}")

def _print_step(num: str, agent: str, msg: str):
    print(f"  {_c(num, 'bold')}  {_c(agent, 'bold')} — {msg}")

def _print_kv(key: str, val: str, color: str = ""):
    val_str = _c(val, color) if color else val
    print(f"     {key:<30} {val_str}")


def _clean_rec(text: str) -> str:
    """Replace template placeholders with human-readable equivalents at render time."""
    return text.replace("[agents]", "affected agents'").replace("[threshold]", "30s")


def _dedup_recommendations(patterns) -> list:
    """
    Group recommendations that share the same template, differing only in agent names.
    Returns list of (canonical_text, [(agent_a, agent_b), ...]) in occurrence order.
    """
    groups: dict = {}
    for p in patterns:
        if p.occurrence_count < 2:  # mirror RECOMMENDATION_THRESHOLD
            continue
        agents_str = " and ".join(p.agents_involved)
        canonical = p.fix_recommendation.replace(agents_str, "[agents]")
        if canonical not in groups:
            groups[canonical] = []
        pair = tuple(sorted(p.agents_involved))
        if pair not in groups[canonical]:
            groups[canonical].append(pair)
    return list(groups.items())


def _print_recommendations(patterns) -> None:
    """Print deduplicated recommendations grouped by template with affected agent pairs."""
    deduped = _dedup_recommendations(patterns)
    if not deduped:
        _print_kv("Recommendations", "None yet (need more observations)")
        return

    print(f"\n     Recommendations ({len(deduped)} unique):")
    for i, (canonical, pairs) in enumerate(deduped, 1):
        print(f"\n     [{i}] {_clean_rec(canonical)}")
        pair_strs = ", ".join(
            f"({a.replace('_agent', '')} ↔ {b.replace('_agent', '')})"
            for a, b in pairs
        )
        print(f"         Affected pairs: {pair_strs}")


# ── Entry point ───────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentWatch Demo")
    parser.add_argument("--pretty", action="store_true", help="Print full JSON output")
    parser.add_argument("--export", action="store_true", help="Export audit report as markdown")
    args = parser.parse_args()

    asyncio.run(run_demo(pretty=args.pretty, export=args.export))
