"""
AgentWatch — Compliance Reporter
==================================
Assembles all Pillar A components into a final AuditReport.

Inputs:  TraceGraph + AuditSteps + DecisionNodes + CounterfactualResults
Output:  AuditReport — fully structured, grounded, compliance-ready

Design decisions:
- overall_risk is enforced by AuditReport's model_validator — can't be wrong
- summary is LLM-generated but grounded: LLM only sees structured data, not raw text
- compliance_data is always populated with quantitative fields — never empty
- to_markdown() produces clean output for PDF conversion or GitHub README
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List

import structlog

from agentwatch.core.llm.client import LLMClient

from agentwatch.core.models.schemas import (
    AuditReport,
    AuditStep,
    CounterfactualResult,
    DecisionNode,
    RiskLevel,
    TraceGraph,
)

logger = structlog.get_logger(__name__)

SUMMARY_TOOL = {
    "name": "generate_audit_summary",
    "description": "Generate a grounded plain-English summary of an agent audit",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary":     {"type": "string", "description": "3-sentence plain English summary"},
            "key_finding": {"type": "string", "description": "Single most important finding — max 20 words"},
            "risk_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific risk factors observed — max 5 items",
            },
        },
        "required": ["summary", "key_finding", "risk_factors"],
    },
}


class ComplianceReporter:
    """
    Final assembly step for Pillar A.
    Combines structured audit data into a compliance-ready AuditReport.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    async def generate_report(
        self,
        graph: TraceGraph,
        steps: List[AuditStep],
        decision_nodes: List[DecisionNode],
        counterfactuals: List[CounterfactualResult],
    ) -> AuditReport:
        """
        Generate a complete AuditReport from all Pillar A outputs.
        """
        logger.info(
            "generating_compliance_report",
            run_id=graph.run_id,
            steps=len(steps),
            decisions=len(decision_nodes),
            counterfactuals=len(counterfactuals),
        )

        critical_decisions = [n for n in decision_nodes if n.is_critical]
        high_risk_steps = [s for s in steps if s.risk_level == RiskLevel.HIGH]

        # Determine overall risk before LLM call — used in summary prompt
        if high_risk_steps:
            overall_risk = RiskLevel.HIGH
        elif any(s.risk_level == RiskLevel.MEDIUM for s in steps):
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW

        # LLM-generated summary — grounded on structured data only
        llm_out = await self._llm_summarize(
            graph, steps, critical_decisions, counterfactuals, overall_risk
        )

        compliance_data = self._build_compliance_data(
            graph, steps, decision_nodes, counterfactuals
        )

        report = AuditReport(
            run_id=graph.run_id,
            steps=steps,
            critical_decisions=critical_decisions,
            counterfactuals=counterfactuals,
            summary=llm_out.get("summary", "Audit completed. See steps for details."),
            key_finding=llm_out.get("key_finding", "No critical findings."),
            overall_risk=overall_risk,
            risk_factors=llm_out.get("risk_factors", []),
            compliance_data=compliance_data,
        )

        logger.info(
            "report_generated",
            run_id=graph.run_id,
            report_id=report.report_id,
            overall_risk=report.overall_risk,
            critical_decisions=len(critical_decisions),
        )

        return report

    async def to_json(self, report: AuditReport) -> str:
        return report.model_dump_json(indent=2)

    async def to_markdown(self, report: AuditReport) -> str:
        """
        Structured markdown — clean enough to paste into a PR comment,
        convert to PDF, or drop into a GitHub README.
        """
        cf_section = ""
        if report.counterfactuals:
            cf_lines = []
            for cf in report.counterfactuals:
                changed = "✅ Changed" if cf.outcome_changed else "➡️ Unchanged"
                cf_lines.append(
                    f"- **{cf.divergence_point}** → {changed} "
                    f"(sensitivity: `{cf.sensitivity_score:.2f}`)\n"
                    f"  - Original: `{cf.original_outcome}` → Counterfactual: `{cf.counterfactual_outcome}`"
                )
            cf_section = "## Counterfactual Analysis\n\n" + "\n".join(cf_lines)

        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        risk_icon = risk_emoji.get(report.overall_risk.value, "⚪")

        steps_md = "\n".join([
            f"### Step {s.step_no} — {s.agent_name}\n"
            f"**Action:** {s.action}\n\n"
            f"**Reasoning:** {s.reasoning}\n\n"
            f"**Outcome:** {s.outcome}\n\n"
            f"**Risk:** {risk_emoji.get(s.risk_level.value, '⚪')} {s.risk_level.value.upper()}\n"
            for s in report.steps
        ])

        risk_factors_md = (
            "\n".join(f"- {rf}" for rf in report.risk_factors)
            if report.risk_factors else "- None identified"
        )

        cd = report.compliance_data
        return f"""# AgentWatch Audit Report

**Report ID:** `{report.report_id}`
**Run ID:** `{report.run_id}`
**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Overall Risk:** {risk_icon} {report.overall_risk.value.upper()}

---

## Executive Summary

{report.summary}

**Key Finding:** {report.key_finding}

---

## Risk Factors

{risk_factors_md}

---

## Audit Trail

{steps_md}

---

{cf_section}

---

## Compliance Data

| Metric | Value |
|--------|-------|
| Total Decisions | {cd.get('total_decisions', 0)} |
| Critical Decisions | {cd.get('critical_decisions_count', 0)} |
| High Risk Steps | {cd.get('high_risk_steps_count', 0)} |
| Counterfactuals Run | {cd.get('counterfactuals_run', 0)} |
| Outcome Changes Detected | {cd.get('outcome_changed_count', 0)} |
| Total Cost (USD) | ${cd.get('total_cost_usd', 0):.4f} |
| Total Latency (ms) | {cd.get('total_latency_ms', 0):.0f} |
| Total Tokens | {cd.get('total_tokens', 0)} |

---
*Generated by AgentWatch v1.0.0*
"""

    # ── Internal ──────────────────────────────────────

    async def _llm_summarize(
        self,
        graph: TraceGraph,
        steps: List[AuditStep],
        critical_decisions: List[DecisionNode],
        counterfactuals: List[CounterfactualResult],
        overall_risk: RiskLevel,
    ) -> dict:
        """
        LLM call for summary generation.
        Only sees structured data — grounding is enforced by what we pass in.
        """
        structured_input = {
            "run_id": graph.run_id,
            "agents": graph.agents_involved,
            "overall_risk": overall_risk.value,
            "total_steps": len(steps),
            "critical_decisions_count": len(critical_decisions),
            "high_risk_steps": [
                {"agent": s.agent_name, "action": s.action, "outcome": s.outcome}
                for s in steps if s.risk_level == RiskLevel.HIGH
            ],
            "outcome_changes_detected": sum(1 for c in counterfactuals if c.outcome_changed),
            "sensitive_divergence_points": [
                c.divergence_point for c in counterfactuals if c.outcome_changed
            ],
        }

        system = (
            "You are a compliance audit summarizer for AI agent systems. "
            "Write in plain English for a non-technical compliance officer. "
            "Ground every statement in the structured data provided. "
            "Do not invent findings not present in the data. "
            "summary must be exactly 3 sentences. "
            "key_finding must reference the specific agents involved and the nature of "
            "the conflict in under 20 words — never use generic phrases like "
            "'high risk steps detected' or 'high risk actions detected'. "
            "risk_factors must be specific observations from the data — not generic warnings."
        )

        result = self.client.chat_with_tool(
            system=system,
            user=f"Summarize this agent audit:\n\n{json.dumps(structured_input, indent=2)}",
            tool=SUMMARY_TOOL,
            tool_name="generate_audit_summary",
            max_tokens=1024,
        )

        return result or {
            "summary": f"Agent run completed with {overall_risk.value} overall risk.",
            "key_finding": "See audit steps for details.",
            "risk_factors": [],
        }

    @staticmethod
    def _build_compliance_data(
        graph: TraceGraph,
        steps: List[AuditStep],
        decision_nodes: List[DecisionNode],
        counterfactuals: List[CounterfactualResult],
    ) -> dict:
        """
        Quantitative compliance metadata — always populated, always numeric.
        This is what goes into a compliance database or audit trail system.
        """
        return {
            "run_id":                   graph.run_id,
            "session_id":               graph.session_id,
            "generated_at":             datetime.utcnow().isoformat(),
            "total_decisions":          len(decision_nodes),
            "critical_decisions_count": sum(1 for n in decision_nodes if n.is_critical),
            "high_risk_steps_count":    sum(1 for s in steps if s.risk_level == RiskLevel.HIGH),
            "medium_risk_steps_count":  sum(1 for s in steps if s.risk_level == RiskLevel.MEDIUM),
            "low_risk_steps_count":     sum(1 for s in steps if s.risk_level == RiskLevel.LOW),
            "counterfactuals_run":      len(counterfactuals),
            "outcome_changed_count":    sum(1 for c in counterfactuals if c.outcome_changed),
            "avg_sensitivity_score":    round(
                sum(c.sensitivity_score or 0 for c in counterfactuals) / len(counterfactuals), 3
            ) if counterfactuals else 0.0,
            "total_cost_usd":           graph.total_cost_usd or 0.0,
            "total_latency_ms":         graph.total_latency_ms or 0.0,
            "total_tokens":             graph.total_tokens or 0,
            "agents_involved":          graph.agents_involved,
            "total_events":             len(graph.events),
        }
