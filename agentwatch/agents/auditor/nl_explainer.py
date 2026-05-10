"""
AgentWatch — NL Explainer
===========================
Converts structured DecisionNodes into a grounded plain-English audit trail.

The grounding contract:
  Every AuditStep.reasoning MUST cite a specific event_id or context key.
  The system prompt enforces this. The validator in AuditStep checks it.
  If the LLM cannot ground a statement, it must say "not determinable from trace"
  rather than invent a reason.

This makes hallucination structurally impossible:
  the LLM can only cite things that exist in the data it was given.
"""

from __future__ import annotations

import json
from typing import List

import structlog

from agentwatch.core.llm.client import LLMClient

from agentwatch.core.models.schemas import (
    AuditStep,
    DecisionNode,
    RiskLevel,
    TraceGraph,
)

logger = structlog.get_logger(__name__)

EXPLAIN_TOOL = {
    "name": "generate_audit_steps",
    "description": "Generate grounded audit steps from decision nodes",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "node_id":           {"type": "string"},
                        "action":            {"type": "string"},
                        "reasoning":         {"type": "string"},
                        "outcome":           {"type": "string"},
                        "risk_level":        {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["node_id", "action", "reasoning", "outcome", "risk_level"],
                },
            }
        },
        "required": ["steps"],
    },
}


class NLExplainer:
    """
    Generates a grounded plain-English audit trail from DecisionNodes.

    One LLM call per run — all nodes batched together so the LLM
    can reason about the full sequence, not just isolated steps.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    async def explain(
        self,
        graph: TraceGraph,
        decision_nodes: List[DecisionNode],
    ) -> List[AuditStep]:
        """
        Produce AuditStep list from DecisionNodes.
        Returned in sequence order, one step per node.
        """
        if not decision_nodes:
            return []

        logger.info(
            "explaining_decisions",
            run_id=graph.run_id,
            nodes=len(decision_nodes),
        )

        raw_steps = await self._llm_explain(graph, decision_nodes)
        steps = self._build_steps(raw_steps, decision_nodes)

        logger.info(
            "explanation_complete",
            run_id=graph.run_id,
            steps=len(steps),
            high_risk=sum(1 for s in steps if s.risk_level == RiskLevel.HIGH),
        )
        return steps

    # ── Internal ──────────────────────────────────────

    async def _llm_explain(
        self,
        graph: TraceGraph,
        nodes: List[DecisionNode],
    ) -> List[dict]:
        """Single batched LLM call for all nodes."""

        nodes_payload = [
            {
                "node_id":          n.node_id,
                "agent_name":       n.agent_name,
                "sequence_no":      n.sequence_no,
                "decision_type":    n.decision_type,
                "chosen_option":    n.chosen_option,
                "options_available": n.options_available,
                "confidence":       n.confidence,
                "context_snapshot": n.context_snapshot,
                "is_critical":      n.is_critical,
                "source_event_ids": n.source_event_ids,
            }
            for n in nodes
        ]

        system = (
            "You are a compliance audit writer for AI agent systems. "
            "Your job is to explain each agent decision in plain English for a compliance report. "
            "\n\nGROUNDING RULES — these are mandatory and non-negotiable:\n"
            "1. Every 'reasoning' field MUST cite at least one specific source using one of:\n"
            "   - 'based on context key [key_name]'\n"
            "   - 'because event [event_id] showed [fact]'\n"
            "   - 'due to [specific value] in input_context'\n"
            "2. If you cannot ground a statement in the data provided, write: "
            "'not determinable from trace data'\n"
            "3. Never invent reasons. Never assume intent. Only report what the data shows.\n"
            "4. 'action' must be a concise verb phrase (max 10 words)\n"
            "5. risk_level rules:\n"
            "   - high: is_critical=true AND confidence < 0.6 (or confidence is null on a critical decision)\n"
            "   - medium: is_critical=true AND confidence >= 0.6\n"
            "   - low: is_critical=false\n"
            "6. Use plain English. No technical jargon. Write as if explaining to a non-technical auditor."
        )

        user = (
            f"Generate audit steps for this agent run (run_id: {graph.run_id}).\n"
            f"Total agents: {len(graph.agents_involved)}\n"
            f"Total events: {len(graph.events)}\n\n"
            f"Decision nodes to explain ({len(nodes_payload)} total):\n"
            f"{json.dumps(nodes_payload, indent=2, default=str)}"
        )

        result = self.client.chat_with_tool(
            system=system,
            user=user,
            tool=EXPLAIN_TOOL,
            tool_name="generate_audit_steps",
            max_tokens=4096,
        )

        return (result or {}).get("steps", [])

    def _build_steps(
        self,
        raw_steps: List[dict],
        nodes: List[DecisionNode],
    ) -> List[AuditStep]:
        """
        Map LLM output back to AuditStep objects.
        Attaches source_event_ids from the original DecisionNode.
        """
        node_map = {n.node_id: n for n in nodes}
        steps: List[AuditStep] = []

        for i, raw in enumerate(raw_steps, start=1):
            node_id = raw.get("node_id", "")
            source_node = node_map.get(node_id)
            source_event_ids = source_node.source_event_ids if source_node else []

            risk_raw = raw.get("risk_level", "low").lower()
            risk = RiskLevel(risk_raw) if risk_raw in RiskLevel._value2member_map_ else RiskLevel.LOW

            step = AuditStep(
                step_no=i,
                agent_name=source_node.agent_name if source_node else "unknown",
                action=raw.get("action", "performed action"),
                reasoning=raw.get("reasoning", "not determinable from trace data"),
                outcome=raw.get("outcome", "outcome not recorded"),
                risk_level=risk,
                source_event_ids=source_event_ids,
            )
            steps.append(step)

        return steps
