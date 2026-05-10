"""
AgentWatch — Root Cause Classifier
=====================================
Enriches a Conflict with its root cause using causal attribution.

Takes the last N events from each conflicting agent and asks:
"What in these agents' histories CAUSED this conflict?"

This is causal attribution applied to agent behavior —
the same methodology as the ML Observability platform, different domain.

Root cause taxonomy (6 types + unknown):
  instruction_ambiguity         — the agents' prompts are unclear or contradictory
  context_window_mismatch       — agents saw different slices of state
  tool_result_divergence        — same tool returned different results to each agent
  logical_contradiction         — the task itself requires contradictory actions
  stale_state                   — one agent acted on outdated information
  permission_boundary_violation — agent exceeded its authorized scope
"""

from __future__ import annotations

import asyncio
import json
from typing import List

import structlog

from agentwatch.core.llm.client import LLMClient

from agentwatch.core.models.schemas import (
    Conflict,
    ConflictStatus,
    RootCauseType,
    TraceGraph,
)

logger = structlog.get_logger(__name__)

# Number of prior events to pull per agent for context
CONTEXT_WINDOW = 5
MAX_CONCURRENT = 3

ROOT_CAUSE_TOOL = {
    "name": "classify_root_cause",
    "description": "Classify the root cause of an inter-agent conflict",
    "input_schema": {
        "type": "object",
        "properties": {
            "root_cause_type": {
                "type": "string",
                "enum": [t.value for t in RootCauseType],
            },
            "root_cause_detail": {
                "type": "string",
                "description": "Specific explanation, max 100 words",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in this classification, 0.0-1.0",
            },
        },
        "required": ["root_cause_type", "root_cause_detail", "confidence"],
    },
}


class RootCauseClassifier:
    """
    Enriches detected Conflicts with causal attribution.

    Conservative by design: if evidence is unclear, returns UNKNOWN
    with low confidence rather than guessing.
    """

    def __init__(self, client: LLMClient):
        self.client = client
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def classify(
        self, conflict: Conflict, graph: TraceGraph
    ) -> Conflict:
        """
        Classify a single conflict's root cause.
        Returns the same Conflict object enriched with root_cause fields.
        """
        # Pull recent event history for each involved agent
        agent_histories = {
            agent: self._get_agent_history(graph, agent, conflict)
            for agent in conflict.agents_involved
        }

        result = await self._llm_classify(conflict, agent_histories)

        conflict.root_cause_type = RootCauseType(
            result.get("root_cause_type", RootCauseType.UNKNOWN.value)
        )
        conflict.root_cause_detail = result.get("root_cause_detail", "")
        conflict.root_cause_confidence = result.get("confidence", 0.0)
        conflict.status = ConflictStatus.RESOLVING

        logger.info(
            "root_cause_classified",
            conflict_id=conflict.conflict_id,
            root_cause=conflict.root_cause_type,
            confidence=conflict.root_cause_confidence,
        )

        return conflict

    async def batch_classify(
        self, conflicts: List[Conflict], graph: TraceGraph
    ) -> List[Conflict]:
        """Classify multiple conflicts in parallel, max MAX_CONCURRENT."""
        async def _run(c: Conflict) -> Conflict:
            async with self._semaphore:
                return await self.classify(c, graph)

        results = await asyncio.gather(
            *[_run(c) for c in conflicts], return_exceptions=True
        )

        enriched: List[Conflict] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(
                    "classification_failed",
                    conflict_id=conflicts[i].conflict_id,
                    error=str(r),
                )
                enriched.append(conflicts[i])
            else:
                enriched.append(r)

        return enriched

    # ── Internal ──────────────────────────────────────

    async def _llm_classify(
        self,
        conflict: Conflict,
        agent_histories: dict,
    ) -> dict:
        system = (
            "You are a causal attribution engine for multi-agent AI systems. "
            "Given a conflict between agents and their recent event histories, "
            "identify the ROOT CAUSE — what in the system's design or state "
            "CAUSED this conflict to occur. "
            "\nRoot cause types:\n"
            "- instruction_ambiguity: agents' prompts/instructions are unclear or conflicting\n"
            "- context_window_mismatch: agents saw different portions of shared state\n"
            "- tool_result_divergence: a shared tool returned inconsistent results\n"
            "- logical_contradiction: the task fundamentally requires contradictory actions\n"
            "- stale_state: one agent acted on outdated information\n"
            "- permission_boundary_violation: agent exceeded its authorized scope\n"
            "- unknown: insufficient evidence to determine root cause\n"
            "\nBe conservative: only assign high confidence (>0.8) when evidence is clear. "
            "root_cause_detail must cite specific evidence from the agent histories — "
            "never make generic statements."
        )

        payload = {
            "conflict_type":       conflict.conflict_type.value,
            "agents_involved":     conflict.agents_involved,
            "conflicting_outputs": conflict.conflicting_outputs,
            "agent_histories":     agent_histories,
        }

        user = (
            f"Classify the root cause of this agent conflict:\n\n"
            f"{json.dumps(payload, indent=2, default=str)}"
        )

        async with self._semaphore:
            result = self.client.chat_with_tool(
                system=system,
                user=user,
                tool=ROOT_CAUSE_TOOL,
                tool_name="classify_root_cause",
                max_tokens=512,
            )

        return result or {
            "root_cause_type": RootCauseType.UNKNOWN.value,
            "root_cause_detail": "Classification failed — insufficient evidence",
            "confidence": 0.0,
        }

    @staticmethod
    def _get_agent_history(
        graph: TraceGraph,
        agent_name: str,
        conflict: Conflict,
    ) -> list:
        """
        Pull the last CONTEXT_WINDOW events for an agent before the conflict.
        Returns serializable dicts — safe for JSON.
        """
        trigger_seq = 0
        if conflict.triggering_event_ids:
            trigger_events = [
                e for e in graph.events
                if e.event_id in conflict.triggering_event_ids
            ]
            if trigger_events:
                trigger_seq = min(e.sequence_no for e in trigger_events)

        agent_events = [
            e for e in graph.events
            if e.agent_name == agent_name
            and (trigger_seq == 0 or e.sequence_no <= trigger_seq)
        ]

        recent = sorted(agent_events, key=lambda e: e.sequence_no)[-CONTEXT_WINDOW:]

        return [
            {
                "sequence_no": e.sequence_no,
                "event_type":  e.event_type.value,
                "input_context": e.input_context,
                "output":      str(e.output) if e.output is not None else None,
                "tool_name":   e.tool_name,
            }
            for e in recent
        ]
