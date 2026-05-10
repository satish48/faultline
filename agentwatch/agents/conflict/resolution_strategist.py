"""
AgentWatch — Resolution Strategist
=====================================
Resolves detected conflicts using one of four strategies.

Strategy selection is deterministic — based on conflict_type + root_cause_confidence.
The LLM is only used for SYNTHESIZE (merging two outputs into one).
ARBITRATE, RESET, and ESCALATE are fully rule-based.

Strategy rules:
  SEMANTIC   → try SYNTHESIZE; if confidence < 0.7 → ARBITRATE
  PROCEDURAL → always ARBITRATE (higher sequence_no agent wins — fresher state)
  TEMPORAL   → always RESET (rerun with corrected context)
  Any type   → ESCALATE if root_cause_confidence < 0.5 (too uncertain to auto-resolve)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

import structlog

from agentwatch.core.llm.client import LLMClient

from agentwatch.core.models.schemas import (
    Conflict,
    ConflictStatus,
    ConflictType,
    ResolutionStrategy,
    RootCauseType,
    TraceGraph,
)

logger = structlog.get_logger(__name__)

SYNTHESIZE_TOOL = {
    "name": "synthesize_outputs",
    "description": "Merge two conflicting agent outputs into one coherent response",
    "input_schema": {
        "type": "object",
        "properties": {
            "synthesized_output": {"type": "string"},
            "elements_from_a":   {"type": "string"},
            "elements_from_b":   {"type": "string"},
            "synthesizable":     {"type": "boolean"},
            "confidence":        {"type": "number"},
        },
        "required": ["synthesized_output", "synthesizable", "confidence"],
    },
}

# If synthesis confidence below this, fall back to ARBITRATE
SYNTHESIS_CONFIDENCE_THRESHOLD = 0.7
# If root cause confidence below this, always ESCALATE
ESCALATION_CONFIDENCE_THRESHOLD = 0.5


class ResolutionStrategist:
    """
    Resolves conflicts. Returns the same Conflict enriched with resolution fields.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    async def resolve(self, conflict: Conflict, graph: TraceGraph) -> Conflict:
        """
        Select and apply the appropriate resolution strategy.
        Returns conflict with resolution fields populated.
        """
        # Override: always escalate if root cause is too uncertain
        if (
            conflict.root_cause_confidence is not None
            and conflict.root_cause_confidence < ESCALATION_CONFIDENCE_THRESHOLD
        ):
            return self._apply_escalate(
                conflict,
                reason="Root cause confidence too low for autonomous resolution",
            )

        # Strategy selection by conflict type
        if conflict.conflict_type == ConflictType.SEMANTIC:
            return await self._resolve_semantic(conflict, graph)
        elif conflict.conflict_type == ConflictType.PROCEDURAL:
            return self._resolve_procedural(conflict, graph)
        elif conflict.conflict_type == ConflictType.TEMPORAL:
            return self._resolve_temporal(conflict, graph)
        else:
            return self._apply_escalate(conflict, reason="Unknown conflict type")

    # ── Strategy implementations ──────────────────────

    async def _resolve_semantic(
        self, conflict: Conflict, graph: TraceGraph
    ) -> Conflict:
        """Try SYNTHESIZE first. Fall back to ARBITRATE if synthesis fails."""
        synth = await self._llm_synthesize(conflict)

        if synth.get("synthesizable") and synth.get("confidence", 0) >= SYNTHESIS_CONFIDENCE_THRESHOLD:
            conflict.resolution_strategy  = ResolutionStrategy.SYNTHESIZE
            conflict.resolution_output    = synth.get("synthesized_output")
            conflict.resolution_reasoning = (
                f"Synthesized from both agents. "
                f"From {conflict.agents_involved[0]}: {synth.get('elements_from_a', 'N/A')}. "
                f"From {conflict.agents_involved[1]}: {synth.get('elements_from_b', 'N/A')}."
            )
        else:
            # Fall back to ARBITRATE — pick agent with more recent output
            return self._apply_arbitrate(
                conflict, graph,
                reason="Synthesis failed or confidence too low — arbitrating instead",
            )

        conflict.status      = ConflictStatus.RESOLVED
        conflict.resolved_at = datetime.utcnow()

        logger.info(
            "conflict_resolved",
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.SYNTHESIZE.value,
        )
        return conflict

    def _resolve_procedural(
        self, conflict: Conflict, graph: TraceGraph
    ) -> Conflict:
        """
        ARBITRATE — pick the agent with the higher sequence number.
        Higher seq = more context seen = fresher state = more authoritative.
        """
        return self._apply_arbitrate(
            conflict, graph,
            reason="Procedural conflict: agent with fresher state (higher sequence) wins",
        )

    def _resolve_temporal(
        self, conflict: Conflict, graph: TraceGraph
    ) -> Conflict:
        """
        RESET — build a corrected context and instruct a rerun.
        Merges both agents' contexts, fresher values win on collision.
        """
        corrected_context = self._build_corrected_context(conflict, graph)

        conflict.resolution_strategy  = ResolutionStrategy.RESET
        conflict.resolution_output    = {
            "action":            "rerun_with_corrected_context",
            "corrected_context": corrected_context,
            "instruction":       (
                "Rerun the conflicting step using corrected_context. "
                "Stale timestamp values have been replaced with fresher data."
            ),
        }
        conflict.resolution_reasoning = (
            "Temporal conflict: one agent used stale context data. "
            "Corrected context merges both agents' state with fresher values winning."
        )
        conflict.status      = ConflictStatus.RESOLVED
        conflict.resolved_at = datetime.utcnow()

        logger.info(
            "conflict_resolved",
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.RESET.value,
        )
        return conflict

    # ── Strategy helpers ──────────────────────────────

    def _apply_escalate(self, conflict: Conflict, reason: str) -> Conflict:
        conflict.resolution_strategy  = ResolutionStrategy.ESCALATE
        conflict.resolution_output    = {
            "human_review_required": True,
            "reason":                reason,
            "conflict_id":           conflict.conflict_id,
            "agents_involved":       conflict.agents_involved,
        }
        conflict.resolution_reasoning = reason
        conflict.status               = ConflictStatus.ESCALATED
        conflict.resolved_at          = datetime.utcnow()

        logger.warning(
            "conflict_escalated",
            conflict_id=conflict.conflict_id,
            reason=reason,
        )
        return conflict

    def _apply_arbitrate(
        self,
        conflict: Conflict,
        graph: TraceGraph,
        reason: str,
    ) -> Conflict:
        """
        Pick the agent with the highest sequence number among the conflicting agents.
        That agent has the most recent context — most authoritative.
        """
        winner = self._pick_higher_seq_agent(conflict, graph)
        winning_output = conflict.conflicting_outputs.get(winner, "unknown")

        conflict.resolution_strategy  = ResolutionStrategy.ARBITRATE
        conflict.resolution_output    = {
            "winning_agent":  winner,
            "winning_output": winning_output,
            "reason":         reason,
        }
        conflict.resolution_reasoning = (
            f"Arbitrated in favor of {winner}: {reason}"
        )
        conflict.status      = ConflictStatus.RESOLVED
        conflict.resolved_at = datetime.utcnow()

        logger.info(
            "conflict_resolved",
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.ARBITRATE.value,
            winner=winner,
        )
        return conflict

    async def _llm_synthesize(self, conflict: Conflict) -> Dict[str, Any]:
        """LLM call to merge two conflicting outputs into one coherent response."""
        agents = conflict.agents_involved
        output_a = conflict.conflicting_outputs.get(agents[0], "") if len(agents) > 0 else ""
        output_b = conflict.conflicting_outputs.get(agents[1], "") if len(agents) > 1 else ""

        system = (
            "You are a conflict resolution engine for multi-agent AI systems. "
            "Given two conflicting agent outputs, produce a single coherent response "
            "that best resolves the conflict while preserving the most important elements "
            "from each agent. "
            "If the outputs are logically incompatible and cannot be merged, "
            "set synthesizable=false. "
            "Cite explicitly which elements you took from each agent."
        )

        user = (
            f"Agent A ({agents[0] if agents else 'unknown'}) output:\n{output_a}\n\n"
            f"Agent B ({agents[1] if len(agents) > 1 else 'unknown'}) output:\n{output_b}\n\n"
            f"Root cause: {conflict.root_cause_type.value if conflict.root_cause_type else 'unknown'}\n\n"
            "Synthesize a resolution."
        )

        try:
            result = self.client.chat_with_tool(
                system=system,
                user=user,
                tool=SYNTHESIZE_TOOL,
                tool_name="synthesize_outputs",
                max_tokens=512,
            )
            return result or {"synthesizable": False, "confidence": 0.0, "synthesized_output": ""}

        except Exception as e:
            logger.error("synthesis_failed", error=str(e))

        return {"synthesizable": False, "confidence": 0.0, "synthesized_output": ""}

    @staticmethod
    def _pick_higher_seq_agent(conflict: Conflict, graph: TraceGraph) -> str:
        """Return the agent with the highest max sequence number in the run."""
        max_seqs: Dict[str, int] = {}
        for ev in graph.events:
            if ev.agent_name in conflict.agents_involved:
                max_seqs[ev.agent_name] = max(
                    max_seqs.get(ev.agent_name, 0), ev.sequence_no
                )

        if not max_seqs:
            return conflict.agents_involved[0]

        return max(max_seqs, key=lambda a: max_seqs[a])

    @staticmethod
    def _build_corrected_context(
        conflict: Conflict, graph: TraceGraph
    ) -> Dict[str, Any]:
        """
        Merge context from both conflicting agents, fresher values winning.
        Used by RESET strategy.
        """
        merged: Dict[str, Any] = {}

        for agent in conflict.agents_involved:
            agent_events = sorted(
                graph.get_events_by_agent(agent),
                key=lambda e: e.sequence_no,
            )
            for ev in agent_events:
                merged.update(ev.input_context)

        return merged
