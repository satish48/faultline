"""
AgentWatch — Conflict Sensor
==============================
Scans a TraceGraph and detects inter-agent conflicts at handoff points.

Three conflict types, three detection methods:
  SEMANTIC   — LLM checks if two agent outputs contradict each other
  PROCEDURAL — Rule-based: two agents claim same tool within 3 sequence steps
  TEMPORAL   — Rule-based: agent context contains stale timestamp vs prior output

Design decisions:
- SEMANTIC uses LLM (only type that requires language understanding)
- PROCEDURAL and TEMPORAL are fully deterministic — no LLM, no hallucination risk
- Each handoff pair is checked independently
- Returns empty list if no conflicts — never raises on clean runs
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

from agentwatch.core.llm.client import LLMClient

from agentwatch.core.models.schemas import (
    AgentEvent,
    AgentEventType,
    Conflict,
    ConflictStatus,
    ConflictType,
    TraceGraph,
)

logger = structlog.get_logger(__name__)

CONTRADICTION_TOOL = {
    "name": "check_contradiction",
    "description": "Check if two agent outputs semantically contradict each other",
    "input_schema": {
        "type": "object",
        "properties": {
            "contradicts":   {"type": "boolean"},
            "explanation":   {"type": "string"},
            "confidence":    {"type": "number"},
        },
        "required": ["contradicts", "explanation", "confidence"],
    },
}


class ConflictSensor:
    """
    Passive scanner — wraps any TraceGraph and returns detected conflicts.
    Does not modify the graph. Does not resolve conflicts.
    """

    def __init__(
        self,
        client: LLMClient,
        procedural_window: int = 3,
        semantic_confidence_threshold: float = 0.75,
    ):
        self.client = client
        self.procedural_window = procedural_window
        self.semantic_threshold = semantic_confidence_threshold

    async def scan(self, graph: TraceGraph) -> List[Conflict]:
        """
        Full conflict scan across all three types.
        Returns all detected conflicts — can be empty.
        """
        if len(graph.agents_involved) < 2:
            return []

        conflicts: List[Conflict] = []

        semantic   = await self._detect_semantic(graph)
        procedural = self._detect_procedural(graph)
        temporal   = self._detect_temporal(graph)

        conflicts.extend(semantic)
        conflicts.extend(procedural)
        conflicts.extend(temporal)

        logger.info(
            "conflict_scan_complete",
            run_id=graph.run_id,
            semantic=len(semantic),
            procedural=len(procedural),
            temporal=len(temporal),
            total=len(conflicts),
        )

        return conflicts

    # ── Detection methods ─────────────────────────────

    async def _detect_semantic(self, graph: TraceGraph) -> List[Conflict]:
        """
        Checks for semantic contradictions between agent outputs.

        Pass 1 — handoff seams: for each HANDOFF event, compare the
        sending agent's last output with the receiving agent's first output.

        Pass 2 — all-pairs: compare the most recent DECISION output of
        every unique agent pair not already caught by pass 1.
        """
        conflicts: List[Conflict] = []
        seen_pairs: set = set()

        # Pass 1: handoff-based detection
        for handoff in graph.get_handoffs():
            from_agent = handoff.agent_name
            to_agent = handoff.metadata.get("to_agent")
            if not to_agent:
                continue

            from_output = self._get_last_output(graph, from_agent, handoff.sequence_no)
            to_output = self._get_first_output(graph, to_agent, handoff.sequence_no)

            if from_output is None or to_output is None:
                continue

            result = await self._check_semantic(
                str(from_output), str(to_output), from_agent, to_agent
            )

            if result["contradicts"] and result["confidence"] >= self.semantic_threshold:
                pair = frozenset({from_agent, to_agent})
                seen_pairs.add(pair)
                conflict = Conflict(
                    run_id=graph.run_id,
                    conflict_type=ConflictType.SEMANTIC,
                    agents_involved=[from_agent, to_agent],
                    conflicting_outputs={
                        from_agent: from_output,
                        to_agent: to_output,
                    },
                    triggering_event_ids=[handoff.event_id],
                    status=ConflictStatus.DETECTED,
                )
                conflicts.append(conflict)
                logger.warning(
                    "semantic_conflict_detected",
                    run_id=graph.run_id,
                    agents=[from_agent, to_agent],
                    confidence=result["confidence"],
                )

        # Pass 2: all-pairs check on most recent DECISION outputs
        agents = graph.agents_involved
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent_a, agent_b = agents[i], agents[j]
                pair = frozenset({agent_a, agent_b})
                if pair in seen_pairs:
                    continue

                decision_evs_a = [
                    e for e in graph.events
                    if e.agent_name == agent_a
                    and e.event_type == AgentEventType.DECISION
                    and e.output is not None
                ]
                decision_evs_b = [
                    e for e in graph.events
                    if e.agent_name == agent_b
                    and e.event_type == AgentEventType.DECISION
                    and e.output is not None
                ]

                if not decision_evs_a or not decision_evs_b:
                    continue

                ev_a = max(decision_evs_a, key=lambda e: e.sequence_no)
                ev_b = max(decision_evs_b, key=lambda e: e.sequence_no)

                result = await self._check_semantic(
                    str(ev_a.output), str(ev_b.output), agent_a, agent_b
                )

                if result["contradicts"] and result["confidence"] >= self.semantic_threshold:
                    seen_pairs.add(pair)
                    conflict = Conflict(
                        run_id=graph.run_id,
                        conflict_type=ConflictType.SEMANTIC,
                        agents_involved=[agent_a, agent_b],
                        conflicting_outputs={
                            agent_a: ev_a.output,
                            agent_b: ev_b.output,
                        },
                        triggering_event_ids=[ev_a.event_id, ev_b.event_id],
                        status=ConflictStatus.DETECTED,
                    )
                    conflicts.append(conflict)
                    logger.warning(
                        "semantic_conflict_detected",
                        run_id=graph.run_id,
                        agents=[agent_a, agent_b],
                        confidence=result["confidence"],
                    )

        return conflicts

    def _detect_procedural(self, graph: TraceGraph) -> List[Conflict]:
        """
        Detects when two agents both emit TOOL_CALL events for the same
        tool within self.procedural_window sequence steps of each other.

        Fully deterministic — no LLM involved.
        """
        conflicts: List[Conflict] = []
        tool_calls = graph.get_events_by_type(AgentEventType.TOOL_CALL)

        # Group by tool_name
        tool_groups: Dict[str, List[AgentEvent]] = {}
        for ev in tool_calls:
            if ev.tool_name:
                tool_groups.setdefault(ev.tool_name, []).append(ev)

        for tool_name, events in tool_groups.items():
            if len(events) < 2:
                continue

            # Check each pair for proximity
            for i in range(len(events)):
                for j in range(i + 1, len(events)):
                    ev_a, ev_b = events[i], events[j]

                    # Different agents, within window
                    if (
                        ev_a.agent_name != ev_b.agent_name
                        and abs(ev_a.sequence_no - ev_b.sequence_no) <= self.procedural_window
                    ):
                        conflict = Conflict(
                            run_id=graph.run_id,
                            conflict_type=ConflictType.PROCEDURAL,
                            agents_involved=[ev_a.agent_name, ev_b.agent_name],
                            conflicting_outputs={
                                ev_a.agent_name: f"called {tool_name} at seq {ev_a.sequence_no}",
                                ev_b.agent_name: f"called {tool_name} at seq {ev_b.sequence_no}",
                            },
                            triggering_event_ids=[ev_a.event_id, ev_b.event_id],
                            status=ConflictStatus.DETECTED,
                        )
                        conflicts.append(conflict)
                        logger.warning(
                            "procedural_conflict_detected",
                            run_id=graph.run_id,
                            tool=tool_name,
                            agents=[ev_a.agent_name, ev_b.agent_name],
                        )

        return conflicts

    def _detect_temporal(self, graph: TraceGraph) -> List[Conflict]:
        """
        Detects when an agent's input_context contains a timestamp field
        that is older than when the event itself occurred, and another agent
        had any event before this one (implying fresher data was available).

        Fully deterministic — no LLM involved.
        """
        if len(graph.agents_involved) < 2:
            return []

        conflicts: List[Conflict] = []
        staleness_keys = ["timestamp", "updated_at", "created_at", "modified_at", "version_date"]

        for ev in graph.events:
            for key in staleness_keys:
                ctx_value = ev.input_context.get(key)
                if ctx_value is None:
                    continue

                ctx_dt = self._try_parse_datetime(str(ctx_value))
                if ctx_dt is None:
                    continue

                event_ts = ev.timestamp.replace(tzinfo=None) if ev.timestamp.tzinfo else ev.timestamp
                if ctx_dt < event_ts:
                    fresher_agent = self._find_fresher_producer(
                        graph, ev.agent_name, ev.sequence_no
                    )
                    if fresher_agent and fresher_agent != ev.agent_name:
                        conflict = Conflict(
                            run_id=graph.run_id,
                            conflict_type=ConflictType.TEMPORAL,
                            agents_involved=[ev.agent_name, fresher_agent],
                            conflicting_outputs={
                                ev.agent_name: f"using stale {key}={ctx_value}",
                                fresher_agent: f"produced fresher {key} data",
                            },
                            triggering_event_ids=[ev.event_id],
                            status=ConflictStatus.DETECTED,
                        )
                        conflicts.append(conflict)
                        logger.warning(
                            "temporal_conflict_detected",
                            run_id=graph.run_id,
                            agent=ev.agent_name,
                            stale_key=key,
                        )
                        break  # one conflict per event is enough

        return conflicts

    # ── LLM helper ────────────────────────────────────

    async def _check_semantic(
        self,
        output_a: str,
        output_b: str,
        agent_a: str,
        agent_b: str,
    ) -> Dict[str, Any]:
        """
        Single LLM call to check semantic contradiction.
        Conservative: only flags clear contradictions, not disagreements.
        """
        system = (
            "You are a semantic contradiction detector for AI agent outputs. "
            "You check if two agent outputs directly contradict each other. "
            "A contradiction means they assert OPPOSITE conclusions about the same subject. "
            "Mere differences in approach, detail level, or tone are NOT contradictions. "
            "Only flag contradicts=true when the outputs are logically incompatible — "
            "accepting the conclusion of one requires rejecting the conclusion of the other. "
            "Be conservative: when in doubt, contradicts=false."
        )

        user = (
            f"Agent A ({agent_a}) output:\n{output_a}\n\n"
            f"Agent B ({agent_b}) output:\n{output_b}\n\n"
            "Do these outputs contradict each other?"
        )

        try:
            result = self.client.chat_with_tool(
                system=system,
                user=user,
                tool=CONTRADICTION_TOOL,
                tool_name="check_contradiction",
                max_tokens=256,
            )
            return result or {"contradicts": False, "explanation": "no result", "confidence": 0.0}

        except Exception as e:
            logger.error("semantic_check_failed", error=str(e))

        return {"contradicts": False, "explanation": "check failed", "confidence": 0.0}

    # ── Utility helpers ───────────────────────────────

    @staticmethod
    def _get_last_output(
        graph: TraceGraph, agent_name: str, before_seq: int
    ) -> Optional[Any]:
        """Get the most recent output from an agent up to and including before_seq."""
        agent_events = [
            e for e in graph.events
            if e.agent_name == agent_name
            and e.sequence_no <= before_seq
            and e.output is not None
        ]
        if not agent_events:
            return None
        return max(agent_events, key=lambda e: e.sequence_no).output

    @staticmethod
    def _get_first_output(
        graph: TraceGraph, agent_name: str, after_seq: int
    ) -> Optional[Any]:
        """Get the first output from an agent after a given sequence number."""
        agent_events = [
            e for e in graph.events
            if e.agent_name == agent_name
            and e.sequence_no > after_seq
            and e.output is not None
        ]
        if not agent_events:
            return None
        return min(agent_events, key=lambda e: e.sequence_no).output

    @staticmethod
    def _try_parse_datetime(value: str) -> Optional[datetime]:
        """Try common datetime formats. Returns None if unparseable."""
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value[:19], fmt) if len(value) >= len(fmt) else datetime.strptime(value, fmt)
            except (ValueError, TypeError):
                continue
        return None

    @staticmethod
    def _find_fresher_producer(
        graph: TraceGraph,
        stale_agent: str,
        before_seq: int,
    ) -> Optional[str]:
        """
        Return the first agent other than stale_agent that had any event
        before before_seq. Presence of any earlier event means that agent
        had fresher state available at the time the stale event occurred.
        """
        for ev in sorted(graph.events, key=lambda e: e.sequence_no):
            if ev.sequence_no >= before_seq:
                break
            if ev.agent_name != stale_agent:
                return ev.agent_name
        return None
