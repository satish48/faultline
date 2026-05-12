"""
AgentWatch — Decision Decomposer
==================================
Reads a TraceGraph and extracts structured DecisionNodes from raw AgentEvents.

Key design decisions:
- Batches all DECISION + TOOL_CALL events into ONE LLM call (not one per event)
  → saves cost, maintains consistency across decisions in same run
- Structured output via tool_use — never parses free text
- is_critical flag is deterministic (rule-based), not LLM-generated
  → no hallucination risk on the most important flag
- source_event_ids always populated → every node fully traceable
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import structlog

from agentwatch.core.llm.client import LLMClient

from agentwatch.core.models.schemas import (
    AgentEvent,
    AgentEventType,
    DecisionNode,
    TraceGraph,
)

logger = structlog.get_logger(__name__)

# Events worth decomposing into decision nodes
DECOMPOSABLE_TYPES = {
    AgentEventType.DECISION,
    AgentEventType.TOOL_CALL,
    AgentEventType.HANDOFF,
}

# Tool schema for structured extraction
DECOMPOSE_TOOL = {
    "name": "extract_decision_nodes",
    "description": "Extract structured decision nodes from agent events",
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "event_id":           {"type": "string"},
                        "decision_type":      {"type": "string"},
                        "options_available":  {"type": "array", "items": {"type": "string"}},
                        "chosen_option":      {"type": "string"},
                        "confidence":         {"type": ["number", "null"]},
                        "alternative_option": {"type": ["string", "null"]},
                        "alternative_outcome":{"type": ["string", "null"]},
                    },
                    "required": ["event_id", "decision_type", "chosen_option", "options_available"],
                },
            }
        },
        "required": ["decisions"],
    },
}


class DecisionDecomposer:
    """
    Extracts structured DecisionNodes from a TraceGraph.

    One LLM call per run (batched) — not per event.
    is_critical is determined by rule, not LLM.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    async def decompose(self, graph: TraceGraph) -> List[DecisionNode]:
        """
        Extract DecisionNodes from all decomposable events in the graph.
        Returns nodes sorted by sequence_no.
        """
        target_events = [
            e for e in graph.events if e.event_type in DECOMPOSABLE_TYPES
        ]

        if not target_events:
            logger.info("no_decomposable_events", run_id=graph.run_id)
            return []

        logger.info(
            "decomposing_events",
            run_id=graph.run_id,
            count=len(target_events),
        )

        extracted = await self._llm_extract(target_events)
        nodes = self._build_nodes(extracted, target_events, graph.run_id)

        logger.info(
            "decomposition_complete",
            run_id=graph.run_id,
            nodes=len(nodes),
            critical=sum(1 for n in nodes if n.is_critical),
        )
        return sorted(nodes, key=lambda n: n.sequence_no)

    # ── Internal ──────────────────────────────────────

    async def _llm_extract(
        self, events: List[AgentEvent]
    ) -> List[Dict[str, Any]]:
        """
        Single batched LLM call to extract decision data from all events.
        Uses tool_use for structured output — no free-text parsing.
        """
        events_payload = [
            {
                "event_id":    e.event_id,
                "agent_name":  e.agent_name,
                "event_type":  e.event_type.value,
                "sequence_no": e.sequence_no,
                "input_context": e.input_context,
                "output":      str(e.output) if e.output is not None else None,
                "tool_name":   e.tool_name,
                "tool_args":   e.tool_args,
            }
            for e in events
        ]

        system = (
            "You are an agent decision analyst. "
            "Extract ONLY what is factually present in the event data. "
            "Never infer, assume, or fabricate information. "
            "If options are not explicitly listed, use [chosen_option, 'alternative not recorded']. "
            "If confidence is not a numeric value in the output, set it to null. "
            "decision_type must be one of: tool_selection, routing, response_generation, "
            "state_mutation, handoff, error_handling."
        )

        user = (
            f"Extract a structured decision node for each of these {len(events_payload)} events.\n\n"
            f"Events:\n{json.dumps(events_payload, indent=2, default=str)}"
        )

        result = self.client.chat_with_tool(
            system=system,
            user=user,
            tool=DECOMPOSE_TOOL,
            tool_name="extract_decision_nodes",
            max_tokens=4096,
        )

        return (result or {}).get("decisions", [])

    def _build_nodes(
        self,
        extracted: List[Dict[str, Any]],
        events: List[AgentEvent],
        run_id: str,
    ) -> List[DecisionNode]:
        """
        Merge LLM-extracted data with rule-based is_critical determination.
        Maps extracted items back to source events by event_id.
        """
        event_map = {e.event_id: e for e in events}
        nodes: List[DecisionNode] = []

        for item in extracted:
            event_id = item.get("event_id")
            source_event = event_map.get(event_id)

            if source_event is None:
                logger.warning("decomposer_unknown_event_id", event_id=event_id)
                continue

            node = DecisionNode(
                run_id=run_id,
                agent_name=source_event.agent_name,
                sequence_no=source_event.sequence_no,
                context_snapshot=source_event.input_context,
                decision_type=item.get("decision_type", "unknown"),
                options_available=item.get("options_available", []),
                chosen_option=item.get("chosen_option", "unknown"),
                confidence=item.get("confidence"),
                alternative_option=item.get("alternative_option"),
                alternative_outcome=item.get("alternative_outcome"),
                is_critical=self._is_critical(source_event, item),
                source_event_ids=[event_id],
            )
            nodes.append(node)

        return nodes

    @staticmethod
    def _is_critical(event: AgentEvent, extracted: Dict[str, Any]) -> bool:
        """
        Rule-based — never LLM-generated.
        A decision is critical if it:
        - Is a HANDOFF, TOOL_CALL, or ERROR event type
        - Has low confidence (< 0.6) — agent was uncertain
        - chosen_option contains a high-stakes action keyword
        """
        if event.event_type in {AgentEventType.HANDOFF, AgentEventType.TOOL_CALL, AgentEventType.ERROR}:
            return True

        confidence = extracted.get("confidence")
        if confidence is not None and confidence < 0.6:
            return True

        chosen = str(extracted.get("chosen_option", "")).lower()
        if any(kw in chosen for kw in ["approve", "block", "execute", "refund", "escalate"]):
            return True

        return False
