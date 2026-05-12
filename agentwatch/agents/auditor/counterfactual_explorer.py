"""
AgentWatch — Counterfactual Explorer
======================================
Explores what-if scenarios for DecisionNodes by modifying context and asking
the LLM what decision the agent would have made with different inputs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import asyncio
import structlog

from agentwatch.core.llm.client import LLMClient
from agentwatch.core.models.schemas import (
    CounterfactualResult,
    DecisionNode,
    TraceGraph,
)

logger = structlog.get_logger(__name__)

SIMULATE_TOOL = {
    "name": "simulate_counterfactual",
    "description": "Simulate what decision an agent would make with a modified input context",
    "input_schema": {
        "type": "object",
        "properties": {
            "chosen_option":   {"type": "string"},
            "reasoning":       {"type": "string"},
            "outcome_changed": {"type": "boolean"},
            "confidence":      {"type": "number"},
        },
        "required": ["chosen_option", "reasoning", "outcome_changed", "confidence"],
    },
}

_THRESHOLD_KEYS = {"risk_score", "threshold", "score", "amount", "value", "limit", "rate"}


class CounterfactualExplorer:
    """
    Explores counterfactual outcomes for DecisionNodes.
    Answers: "Would the agent have decided differently if its context had been different?"
    """

    def __init__(self, client: LLMClient):
        self.client = client
        self._semaphore = asyncio.Semaphore(5)

    async def explore(
        self,
        graph: TraceGraph,
        node: DecisionNode,
        modified_context: Dict[str, Any],
    ) -> CounterfactualResult:
        """
        Run a single counterfactual: what would the agent have decided
        given modified_context instead of node.context_snapshot?
        """
        divergence_point = self._find_divergence_point(node.context_snapshot, modified_context)

        system = (
            "You are a counterfactual reasoning engine for AI agent decisions. "
            "Given an agent's original decision context and a modified context, "
            "predict what decision the agent would have made with the modified input. "
            "Only reason from the provided data — never invent information."
        )

        user = (
            f"Agent: {node.agent_name}\n"
            f"Decision type: {node.decision_type}\n"
            f"Original context: {json.dumps(node.context_snapshot, default=str)}\n"
            f"Original decision: {node.chosen_option}\n"
            f"Modified context: {json.dumps(modified_context, default=str)}\n"
            f"Options available: {node.options_available}\n\n"
            "What decision would the agent make with the modified context?"
        )

        result = self.client.chat_with_tool(
            system=system,
            user=user,
            tool=SIMULATE_TOOL,
            tool_name="simulate_counterfactual",
            max_tokens=512,
        ) or {}

        cf_outcome = result.get("chosen_option", node.chosen_option)
        outcome_changed = result.get("outcome_changed", cf_outcome != node.chosen_option)
        cf_confidence = result.get("confidence", node.confidence)

        sensitivity = self._sensitivity_score(node.confidence, cf_confidence, outcome_changed)

        logger.info(
            "counterfactual_explored",
            run_id=graph.run_id,
            node_id=node.node_id,
            outcome_changed=outcome_changed,
            divergence_point=divergence_point,
        )

        return CounterfactualResult(
            node_id=node.node_id,
            run_id=graph.run_id,
            counterfactual_input=modified_context,
            original_outcome=node.chosen_option,
            counterfactual_outcome=cf_outcome,
            outcome_changed=outcome_changed,
            sensitivity_score=sensitivity,
            divergence_point=divergence_point,
            explanation=result.get("reasoning", ""),
        )

    async def batch_explore(
        self,
        graph,
        nodes: list,
        modifications: list,
    ) -> list:
        """
        Run multiple counterfactuals in parallel.
        Zips nodes with modifications — must be same length.
        Max 5 concurrent LLM calls via semaphore.
        """
        if len(nodes) != len(modifications):
            raise ValueError("nodes and modifications must be same length")

        async def _run(node, mod):
            async with self._semaphore:
                return await self.explore(graph, node, mod)

        results = await asyncio.gather(
            *[_run(n, m) for n, m in zip(nodes, modifications)],
            return_exceptions=True,
        )

        clean = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"counterfactual_failed: node={nodes[i].node_id} error={r}")
            else:
                clean.append(r)

        return clean

    async def find_sensitive_decisions(
        self,
        graph: TraceGraph,
        nodes: List[DecisionNode],
    ) -> List[DecisionNode]:
        """
        Returns DecisionNodes where auto-modified context produces a different outcome.
        These are the decisions most sensitive to input variation.
        """
        sensitive: List[DecisionNode] = []

        for node in nodes:
            modified = self._auto_modify(node.context_snapshot)
            result = await self.explore(graph, node, modified)
            if result.outcome_changed:
                sensitive.append(node)

        return sensitive

    @staticmethod
    def _sensitivity_score(
        original_confidence: Optional[float],
        cf_confidence: Optional[float],
        outcome_changed: bool,
    ) -> float:
        delta = abs((original_confidence or 0.5) - (cf_confidence or 0.5))
        if outcome_changed:
            return round(min(1.0, 0.6 + delta), 3)
        return round(delta * 0.3, 3)

    @staticmethod
    def _find_divergence_point(
        original_ctx: Dict[str, Any], modified_ctx: Dict[str, Any]
    ) -> str:
        changed = [
            k for k in original_ctx
            if k in modified_ctx and original_ctx[k] != modified_ctx[k]
        ]
        changed += [k for k in modified_ctx if k not in original_ctx]

        if not changed:
            return "no_change"
        if len(changed) == 1:
            return changed[0]
        return "multiple_keys:" + ",".join(sorted(changed))

    @staticmethod
    def _auto_modify(ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a minimally-modified version of ctx for counterfactual testing.
        Flips the first boolean found, or doubles the first numeric threshold key.
        """
        modified = dict(ctx)

        for key, val in ctx.items():
            if isinstance(val, bool):
                modified[key] = not val
                return modified

        for key, val in ctx.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if any(tk in key.lower() for tk in _THRESHOLD_KEYS):
                    modified[key] = val * 2
                    return modified

        return modified

