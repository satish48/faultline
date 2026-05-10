"""
AgentWatch — Conflict Memory Store
=====================================
Accumulates conflict patterns across runs and surfaces fix recommendations.

This is the compounding moat — the system gets smarter with every conflict it sees.
After 2+ occurrences of the same pattern, a specific fix recommendation is surfaced.

Storage: JSON file — simple, portable, readable.
Pattern matching: conflict_type + root_cause_type + frozenset(agents_involved)

Fix recommendation templates are specific — not generic advice:
  "rewrite the {agent} system prompt to explicitly handle {scenario}"
  NOT "improve your prompts"
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import structlog

from agentwatch.core.models.schemas import (
    Conflict,
    ConflictPattern,
    ConflictType,
    RootCauseType,
)

logger = structlog.get_logger(__name__)

# Surface recommendations for patterns seen this many times or more
RECOMMENDATION_THRESHOLD = 2

# Fix recommendation templates keyed by root_cause_type
FIX_TEMPLATES: Dict[str, str] = {
    RootCauseType.INSTRUCTION_AMBIGUITY.value: (
        "Rewrite the {agents} system prompts to explicitly define behavior "
        "when handling {scenario}. Add a negative example showing what NOT to do."
    ),
    RootCauseType.CONTEXT_WINDOW_MISMATCH.value: (
        "Pass shared state explicitly in the handoff payload between {agents}. "
        "Do not rely on agents having independently retrieved the same context."
    ),
    RootCauseType.TOOL_RESULT_DIVERGENCE.value: (
        "Add input validation before tool calls in {agents}. "
        "Ensure both agents call the tool with identical parameters, "
        "or designate one agent as the single source of truth for this tool."
    ),
    RootCauseType.LOGICAL_CONTRADICTION.value: (
        "Add a dedicated arbitration step between {agents} before either executes. "
        "Consider a Devil's Advocate review pattern to surface contradictions early."
    ),
    RootCauseType.STALE_STATE.value: (
        "Add timestamp validation to {agents} context ingestion. "
        "Reject context older than [threshold] seconds and re-fetch before acting."
    ),
    RootCauseType.PERMISSION_BOUNDARY_VIOLATION.value: (
        "Explicitly define the permission scope for {agents} in their system prompts. "
        "Add a permission check tool call before any state-mutating action."
    ),
    RootCauseType.UNKNOWN.value: (
        "Review the interaction between {agents} manually. "
        "Instrument additional logging at handoff points to capture more context."
    ),
}


class ConflictMemoryStore:
    """
    Persistent conflict pattern store.
    Loads from JSON on init, saves after every upsert.
    Thread-safe for single-process use (asyncio).
    """

    def __init__(self, store_path: str = "conflict_memory.json"):
        self.store_path = store_path
        self._patterns: Dict[str, ConflictPattern] = {}
        self._conflict_log: List[str] = []
        self._load()

    # ── Public API ────────────────────────────────────

    async def record(self, conflict: Conflict) -> None:
        """Log a conflict ID to the memory store."""
        self._conflict_log.append(conflict.conflict_id)
        self._save()

    async def find_pattern(self, conflict: Conflict) -> Optional[ConflictPattern]:
        """
        Search for an existing pattern matching this conflict.
        Match key: conflict_type + root_cause_type + sorted agent names.
        """
        key = self._pattern_key(conflict)
        return self._patterns.get(key)

    async def upsert_pattern(self, conflict: Conflict) -> ConflictPattern:
        """
        Create or update a conflict pattern from a resolved conflict.
        Increments occurrence_count on match.
        Generates fix recommendation on first creation.
        Persists after every upsert.
        """
        key = self._pattern_key(conflict)
        existing = self._patterns.get(key)

        if existing:
            existing.occurrence_count += 1
            existing.last_seen_at = datetime.utcnow()
            if conflict.conflict_id not in existing.example_conflict_ids:
                existing.example_conflict_ids.append(conflict.conflict_id)
            pattern = existing
        else:
            pattern = ConflictPattern(
                conflict_type=conflict.conflict_type,
                root_cause_type=conflict.root_cause_type,
                agents_involved=conflict.agents_involved,
                trigger_description=self._describe_trigger(conflict),
                fix_recommendation=self._generate_fix(conflict),
                fix_specificity=self._generate_specificity(conflict),
                example_conflict_ids=[conflict.conflict_id],
            )
            self._patterns[key] = pattern

        self._save()

        logger.info(
            "pattern_upserted",
            key=key,
            occurrences=pattern.occurrence_count,
            conflict_type=conflict.conflict_type.value,
        )

        return pattern

    async def get_all_patterns(self) -> List[ConflictPattern]:
        """Return all known patterns, sorted by occurrence_count descending."""
        return sorted(
            self._patterns.values(),
            key=lambda p: p.occurrence_count,
            reverse=True,
        )

    async def get_recommendations(self) -> List[str]:
        """
        Return fix recommendations for patterns that have occurred
        at least RECOMMENDATION_THRESHOLD times.
        Most frequent patterns first.
        """
        patterns = await self.get_all_patterns()
        return [
            p.fix_recommendation
            for p in patterns
            if p.occurrence_count >= RECOMMENDATION_THRESHOLD
        ]

    async def get_stats(self) -> dict:
        """Summary statistics for the dashboard."""
        patterns = list(self._patterns.values())
        return {
            "total_patterns":    len(patterns),
            "total_conflicts":   len(self._conflict_log),
            "recommendations":   len(await self.get_recommendations()),
            "most_common_type":  (
                max(patterns, key=lambda p: p.occurrence_count).conflict_type.value
                if patterns else None
            ),
        }

    def reset(self) -> None:
        """Clear all patterns and logs. Used in testing."""
        self._patterns = {}
        self._conflict_log = []
        self._save()

    # ── Internal ──────────────────────────────────────

    @staticmethod
    def _pattern_key(conflict: Conflict) -> str:
        """
        Deterministic key for pattern matching.
        Agents sorted so order doesn't matter.
        """
        agents_key = "+".join(sorted(conflict.agents_involved))
        rc = conflict.root_cause_type.value if conflict.root_cause_type else "unknown"
        return f"{conflict.conflict_type.value}::{rc}::{agents_key}"

    @staticmethod
    def _describe_trigger(conflict: Conflict) -> str:
        agents = " and ".join(conflict.agents_involved)
        rc = conflict.root_cause_type.value if conflict.root_cause_type else "unknown cause"
        return (
            f"{conflict.conflict_type.value.capitalize()} conflict between {agents} "
            f"caused by {rc.replace('_', ' ')}."
        )

    @staticmethod
    def _generate_fix(conflict: Conflict) -> str:
        """Fill the fix template for this conflict's root cause."""
        rc = conflict.root_cause_type.value if conflict.root_cause_type else RootCauseType.UNKNOWN.value
        template = FIX_TEMPLATES.get(rc, FIX_TEMPLATES[RootCauseType.UNKNOWN.value])

        agents_str = " and ".join(conflict.agents_involved)
        scenario = conflict.conflict_type.value.replace("_", " ")

        return template.format(agents=agents_str, scenario=scenario)

    @staticmethod
    def _generate_specificity(conflict: Conflict) -> str:
        """One-line specificity statement for the UI."""
        rc = conflict.root_cause_type.value if conflict.root_cause_type else "unknown"
        agents = ", ".join(conflict.agents_involved)
        return f"Root cause: {rc.replace('_', ' ')} — affects {agents}"

    def _load(self) -> None:
        if not os.path.exists(self.store_path):
            return
        try:
            with open(self.store_path) as f:
                data = json.load(f)
            for key, raw in data.get("patterns", {}).items():
                self._patterns[key] = ConflictPattern.model_validate(raw)
            self._conflict_log = data.get("conflict_log", [])
        except Exception as e:
            logger.error("memory_store_load_failed", error=str(e))

    def _save(self) -> None:
        try:
            data = {
                "patterns":     {k: v.model_dump(mode="json") for k, v in self._patterns.items()},
                "conflict_log": self._conflict_log,
            }
            with open(self.store_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error("memory_store_save_failed", error=str(e))
