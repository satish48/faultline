"""
Tests for Conflict Detector — Pillar B.
All four components: ConflictSensor, RootCauseClassifier,
ResolutionStrategist, ConflictMemoryStore.
Run with: pytest agentwatch/tests/test_conflict.py -v
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from agentwatch.core.models.schemas import (
    AgentEvent,
    AgentEventType,
    Conflict,
    ConflictStatus,
    ConflictType,
    ResolutionStrategy,
    RootCauseType,
    TraceGraph,
)
from agentwatch.agents.conflict.conflict_sensor import ConflictSensor
from agentwatch.agents.conflict.root_cause_classifier import RootCauseClassifier
from agentwatch.agents.conflict.resolution_strategist import ResolutionStrategist
from agentwatch.agents.conflict.conflict_memory import ConflictMemoryStore


# ── Helpers ───────────────────────────────────────────

def make_event(agent, etype, seq, run_id="run-t", output=None,
               context=None, tool_name=None, to_agent=None):
    meta = {}
    if to_agent:
        meta["to_agent"] = to_agent
    return AgentEvent(
        run_id=run_id,
        agent_name=agent,
        event_type=etype,
        sequence_no=seq,
        input_context=context or {},
        output=output,
        tool_name=tool_name,
        metadata=meta,
    )


def make_graph(events, run_id="run-t"):
    agents = list({e.agent_name for e in events})
    return TraceGraph(
        run_id=run_id,
        session_id="sess-t",
        agents_involved=agents,
        events=events,
    )


def make_mock_client(tool_name, output):
    client = MagicMock()
    client.chat_with_tool.return_value = output
    return client


def make_conflict(
    conflict_type=ConflictType.SEMANTIC,
    agents=None,
    run_id="run-t",
    root_cause=None,
    root_cause_confidence=0.8,
):
    agents = agents or ["agent_a", "agent_b"]
    c = Conflict(
        run_id=run_id,
        conflict_type=conflict_type,
        agents_involved=agents,
        conflicting_outputs={
            agents[0]: "approve refund",
            agents[1]: "block refund",
        },
    )
    if root_cause:
        c.root_cause_type = root_cause
        c.root_cause_confidence = root_cause_confidence
    return c



# ConflictSensor Tests


@pytest.mark.asyncio
async def test_semantic_conflict_detected():
    """Two agents with contradictory outputs → SEMANTIC conflict."""
    events = [
        make_event("cs_agent", AgentEventType.DECISION, 1, output="approve refund"),
        make_event("cs_agent", AgentEventType.HANDOFF, 2, to_agent="compliance_agent",
                   output="handoff to compliance"),
        make_event("compliance_agent", AgentEventType.DECISION, 3,
                   output="block refund — risk threshold exceeded"),
    ]
    graph = make_graph(events)

    mock_output = {
        "contradicts": True,
        "explanation": "approve vs block are opposite decisions on same request",
        "confidence": 0.92,
    }
    client = make_mock_client("check_contradiction", mock_output)
    sensor = ConflictSensor(client=client, semantic_confidence_threshold=0.75)

    conflicts = await sensor.scan(graph)

    semantic = [c for c in conflicts if c.conflict_type == ConflictType.SEMANTIC]
    assert len(semantic) >= 1
    assert "cs_agent" in semantic[0].agents_involved
    assert "compliance_agent" in semantic[0].agents_involved


@pytest.mark.asyncio
async def test_no_false_positive_on_non_contradiction():
    """Agents with different but compatible outputs → no SEMANTIC conflict."""
    events = [
        make_event("agent_a", AgentEventType.DECISION, 1, output="proceed with caution"),
        make_event("agent_a", AgentEventType.HANDOFF, 2, to_agent="agent_b",
                   output="handoff"),
        make_event("agent_b", AgentEventType.DECISION, 3, output="proceed carefully"),
    ]
    graph = make_graph(events)

    mock_output = {
        "contradicts": False,
        "explanation": "both suggest proceeding — no contradiction",
        "confidence": 0.1,
    }
    client = make_mock_client("check_contradiction", mock_output)
    sensor = ConflictSensor(client=client)
    conflicts = await sensor.scan(graph)

    semantic = [c for c in conflicts if c.conflict_type == ConflictType.SEMANTIC]
    assert len(semantic) == 0


@pytest.mark.asyncio
async def test_procedural_conflict_detected():
    """Two agents both call same tool within window → PROCEDURAL conflict."""
    events = [
        make_event("billing_agent", AgentEventType.TOOL_CALL, 1,
                   tool_name="process_refund"),
        make_event("cs_agent", AgentEventType.TOOL_CALL, 2,
                   tool_name="process_refund"),
    ]
    graph = make_graph(events)

    client = MagicMock()
    sensor = ConflictSensor(client=client, procedural_window=3)
    conflicts = await sensor.scan(graph)

    procedural = [c for c in conflicts if c.conflict_type == ConflictType.PROCEDURAL]
    assert len(procedural) >= 1
    assert "billing_agent" in procedural[0].agents_involved
    assert "cs_agent" in procedural[0].agents_involved


@pytest.mark.asyncio
async def test_procedural_no_conflict_outside_window():
    """Same tool but far apart in sequence → no PROCEDURAL conflict."""
    events = [
        make_event("billing", AgentEventType.TOOL_CALL, 1, tool_name="lookup_order"),
        make_event("cs", AgentEventType.TOOL_CALL, 10, tool_name="lookup_order"),
    ]
    graph = make_graph(events)
    client = MagicMock()
    sensor = ConflictSensor(client=client, procedural_window=3)
    conflicts = await sensor.scan(graph)
    procedural = [c for c in conflicts if c.conflict_type == ConflictType.PROCEDURAL]
    assert len(procedural) == 0


@pytest.mark.asyncio
async def test_temporal_conflict_detected():
    """Agent context has timestamp older than run → TEMPORAL conflict."""
    stale_ts = (datetime.utcnow() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")

    # Another agent produced fresher timestamp data
    events = [
        make_event("data_agent", AgentEventType.TOOL_RESULT, 1,
                   output="fetched fresh order data with updated_at"),
        make_event("billing_agent", AgentEventType.DECISION, 2,
                   context={"order_id": "o1", "updated_at": stale_ts},
                   output="processed"),
    ]
    graph = make_graph(events)
    client = MagicMock()
    sensor = ConflictSensor(client=client)
    conflicts = await sensor.scan(graph)

    temporal = [c for c in conflicts if c.conflict_type == ConflictType.TEMPORAL]
    assert len(temporal) >= 1
    assert "billing_agent" in temporal[0].agents_involved


@pytest.mark.asyncio
async def test_single_agent_graph_no_conflicts():
    """Only one agent → no conflicts possible."""
    events = [
        make_event("solo_agent", AgentEventType.DECISION, 1, output="approve"),
        make_event("solo_agent", AgentEventType.END, 2, output="done"),
    ]
    graph = make_graph(events)
    client = MagicMock()
    sensor = ConflictSensor(client=client)
    conflicts = await sensor.scan(graph)
    assert conflicts == []
    client.chat_with_tool.assert_not_called()



# RootCauseClassifier Tests


@pytest.mark.asyncio
async def test_root_cause_classification():
    """Conflict classified with instruction_ambiguity."""
    conflict = make_conflict(ConflictType.SEMANTIC)
    graph = make_graph([
        make_event("agent_a", AgentEventType.DECISION, 1, output="approve"),
        make_event("agent_b", AgentEventType.DECISION, 2, output="block"),
    ])

    mock_output = {
        "root_cause_type": "instruction_ambiguity",
        "root_cause_detail": "Both agents received conflicting instructions about refund threshold",
        "confidence": 0.85,
    }
    client = make_mock_client("classify_root_cause", mock_output)
    classifier = RootCauseClassifier(client=client)

    enriched = await classifier.classify(conflict, graph)

    assert enriched.root_cause_type == RootCauseType.INSTRUCTION_AMBIGUITY
    assert enriched.root_cause_confidence == pytest.approx(0.85)
    assert enriched.status == ConflictStatus.RESOLVING


@pytest.mark.asyncio
async def test_batch_classify_returns_all():
    """batch_classify processes all conflicts."""
    conflicts = [make_conflict() for _ in range(3)]
    graph = make_graph([
        make_event("agent_a", AgentEventType.DECISION, i+1, output=f"out_{i}")
        for i in range(3)
    ])

    mock_output = {
        "root_cause_type": "context_window_mismatch",
        "root_cause_detail": "Agents saw different state slices",
        "confidence": 0.7,
    }
    client = make_mock_client("classify_root_cause", mock_output)
    classifier = RootCauseClassifier(client=client)

    results = await classifier.batch_classify(conflicts, graph)

    assert len(results) == 3
    assert all(r.root_cause_type == RootCauseType.CONTEXT_WINDOW_MISMATCH for r in results)



# ResolutionStrategist Tests


@pytest.mark.asyncio
async def test_semantic_conflict_synthesized():
    """Semantic conflict with high synthesis confidence → SYNTHESIZE."""
    conflict = make_conflict(
        ConflictType.SEMANTIC,
        root_cause=RootCauseType.INSTRUCTION_AMBIGUITY,
        root_cause_confidence=0.8,
    )
    graph = make_graph([
        make_event("agent_a", AgentEventType.DECISION, 1, output="approve"),
        make_event("agent_b", AgentEventType.DECISION, 2, output="block"),
    ])

    mock_output = {
        "synthesized_output": "Approve refund with compliance review flag",
        "elements_from_a": "approval decision",
        "elements_from_b": "compliance flag requirement",
        "synthesizable": True,
        "confidence": 0.85,
    }
    client = make_mock_client("synthesize_outputs", mock_output)
    strategist = ResolutionStrategist(client=client)

    resolved = await strategist.resolve(conflict, graph)

    assert resolved.resolution_strategy == ResolutionStrategy.SYNTHESIZE
    assert resolved.status == ConflictStatus.RESOLVED
    assert resolved.resolved_at is not None


@pytest.mark.asyncio
async def test_semantic_falls_back_to_arbitrate_on_low_confidence():
    """Synthesis confidence < threshold → falls back to ARBITRATE."""
    conflict = make_conflict(
        ConflictType.SEMANTIC,
        root_cause=RootCauseType.LOGICAL_CONTRADICTION,
        root_cause_confidence=0.8,
    )
    graph = make_graph([
        make_event("agent_a", AgentEventType.DECISION, 1, output="approve"),
        make_event("agent_b", AgentEventType.DECISION, 5, output="block"),
    ])

    mock_output = {
        "synthesized_output": "",
        "synthesizable": False,
        "confidence": 0.3,
    }
    client = make_mock_client("synthesize_outputs", mock_output)
    strategist = ResolutionStrategist(client=client)

    resolved = await strategist.resolve(conflict, graph)

    assert resolved.resolution_strategy == ResolutionStrategy.ARBITRATE
    assert resolved.status == ConflictStatus.RESOLVED


@pytest.mark.asyncio
async def test_procedural_always_arbitrates():
    """Procedural conflict → always ARBITRATE, no LLM call."""
    conflict = make_conflict(
        ConflictType.PROCEDURAL,
        root_cause=RootCauseType.TOOL_RESULT_DIVERGENCE,
        root_cause_confidence=0.9,
    )
    graph = make_graph([
        make_event("agent_a", AgentEventType.TOOL_CALL, 1),
        make_event("agent_b", AgentEventType.TOOL_CALL, 4),
    ])

    client = MagicMock()
    strategist = ResolutionStrategist(client=client)

    resolved = await strategist.resolve(conflict, graph)

    assert resolved.resolution_strategy == ResolutionStrategy.ARBITRATE
    client.chat_with_tool.assert_not_called()


@pytest.mark.asyncio
async def test_temporal_always_resets():
    """Temporal conflict → always RESET, no LLM call."""
    conflict = make_conflict(
        ConflictType.TEMPORAL,
        root_cause=RootCauseType.STALE_STATE,
        root_cause_confidence=0.9,
    )
    graph = make_graph([
        make_event("agent_a", AgentEventType.DECISION, 1,
                   context={"updated_at": "2020-01-01T00:00:00"}),
    ])

    client = MagicMock()
    strategist = ResolutionStrategist(client=client)

    resolved = await strategist.resolve(conflict, graph)

    assert resolved.resolution_strategy == ResolutionStrategy.RESET
    assert "corrected_context" in resolved.resolution_output
    client.chat_with_tool.assert_not_called()


@pytest.mark.asyncio
async def test_escalate_on_low_root_cause_confidence():
    """root_cause_confidence < 0.5 → always ESCALATE regardless of type."""
    conflict = make_conflict(
        ConflictType.SEMANTIC,
        root_cause=RootCauseType.UNKNOWN,
        root_cause_confidence=0.3,
    )
    graph = make_graph([make_event("agent_a", AgentEventType.DECISION, 1)])

    client = MagicMock()
    strategist = ResolutionStrategist(client=client)

    resolved = await strategist.resolve(conflict, graph)

    assert resolved.resolution_strategy == ResolutionStrategy.ESCALATE
    assert resolved.status == ConflictStatus.ESCALATED
    assert resolved.resolution_output["human_review_required"] is True
    client.chat_with_tool.assert_not_called()



# ConflictMemoryStore Tests


@pytest.fixture
def memory_store(tmp_path):
    path = str(tmp_path / "test_memory.json")
    store = ConflictMemoryStore(store_path=path)
    return store


@pytest.mark.asyncio
async def test_upsert_creates_pattern(memory_store):
    """First conflict → pattern created with occurrence_count=1."""
    conflict = make_conflict(
        ConflictType.SEMANTIC,
        root_cause=RootCauseType.INSTRUCTION_AMBIGUITY,
    )
    pattern = await memory_store.upsert_pattern(conflict)

    assert pattern.occurrence_count == 1
    assert pattern.conflict_type == ConflictType.SEMANTIC
    assert pattern.root_cause_type == RootCauseType.INSTRUCTION_AMBIGUITY


@pytest.mark.asyncio
async def test_upsert_accumulates_same_pattern(memory_store):
    """Same conflict type + agents → occurrence_count increments."""
    conflict = make_conflict(
        ConflictType.SEMANTIC,
        root_cause=RootCauseType.INSTRUCTION_AMBIGUITY,
        agents=["agent_a", "agent_b"],
    )

    for _ in range(3):
        await memory_store.upsert_pattern(conflict)

    patterns = await memory_store.get_all_patterns()
    assert len(patterns) == 1
    assert patterns[0].occurrence_count == 3


@pytest.mark.asyncio
async def test_recommendations_returned_above_threshold(memory_store):
    """Pattern seen >= RECOMMENDATION_THRESHOLD → recommendation returned."""
    conflict = make_conflict(
        ConflictType.PROCEDURAL,
        root_cause=RootCauseType.TOOL_RESULT_DIVERGENCE,
    )

    for _ in range(2):
        await memory_store.upsert_pattern(conflict)

    recs = await memory_store.get_recommendations()
    assert len(recs) >= 1
    assert any("agent" in r.lower() or "tool" in r.lower() for r in recs)


@pytest.mark.asyncio
async def test_no_recommendation_below_threshold(memory_store):
    """Pattern seen only once → no recommendation yet."""
    conflict = make_conflict(
        ConflictType.TEMPORAL,
        root_cause=RootCauseType.STALE_STATE,
    )
    await memory_store.upsert_pattern(conflict)

    recs = await memory_store.get_recommendations()
    assert len(recs) == 0


@pytest.mark.asyncio
async def test_different_patterns_stored_separately(memory_store):
    """Different conflict types create separate patterns."""
    c1 = make_conflict(ConflictType.SEMANTIC, root_cause=RootCauseType.INSTRUCTION_AMBIGUITY,
                       agents=["a", "b"])
    c2 = make_conflict(ConflictType.PROCEDURAL, root_cause=RootCauseType.TOOL_RESULT_DIVERGENCE,
                       agents=["a", "b"])

    await memory_store.upsert_pattern(c1)
    await memory_store.upsert_pattern(c2)

    patterns = await memory_store.get_all_patterns()
    assert len(patterns) == 2


@pytest.mark.asyncio
async def test_memory_persists_to_disk(tmp_path):
    """Patterns written to disk survive store reload."""
    path = str(tmp_path / "persist_test.json")
    store1 = ConflictMemoryStore(store_path=path)
    conflict = make_conflict(
        ConflictType.SEMANTIC,
        root_cause=RootCauseType.INSTRUCTION_AMBIGUITY,
    )
    await store1.upsert_pattern(conflict)

    # Reload from disk
    store2 = ConflictMemoryStore(store_path=path)
    patterns = await store2.get_all_patterns()

    assert len(patterns) == 1
    assert patterns[0].conflict_type == ConflictType.SEMANTIC


@pytest.mark.asyncio
async def test_fix_recommendation_is_specific(memory_store):
    """Fix recommendation references agents and scenario — not generic."""
    conflict = make_conflict(
        ConflictType.SEMANTIC,
        root_cause=RootCauseType.INSTRUCTION_AMBIGUITY,
        agents=["billing_agent", "compliance_agent"],
    )
    pattern = await memory_store.upsert_pattern(conflict)

    assert "billing_agent" in pattern.fix_recommendation or "compliance_agent" in pattern.fix_recommendation
    assert len(pattern.fix_recommendation) > 20
