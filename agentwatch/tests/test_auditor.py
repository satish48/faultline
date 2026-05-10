"""
Tests for Decision Auditor — Pillar A.
Uses mocked Anthropic client — no real API calls.
Run with: pytest agentwatch/tests/test_auditor.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch
from typing import List

import pytest

from agentwatch.core.models.schemas import (
    AgentEvent,
    AgentEventType,
    AuditStep,
    DecisionNode,
    RiskLevel,
    TraceGraph,
)
from agentwatch.agents.auditor.decision_decomposer import DecisionDecomposer
from agentwatch.agents.auditor.nl_explainer import NLExplainer


# ── Helpers ───────────────────────────────────────────

def make_event(
    agent: str,
    event_type: AgentEventType,
    seq: int,
    run_id: str = "run-test",
    context: dict = None,
    output: str = None,
    tool_name: str = None,
    confidence: float = None,
) -> AgentEvent:
    return AgentEvent(
        run_id=run_id,
        agent_name=agent,
        event_type=event_type,
        sequence_no=seq,
        input_context=context or {},
        output=output,
        tool_name=tool_name,
        metadata={"confidence": confidence} if confidence else {},
    )


def make_graph(events: List[AgentEvent], run_id: str = "run-test") -> TraceGraph:
    agents = list({e.agent_name for e in events})
    return TraceGraph(
        run_id=run_id,
        session_id="sess-test",
        agents_involved=agents,
        events=events,
    )


def make_mock_client(tool_name: str, tool_output: dict) -> MagicMock:
    """Build a mock LLMClient that returns tool_output from chat_with_tool."""
    client = MagicMock()
    client.chat_with_tool.return_value = tool_output
    return client


# ── DecisionDecomposer Tests ──────────────────────────

@pytest.mark.asyncio
async def test_decompose_extracts_decision_nodes():
    """Two DECISION events → two DecisionNodes returned."""
    run_id = "run-001"
    events = [
        make_event("router", AgentEventType.DECISION, 1, run_id, {"query": "refund"}, "escalate"),
        make_event("billing", AgentEventType.DECISION, 2, run_id, {"order_id": "o1"}, "approve"),
    ]
    graph = make_graph(events, run_id)

    mock_output = {
        "decisions": [
            {
                "event_id": events[0].event_id,
                "decision_type": "routing",
                "options_available": ["escalate", "resolve_directly"],
                "chosen_option": "escalate",
                "confidence": 0.85,
                "alternative_option": "resolve_directly",
                "alternative_outcome": "resolved without escalation",
            },
            {
                "event_id": events[1].event_id,
                "decision_type": "response_generation",
                "options_available": ["approve", "deny"],
                "chosen_option": "approve",
                "confidence": 0.9,
                "alternative_option": "deny",
                "alternative_outcome": "refund denied",
            },
        ]
    }

    client = make_mock_client("extract_decision_nodes", mock_output)
    decomposer = DecisionDecomposer(client=client)
    nodes = await decomposer.decompose(graph)

    assert len(nodes) == 2
    assert nodes[0].chosen_option == "escalate"
    assert nodes[1].chosen_option == "approve"
    assert nodes[0].sequence_no < nodes[1].sequence_no


@pytest.mark.asyncio
async def test_critical_decisions_flagged_for_handoff():
    """HANDOFF event → DecisionNode with is_critical=True."""
    run_id = "run-002"
    events = [
        make_event("agent_a", AgentEventType.HANDOFF, 1, run_id,
                   {"task": "review"}, "handoff to compliance"),
    ]
    graph = make_graph(events, run_id)

    mock_output = {
        "decisions": [{
            "event_id": events[0].event_id,
            "decision_type": "handoff",
            "options_available": ["handoff to compliance", "handle internally"],
            "chosen_option": "handoff to compliance",
            "confidence": 0.95,
            "alternative_option": "handle internally",
            "alternative_outcome": None,
        }]
    }

    client = make_mock_client("extract_decision_nodes", mock_output)
    decomposer = DecisionDecomposer(client=client)
    nodes = await decomposer.decompose(graph)

    assert len(nodes) == 1
    assert nodes[0].is_critical is True


@pytest.mark.asyncio
async def test_critical_flag_for_low_confidence():
    """DECISION with confidence < 0.6 → is_critical=True."""
    run_id = "run-003"
    events = [
        make_event("classifier", AgentEventType.DECISION, 1, run_id,
                   {"input": "ambiguous query"}, "maybe_escalate"),
    ]
    graph = make_graph(events, run_id)

    mock_output = {
        "decisions": [{
            "event_id": events[0].event_id,
            "decision_type": "routing",
            "options_available": ["maybe_escalate", "resolve"],
            "chosen_option": "maybe_escalate",
            "confidence": 0.45,  # low confidence → critical
            "alternative_option": "resolve",
            "alternative_outcome": None,
        }]
    }

    client = make_mock_client("extract_decision_nodes", mock_output)
    decomposer = DecisionDecomposer(client=client)
    nodes = await decomposer.decompose(graph)

    assert nodes[0].is_critical is True


@pytest.mark.asyncio
async def test_tool_call_is_critical():
    """TOOL_CALL events are always critical."""
    run_id = "run-004"
    events = [
        make_event("billing", AgentEventType.TOOL_CALL, 1, run_id,
                   {"amount": 150}, tool_name="process_refund"),
    ]
    graph = make_graph(events, run_id)

    mock_output = {
        "decisions": [{
            "event_id": events[0].event_id,
            "decision_type": "tool_selection",
            "options_available": ["process_refund", "lookup_order"],
            "chosen_option": "process_refund",
            "confidence": 0.99,
            "alternative_option": None,
            "alternative_outcome": None,
        }]
    }

    client = make_mock_client("extract_decision_nodes", mock_output)
    decomposer = DecisionDecomposer(client=client)
    nodes = await decomposer.decompose(graph)

    assert nodes[0].is_critical is True


@pytest.mark.asyncio
async def test_empty_graph_returns_empty_nodes():
    """Graph with no decomposable events returns empty list."""
    events = [
        make_event("agent_a", AgentEventType.START, 1),
        make_event("agent_a", AgentEventType.END, 2),
    ]
    graph = make_graph(events)

    client = MagicMock()
    decomposer = DecisionDecomposer(client=client)
    nodes = await decomposer.decompose(graph)

    assert nodes == []
    client.chat_with_tool.assert_not_called()


@pytest.mark.asyncio
async def test_source_event_ids_populated():
    """Every DecisionNode must have source_event_ids linked to real events."""
    run_id = "run-005"
    events = [
        make_event("agent_a", AgentEventType.DECISION, 1, run_id, {}, "approve"),
    ]
    graph = make_graph(events, run_id)

    mock_output = {
        "decisions": [{
            "event_id": events[0].event_id,
            "decision_type": "response_generation",
            "options_available": ["approve", "deny"],
            "chosen_option": "approve",
            "confidence": 0.8,
            "alternative_option": "deny",
            "alternative_outcome": None,
        }]
    }

    client = make_mock_client("extract_decision_nodes", mock_output)
    decomposer = DecisionDecomposer(client=client)
    nodes = await decomposer.decompose(graph)

    assert nodes[0].source_event_ids == [events[0].event_id]


# ── NLExplainer Tests ─────────────────────────────────

def make_decision_node(
    agent: str,
    seq: int,
    run_id: str = "run-test",
    is_critical: bool = False,
    confidence: float = 0.9,
    source_event_id: str = None,
) -> DecisionNode:
    return DecisionNode(
        run_id=run_id,
        agent_name=agent,
        sequence_no=seq,
        context_snapshot={"order_id": "o1", "amount": 150},
        decision_type="routing",
        options_available=["approve", "deny"],
        chosen_option="approve",
        confidence=confidence,
        alternative_option="deny",
        is_critical=is_critical,
        source_event_ids=[source_event_id or str(uuid.uuid4())],
    )


@pytest.mark.asyncio
async def test_explainer_produces_audit_steps():
    """One DecisionNode → one AuditStep returned."""
    node = make_decision_node("billing", 1, is_critical=True)
    graph = make_graph([], "run-test")

    node_id = node.node_id
    mock_output = {
        "steps": [{
            "node_id": node_id,
            "action": "approved customer refund",
            "reasoning": "based on context key amount=150 which exceeded refund threshold",
            "outcome": "refund of $150 approved",
            "risk_level": "medium",
        }]
    }

    client = make_mock_client("generate_audit_steps", mock_output)
    explainer = NLExplainer(client=client)
    steps = await explainer.explain(graph, [node])

    assert len(steps) == 1
    assert steps[0].action == "approved customer refund"
    assert steps[0].risk_level == RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_explainer_reasoning_is_grounded():
    """reasoning field must reference a context key or event."""
    node = make_decision_node("router", 1)
    graph = make_graph([], "run-test")

    mock_output = {
        "steps": [{
            "node_id": node.node_id,
            "action": "routed to compliance agent",
            "reasoning": "based on context key order_amount > 100 which triggered compliance review",
            "outcome": "handed off to compliance",
            "risk_level": "medium",
        }]
    }

    client = make_mock_client("generate_audit_steps", mock_output)
    explainer = NLExplainer(client=client)
    steps = await explainer.explain(graph, [node])

    grounding_kws = ["context", "event", "based on", "because", "due to", "value", "input"]
    assert any(kw in steps[0].reasoning.lower() for kw in grounding_kws), (
        f"Reasoning not grounded: {steps[0].reasoning}"
    )


@pytest.mark.asyncio
async def test_explainer_source_event_ids_preserved():
    """AuditStep must inherit source_event_ids from its DecisionNode."""
    event_id = str(uuid.uuid4())
    node = make_decision_node("agent_a", 1, source_event_id=event_id)
    graph = make_graph([], "run-test")

    mock_output = {
        "steps": [{
            "node_id": node.node_id,
            "action": "made decision",
            "reasoning": "based on context key user_id",
            "outcome": "decision recorded",
            "risk_level": "low",
        }]
    }

    client = make_mock_client("generate_audit_steps", mock_output)
    explainer = NLExplainer(client=client)
    steps = await explainer.explain(graph, [node])

    assert event_id in steps[0].source_event_ids


@pytest.mark.asyncio
async def test_explainer_risk_level_high_for_critical_low_confidence():
    """Critical + low confidence → HIGH risk step."""
    node = make_decision_node("agent_a", 1, is_critical=True, confidence=0.4)
    graph = make_graph([], "run-test")

    mock_output = {
        "steps": [{
            "node_id": node.node_id,
            "action": "escalated with low confidence",
            "reasoning": "based on context key ambiguity_score=0.8",
            "outcome": "escalated to human",
            "risk_level": "high",
        }]
    }

    client = make_mock_client("generate_audit_steps", mock_output)
    explainer = NLExplainer(client=client)
    steps = await explainer.explain(graph, [node])

    assert steps[0].risk_level == RiskLevel.HIGH


@pytest.mark.asyncio
async def test_explainer_empty_nodes_returns_empty():
    """No nodes → no steps, no LLM call."""
    graph = make_graph([], "run-test")
    client = MagicMock()
    explainer = NLExplainer(client=client)
    steps = await explainer.explain(graph, [])

    assert steps == []
    client.chat_with_tool.assert_not_called()


@pytest.mark.asyncio
async def test_explainer_step_ordering():
    """Steps returned in sequence order matching nodes."""
    nodes = [
        make_decision_node("agent_a", 1),
        make_decision_node("agent_b", 2),
        make_decision_node("agent_c", 3),
    ]
    graph = make_graph([], "run-test")

    mock_output = {
        "steps": [
            {"node_id": nodes[0].node_id, "action": "step one", "reasoning": "based on context key x", "outcome": "done", "risk_level": "low"},
            {"node_id": nodes[1].node_id, "action": "step two", "reasoning": "based on context key y", "outcome": "done", "risk_level": "low"},
            {"node_id": nodes[2].node_id, "action": "step three", "reasoning": "based on context key z", "outcome": "done", "risk_level": "low"},
        ]
    }

    client = make_mock_client("generate_audit_steps", mock_output)
    explainer = NLExplainer(client=client)
    steps = await explainer.explain(graph, nodes)

    assert [s.step_no for s in steps] == [1, 2, 3]
    assert [s.action for s in steps] == ["step one", "step two", "step three"]


# ══════════════════════════════════════════════════════
# CounterfactualExplorer Tests
# ══════════════════════════════════════════════════════

from agentwatch.agents.auditor.counterfactual_explorer import CounterfactualExplorer
from agentwatch.agents.auditor.compliance_reporter import ComplianceReporter


@pytest.mark.asyncio
async def test_counterfactual_outcome_changes():
    """Modified context that flips decision → outcome_changed=True."""
    node = make_decision_node("billing", 1, confidence=0.85)
    node.context_snapshot = {"amount": 50, "customer_tier": "standard"}
    node.chosen_option = "approve"

    graph = make_graph([], "run-cf")

    mock_output = {
        "chosen_option": "deny",
        "reasoning": "based on context key amount=200 which exceeds threshold",
        "outcome_changed": True,
        "confidence": 0.9,
    }
    client = make_mock_client("simulate_counterfactual", mock_output)
    explorer = CounterfactualExplorer(client=client)

    result = await explorer.explore(
        graph, node, {"amount": 200, "customer_tier": "standard"}
    )

    assert result.outcome_changed is True
    assert result.original_outcome == "approve"
    assert result.counterfactual_outcome == "deny"
    assert result.divergence_point == "amount"


@pytest.mark.asyncio
async def test_counterfactual_outcome_unchanged():
    """Non-material context change → outcome_changed=False."""
    node = make_decision_node("router", 1, confidence=0.95)
    node.context_snapshot = {"user_id": "u1", "priority": "normal"}
    node.chosen_option = "resolve_directly"

    graph = make_graph([], "run-cf2")

    mock_output = {
        "chosen_option": "resolve_directly",
        "reasoning": "based on context key priority=normal, decision unchanged",
        "outcome_changed": False,
        "confidence": 0.93,
    }
    client = make_mock_client("simulate_counterfactual", mock_output)
    explorer = CounterfactualExplorer(client=client)

    result = await explorer.explore(
        graph, node, {"user_id": "u1", "priority": "high"}
    )

    assert result.outcome_changed is False
    assert result.sensitivity_score < 0.5


@pytest.mark.asyncio
async def test_sensitivity_score_high_when_outcome_changes():
    """outcome_changed=True → sensitivity_score >= 0.6."""
    from agentwatch.agents.auditor.counterfactual_explorer import CounterfactualExplorer
    score = CounterfactualExplorer._sensitivity_score(
        original_confidence=0.9,
        cf_confidence=0.3,
        outcome_changed=True,
    )
    assert score >= 0.6


@pytest.mark.asyncio
async def test_sensitivity_score_low_when_unchanged():
    """outcome_changed=False → sensitivity_score < 0.3."""
    score = CounterfactualExplorer._sensitivity_score(
        original_confidence=0.9,
        cf_confidence=0.88,
        outcome_changed=False,
    )
    assert score < 0.3


@pytest.mark.asyncio
async def test_divergence_point_single_key():
    """One changed key → divergence_point = that key name."""
    point = CounterfactualExplorer._find_divergence_point(
        {"amount": 50, "tier": "standard"},
        {"amount": 200, "tier": "standard"},
    )
    assert point == "amount"


@pytest.mark.asyncio
async def test_divergence_point_multiple_keys():
    """Two changed keys → divergence_point starts with 'multiple_keys'."""
    point = CounterfactualExplorer._find_divergence_point(
        {"amount": 50, "tier": "standard"},
        {"amount": 200, "tier": "premium"},
    )
    assert "multiple_keys" in point


@pytest.mark.asyncio
async def test_auto_modify_flips_boolean():
    """Auto-modify picks boolean key and flips it."""
    ctx = {"eligible": True, "amount": 100}
    modified = CounterfactualExplorer._auto_modify(ctx)
    assert modified["eligible"] is False


@pytest.mark.asyncio
async def test_auto_modify_doubles_threshold_numeric():
    """Auto-modify doubles numeric threshold key."""
    ctx = {"risk_score": 0.4, "name": "alice"}
    modified = CounterfactualExplorer._auto_modify(ctx)
    assert modified["risk_score"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_find_sensitive_decisions_returns_changed_nodes():
    """find_sensitive_decisions returns only nodes where outcome changed."""
    nodes = [
        make_decision_node("agent_a", 1),
        make_decision_node("agent_b", 2),
    ]
    nodes[0].context_snapshot = {"amount": 50}
    nodes[1].context_snapshot = {"tier": "standard"}

    graph = make_graph([], "run-sens")

    # First node: outcome changes; second: does not
    call_count = [0]
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        changed = call_count[0] == 1
        return {
            "chosen_option": "deny" if changed else "approve",
            "reasoning": "based on context key amount",
            "outcome_changed": changed,
            "confidence": 0.8,
        }

    client = MagicMock()
    client.chat_with_tool.side_effect = side_effect

    explorer = CounterfactualExplorer(client=client)
    sensitive = await explorer.find_sensitive_decisions(graph, nodes)

    assert len(sensitive) == 1
    assert sensitive[0].agent_name == "agent_a"


# ══════════════════════════════════════════════════════
# ComplianceReporter Tests
# ══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_compliance_report_generated():
    """Full report generated with all required fields."""
    node = make_decision_node("billing", 1, is_critical=True, confidence=0.4)
    step = AuditStep(
        step_no=1, agent_name="billing",
        action="approved refund",
        reasoning="based on context key amount=150",
        outcome="refund approved",
        risk_level=RiskLevel.HIGH,
        source_event_ids=[str(uuid.uuid4())],
    )
    graph = make_graph([], "run-rep")

    mock_output = {
        "summary": "The agent run completed with high risk. One critical decision was made by billing agent. Counterfactual analysis shows the decision is sensitive to amount changes.",
        "key_finding": "Billing agent approved $150 refund with low confidence of 0.4.",
        "risk_factors": ["Low confidence critical decision", "Amount threshold sensitivity"],
    }
    client = make_mock_client("generate_audit_summary", mock_output)
    reporter = ComplianceReporter(client=client)

    report = await reporter.generate_report(graph, [step], [node], [])

    assert report.run_id == "run-rep"
    assert report.overall_risk == RiskLevel.HIGH
    assert len(report.steps) == 1
    assert report.compliance_data["total_decisions"] == 1
    assert report.compliance_data["high_risk_steps_count"] == 1


@pytest.mark.asyncio
async def test_overall_risk_escalation():
    """HIGH step in list → overall_risk=HIGH regardless of initial value."""
    steps = [
        AuditStep(step_no=1, agent_name="a", action="act", reasoning="based on context key x",
                  outcome="done", risk_level=RiskLevel.HIGH, source_event_ids=[]),
        AuditStep(step_no=2, agent_name="b", action="act", reasoning="based on context key y",
                  outcome="done", risk_level=RiskLevel.LOW, source_event_ids=[]),
    ]
    graph = make_graph([], "run-esc")

    mock_output = {
        "summary": "High risk run detected.", "key_finding": "One high risk step.",
        "risk_factors": ["High risk action by agent a"],
    }
    client = make_mock_client("generate_audit_summary", mock_output)
    reporter = ComplianceReporter(client=client)

    report = await reporter.generate_report(graph, steps, [], [])
    assert report.overall_risk == RiskLevel.HIGH


@pytest.mark.asyncio
async def test_compliance_data_fields_populated():
    """compliance_data dict always has all quantitative fields."""
    graph = make_graph([], "run-cd")
    graph.total_cost_usd = 0.042
    graph.total_latency_ms = 3200.0
    graph.total_tokens = 1400

    mock_output = {
        "summary": "Low risk run.", "key_finding": "No issues found.",
        "risk_factors": [],
    }
    client = make_mock_client("generate_audit_summary", mock_output)
    reporter = ComplianceReporter(client=client)

    report = await reporter.generate_report(graph, [], [], [])

    cd = report.compliance_data
    assert cd["total_cost_usd"] == pytest.approx(0.042)
    assert cd["total_latency_ms"] == pytest.approx(3200.0)
    assert cd["total_tokens"] == 1400
    assert "generated_at" in cd


@pytest.mark.asyncio
async def test_to_markdown_contains_key_sections():
    """to_markdown output contains all required sections."""
    step = AuditStep(
        step_no=1, agent_name="router",
        action="routed request",
        reasoning="based on context key query_type",
        outcome="routed to billing",
        risk_level=RiskLevel.LOW,
        source_event_ids=[],
    )
    graph = make_graph([], "run-md")

    mock_output = {
        "summary": "Normal run.", "key_finding": "No issues.",
        "risk_factors": ["None"],
    }
    client = make_mock_client("generate_audit_summary", mock_output)
    reporter = ComplianceReporter(client=client)
    report = await reporter.generate_report(graph, [step], [], [])

    md = await reporter.to_markdown(report)

    assert "# AgentWatch Audit Report" in md
    assert "Executive Summary" in md
    assert "Audit Trail" in md
    assert "Compliance Data" in md
    assert "routed request" in md