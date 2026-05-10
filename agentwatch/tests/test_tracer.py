"""
Tests for ExecutionTracer and TraceStore.
Run with: pytest agentwatch/tests/test_tracer.py -v
"""

import asyncio
import pytest
from agentwatch.core.models.schemas import AgentEventType, TraceGraph
from agentwatch.core.tracer.execution_tracer import ExecutionTracer, _state_hash
from agentwatch.core.store.trace_store import TraceStore


# ── Fixtures ──────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    return TraceStore(database_url=f"sqlite+aiosqlite:///{db}")


@pytest.fixture
def tracer(store):
    return ExecutionTracer(
        session_id="test-session",
        store=store,
        cost_per_1k_tokens=0.003,
        loop_detection_threshold=3,
    )


# ── Tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_and_end_run(tracer, store):
    """Full run: start → emit 3 events → end → TraceGraph has 3 events."""
    await store.init_db()
    run_id = await tracer.start_run()
    assert run_id is not None

    for i, etype in enumerate([
        AgentEventType.DECISION,
        AgentEventType.TOOL_CALL,
        AgentEventType.HANDOFF,
    ], start=1):
        async with tracer.trace_step("agent_a", etype, {"step": i}) as ev:
            ev.output = f"result_{i}"
            ev.tokens_used = 100

    graph = await tracer.end_run(success=True)

    assert isinstance(graph, TraceGraph)
    assert len(graph.events) == 3
    assert graph.success is True
    assert graph.total_tokens == 300
    assert graph.total_cost_usd == pytest.approx(0.0003 * 3)  # 100 tokens * 0.003/1k = 0.0003 each


@pytest.mark.asyncio
async def test_loop_detection(tracer, store):
    """Same state hash 3 times → loop flagged on third event."""
    await store.init_db()
    await tracer.start_run()

    repeated_ctx = {"user_id": "u1", "action": "retry"}

    loop_events = []
    for _ in range(3):
        async with tracer.trace_step("agent_a", AgentEventType.DECISION, repeated_ctx) as ev:
            ev.output = "retrying"
            loop_events.append(ev)

    await tracer.end_run()

    # First two: no loop
    assert not loop_events[0].metadata.get("loop_detected")
    assert not loop_events[1].metadata.get("loop_detected")
    # Third: loop detected
    assert loop_events[2].metadata.get("loop_detected") is True


@pytest.mark.asyncio
async def test_cost_tracking(tracer, store):
    """tokens_used=1000 at 0.003/1k → cost_usd=0.003."""
    await store.init_db()
    await tracer.start_run()

    async with tracer.trace_step("agent_a", AgentEventType.LLM_CALL, {}) as ev:
        ev.tokens_used = 1000

    graph = await tracer.end_run()

    assert graph.events[0].cost_usd == pytest.approx(0.003)
    assert graph.total_cost_usd == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_handoff_creates_cross_edge(tracer, store):
    """Handoff from agent_a to agent_b creates a direct edge."""
    await store.init_db()
    await tracer.start_run()

    handoff_ev = await tracer.record_handoff(
        from_agent="agent_a",
        to_agent="agent_b",
        payload={"task": "review"},
    )

    async with tracer.trace_step("agent_b", AgentEventType.DECISION, {}) as b_ev:
        b_ev.output = "reviewed"

    graph = await tracer.end_run()

    # Sequential edge: handoff → b_ev
    assert b_ev.event_id in graph.edges.get(handoff_ev.event_id, [])


@pytest.mark.asyncio
async def test_trace_store_save_and_retrieve(tracer, store):
    """Save a trace then retrieve by run_id — full round-trip."""
    await store.init_db()
    run_id = await tracer.start_run()

    async with tracer.trace_step("agent_a", AgentEventType.DECISION, {"x": 1}) as ev:
        ev.output = "decided"

    graph = await tracer.end_run()

    retrieved = await store.get_trace(run_id)
    assert retrieved is not None
    assert retrieved.run_id == run_id
    assert len(retrieved.events) == 1


@pytest.mark.asyncio
async def test_trace_store_returns_none_for_missing(store):
    """get_trace on unknown run_id returns None, not an exception."""
    await store.init_db()
    result = await store.get_trace("does-not-exist")
    assert result is None


@pytest.mark.asyncio
async def test_trace_store_delete(tracer, store):
    """Delete a trace — gone from store."""
    await store.init_db()
    run_id = await tracer.start_run()
    async with tracer.trace_step("agent_a", AgentEventType.END, {}) as ev:
        ev.output = "done"
    await tracer.end_run()

    deleted = await store.delete_trace(run_id)
    assert deleted is True
    assert await store.get_trace(run_id) is None


@pytest.mark.asyncio
async def test_error_in_step_captured(tracer, store):
    """Exception inside trace_step → ERROR event recorded, exception re-raised."""
    await store.init_db()
    await tracer.start_run()

    with pytest.raises(ValueError):
        async with tracer.trace_step("agent_a", AgentEventType.LLM_CALL, {}) as ev:
            raise ValueError("LLM timeout")

    graph = await tracer.end_run(success=False, error_message="LLM timeout")

    error_events = [e for e in graph.events if e.event_type == AgentEventType.ERROR]
    assert len(error_events) == 1
    assert "LLM timeout" in error_events[0].metadata.get("error", "")


@pytest.mark.asyncio
async def test_state_hash_determinism():
    """Same state always produces same hash."""
    state = {"user": "alice", "action": "refund", "amount": 100}
    h1 = _state_hash(state)
    h2 = _state_hash(state)
    assert h1 == h2
    assert len(h1) == 16


@pytest.mark.asyncio
async def test_list_traces(tracer, store):
    """list_traces returns all runs for a session."""
    await store.init_db()

    for _ in range(3):
        await tracer.start_run()
        async with tracer.trace_step("agent_a", AgentEventType.END, {}) as ev:
            ev.output = "done"
        await tracer.end_run()

    traces = await store.list_traces("test-session")
    assert len(traces) == 3
