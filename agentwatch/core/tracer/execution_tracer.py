"""
AgentWatch — Execution Tracer
==============================
Non-invasive instrumentation layer for any multi-agent system.

Wraps agent functions via decorator or context manager — zero changes
to the agent's own code required.

Captures per-event:
  - Latency (time.monotonic)
  - Token usage + cost (caller sets event.tokens_used)
  - State hash (for Loop Killer)
  - Full input context and output

Emits a fully structured TraceGraph on run completion.
Both Pillar A (Decision Auditor) and Pillar B (Conflict Detector) query this graph.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import structlog

from agentwatch.core.models.schemas import (
    AgentEvent,
    AgentEventType,
    LoopDetectionResult,
    TraceGraph,
)
from agentwatch.core.store.trace_store import TraceStore

logger = structlog.get_logger(__name__)


def _state_hash(state: Any) -> str:
    """
    Deterministic 16-char SHA256 prefix of any state object.
    Used by Loop Killer: if the same hash appears >= threshold times
    in one run, the agents are looping.
    """
    try:
        raw = json.dumps(state, sort_keys=True, default=str)
    except (TypeError, ValueError):
        raw = str(state)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ExecutionTracer:
    """
    Instruments a multi-agent run from start to finish.

    One tracer instance = one session (e.g. one user conversation).
    Each call to start_run() begins a new run within that session.

    Thread safety: asyncio.Lock guards all event list mutations.
    Multiple concurrent agents writing events is safe.
    """

    def __init__(
        self,
        session_id: str,
        store: Optional[TraceStore] = None,
        cost_per_1k_tokens: float = 0.003,
        loop_detection_threshold: int = 3,
    ):
        self.session_id = session_id
        self.store = store or TraceStore()
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.loop_detection_threshold = loop_detection_threshold

        self._run_id: Optional[str] = None
        self._events: List[AgentEvent] = []
        self._seq: int = 0
        self._hash_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._run_start: Optional[float] = None

    # ── Public API ──

    async def start_run(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Begin a new traced run. Returns the run_id."""
        async with self._lock:
            self._run_id = str(uuid.uuid4())
            self._events = []
            self._seq = 0
            self._hash_counts = {}
            self._run_start = time.monotonic()

        logger.info("run_started", run_id=self._run_id, session=self.session_id)
        return self._run_id

    async def end_run(
        self,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> TraceGraph:
        """
        Finalize the run, build TraceGraph, persist to store, return it.
        Clears internal state so the tracer is ready for the next run.
        """
        if not self._run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        elapsed_ms = (time.monotonic() - self._run_start) * 1000 if self._run_start else None

        async with self._lock:
            events_snapshot = list(self._events)

        total_cost    = sum(e.cost_usd    for e in events_snapshot if e.cost_usd    is not None)
        total_tokens  = sum(e.tokens_used for e in events_snapshot if e.tokens_used is not None)
        agents        = list({e.agent_name for e in events_snapshot})

        graph = TraceGraph(
            run_id=self._run_id,
            session_id=self.session_id,
            completed_at=datetime.utcnow(),
            agents_involved=agents,
            events=events_snapshot,
            edges=self._build_edges(events_snapshot),
            total_latency_ms=elapsed_ms,
            total_cost_usd=total_cost if total_cost else None,
            total_tokens=total_tokens if total_tokens else None,
            success=success,
            error_message=error_message,
            seen_state_hashes=list(self._hash_counts.keys()),
        )

        await self.store.save_trace(graph)

        logger.info(
            "run_ended",
            run_id=self._run_id,
            events=len(events_snapshot),
            cost_usd=total_cost,
            latency_ms=elapsed_ms,
            success=success,
        )

        # Reset for next run
        self._run_id = None
        return graph

    @asynccontextmanager
    async def trace_step(
        self,
        agent_name: str,
        event_type: AgentEventType,
        input_context: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Context manager for a single agent step.

        Caller can set event.output, event.tokens_used inside the block.
        Latency and cost are computed automatically on exit.

        Usage:
            async with tracer.trace_step("router", AgentEventType.DECISION, ctx) as ev:
                ev.output = await agent.decide(ctx)
                ev.tokens_used = 200
        """
        if not self._run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        ctx = input_context or {}
        s_hash = _state_hash(ctx)
        loop_result = self._check_loop(s_hash)

        async with self._lock:
            seq = self._next_seq()

        event = AgentEvent(
            run_id=self._run_id,
            agent_name=agent_name,
            event_type=event_type,
            sequence_no=seq,
            input_context=ctx,
            tool_name=tool_name,
            tool_args=tool_args,
            state_hash=s_hash,
        )

        t0 = time.monotonic()
        error_occurred = False

        try:
            yield event

        except Exception as exc:
            error_occurred = True
            event.event_type = AgentEventType.ERROR
            event.metadata["error"] = str(exc)
            event.metadata["error_type"] = type(exc).__name__
            logger.error("step_error", agent=agent_name, error=str(exc))
            raise

        finally:
            event.latency_ms = (time.monotonic() - t0) * 1000
            if event.tokens_used:
                event.cost_usd = (event.tokens_used / 1000) * self.cost_per_1k_tokens

            if loop_result.loop_detected:
                event.metadata["loop_detected"] = True
                event.metadata["loop_hash"] = s_hash
                event.metadata["loop_count"] = self._hash_counts.get(s_hash, 0)
                logger.warning(
                    "loop_detected",
                    run_id=self._run_id,
                    agent=agent_name,
                    hash=s_hash,
                    count=self._hash_counts.get(s_hash),
                )

            async with self._lock:
                self._events.append(event)

    def trace_agent(self, agent_name: str) -> Callable:
        """
        Decorator: wraps an entire async agent function.
        Emits START and END (or ERROR) events automatically.

        Usage:
            @tracer.trace_agent("my_agent")
            async def my_agent(state: dict) -> dict:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self.trace_step(
                    agent_name=agent_name,
                    event_type=AgentEventType.START,
                    input_context={"kwargs": list(kwargs.keys())},
                ) as start_ev:
                    start_ev.output = "started"

                try:
                    result = await fn(*args, **kwargs)

                    async with self.trace_step(
                        agent_name=agent_name,
                        event_type=AgentEventType.END,
                    ) as end_ev:
                        end_ev.output = "completed"

                    return result

                except Exception as exc:
                    async with self.trace_step(
                        agent_name=agent_name,
                        event_type=AgentEventType.ERROR,
                        input_context={"error": str(exc)},
                    ) as err_ev:
                        err_ev.output = str(exc)
                    raise

            return wrapper
        return decorator

    async def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        payload: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        """
        Explicitly record an agent-to-agent handoff.
        These are the seams where ConflictSensor looks for conflicts.
        """
        recorded_event: Optional[AgentEvent] = None

        async with self.trace_step(
            agent_name=from_agent,
            event_type=AgentEventType.HANDOFF,
            input_context=context or {},
        ) as ev:
            ev.output = {"to_agent": to_agent, "payload": payload}
            ev.metadata["to_agent"] = to_agent
            recorded_event = ev

        return recorded_event  # type: ignore[return-value]

    def get_run_id(self) -> Optional[str]:
        return self._run_id

    # ── Internal ───
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _check_loop(self, state_hash: str) -> LoopDetectionResult:
        """
        Increment hash counter. If count >= threshold, a loop is detected.
        The Loop Killer uses this result attached to the event metadata.
        """
        self._hash_counts[state_hash] = self._hash_counts.get(state_hash, 0) + 1
        count = self._hash_counts[state_hash]

        if count >= self.loop_detection_threshold:
            return LoopDetectionResult(
                run_id=self._run_id or "unknown",
                loop_detected=True,
                repeated_state_hash=state_hash,
                loop_length=count,
                forced_resolution=(
                    f"State '{state_hash}' seen {count}x. "
                    "Forcing escalation to prevent budget exhaustion."
                ),
            )

        return LoopDetectionResult(
            run_id=self._run_id or "unknown",
            loop_detected=False,
        )

    @staticmethod
    def _build_edges(events: List[AgentEvent]) -> Dict[str, List[str]]:
        """
        Build the edge graph from the ordered event list.
        Sequential events: A → B.
        Handoffs: also add a direct edge to the next event of the target agent.
        """
        edges: Dict[str, List[str]] = {}

        for i, ev in enumerate(events[:-1]):
            nxt = events[i + 1]
            edges.setdefault(ev.event_id, []).append(nxt.event_id)

            if ev.event_type == AgentEventType.HANDOFF:
                to_agent = ev.metadata.get("to_agent")
                if to_agent:
                    for future_ev in events[i + 1:]:
                        if future_ev.agent_name == to_agent:
                            edges.setdefault(ev.event_id, []).append(future_ev.event_id)
                            break

        return edges