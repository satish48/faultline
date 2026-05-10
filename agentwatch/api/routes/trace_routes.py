"""
AgentWatch — Trace Routes
===========================
POST /trace      — submit a new agent run trace
GET  /trace/{id} — retrieve a stored trace
GET  /traces     — list traces for a session
DELETE /trace/{id} — delete a trace
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from agentwatch.api.dependencies import get_conflict_memory, get_sensor, get_store
from agentwatch.agents.conflict.conflict_memory import ConflictMemoryStore
from agentwatch.agents.conflict.conflict_sensor import ConflictSensor
from agentwatch.core.models.schemas import (
    TraceGraph,
    TraceRequest,
    TraceResponse,
)
from agentwatch.core.store.trace_store import TraceStore

logger = structlog.get_logger(__name__)
router = APIRouter()


async def _background_conflict_scan(
    graph: TraceGraph,
    sensor: ConflictSensor,
    memory: ConflictMemoryStore,
) -> None:
    """
    Runs conflict scan in background after trace is stored.
    Does not block the POST /trace response.
    """
    try:
        conflicts = await sensor.scan(graph)
        for conflict in conflicts:
            await memory.record(conflict)
        if conflicts:
            logger.info(
                "background_conflicts_found",
                run_id=graph.run_id,
                count=len(conflicts),
            )
    except Exception as e:
        logger.error("background_conflict_scan_failed", error=str(e))


@router.post("", response_model=TraceResponse, status_code=201)
async def submit_trace(
    request: Request,
    body: TraceRequest,
    background_tasks: BackgroundTasks,
    store: TraceStore = Depends(get_store),
    sensor: ConflictSensor = Depends(get_sensor),
    memory: ConflictMemoryStore = Depends(get_conflict_memory),
) -> TraceResponse:
    """
    Submit a completed agent run trace for storage and analysis.
    Conflict scan runs in background — response is immediate.
    """
    run_id = body.agent_events[0].run_id

    graph = TraceGraph(
        run_id=run_id,
        session_id=body.session_id,
        agents_involved=list({e.agent_name for e in body.agent_events}),
        events=body.agent_events,
        metadata=body.metadata if hasattr(body, 'metadata') else {},
    )

    await store.save_trace(graph)

    # Non-blocking background conflict scan
    background_tasks.add_task(_background_conflict_scan, graph, sensor, memory)

    logger.info("trace_submitted", run_id=run_id, events=len(body.agent_events))

    return TraceResponse(
        run_id=run_id,
        session_id=body.session_id,
        status="stored",
        event_count=len(body.agent_events),
    )


@router.get("/{run_id}", response_model=TraceGraph)
async def get_trace(
    run_id: str,
    store: TraceStore = Depends(get_store),
) -> TraceGraph:
    """Retrieve a stored TraceGraph by run_id."""
    graph = await store.get_trace(run_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Trace {run_id} not found")
    return graph


@router.get("", response_model=List[TraceGraph])
async def list_traces(
    session_id: str = Query(..., description="Session ID to list traces for"),
    limit: int = Query(default=50, ge=1, le=200),
    store: TraceStore = Depends(get_store),
) -> List[TraceGraph]:
    """List all traces for a session, most recent first."""
    return await store.list_traces(session_id=session_id, limit=limit)


@router.delete("/{run_id}", status_code=204)
async def delete_trace(
    run_id: str,
    store: TraceStore = Depends(get_store),
) -> None:
    """Delete a trace. Returns 404 if not found."""
    deleted = await store.delete_trace(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Trace {run_id} not found")
