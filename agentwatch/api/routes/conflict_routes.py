"""
AgentWatch — Conflict Routes
==============================
GET  /conflicts              — list conflicts with filters
GET  /conflicts/patterns     — get all conflict patterns
GET  /conflicts/recommendations — get fix recommendations
GET  /conflicts/{id}         — get single conflict detail
POST /conflicts/{id}/resolve — manually trigger resolution
POST /conflicts/scan/{run_id} — run conflict scan on a stored trace
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from agentwatch.api.dependencies import (
    get_classifier,
    get_conflict_memory,
    get_sensor,
    get_store,
    get_strategist,
)
from agentwatch.agents.conflict.conflict_memory import ConflictMemoryStore
from agentwatch.agents.conflict.conflict_sensor import ConflictSensor
from agentwatch.agents.conflict.resolution_strategist import ResolutionStrategist
from agentwatch.agents.conflict.root_cause_classifier import RootCauseClassifier
from agentwatch.core.models.schemas import (
    Conflict,
    ConflictPattern,
    ConflictStatus,
    ConflictSummary,
    ConflictType,
    ResolutionStrategy,
    RootCauseType,
)
from agentwatch.core.store.trace_store import TraceStore

_DEMO_CACHE_PATH = Path(__file__).parent.parent.parent / "demo" / "demo_cache.json"

logger = structlog.get_logger(__name__)
router = APIRouter()

# In-memory conflict store for v1 (populated by scan results)
_conflict_store: Dict[str, Conflict] = {}


@router.post("/scan/{run_id}", response_model=List[ConflictSummary])
async def scan_for_conflicts(
    run_id: str,
    store: TraceStore = Depends(get_store),
    sensor: ConflictSensor = Depends(get_sensor),
    classifier: RootCauseClassifier = Depends(get_classifier),
    memory: ConflictMemoryStore = Depends(get_conflict_memory),
) -> List[ConflictSummary]:
    """
    Run a full conflict scan on a stored trace.
    Classifies root causes and updates conflict memory.
    """
    # ── Demo mode — return cached conflicts, no LLM calls ─────────────────
    DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
    if DEMO_MODE:
        with open(_DEMO_CACHE_PATH) as _f:
            data = json.load(_f)
        agents = data.get("agents_involved", ["agent_a", "agent_b", "agent_c"])
        cs_agent   = next((a for a in agents if "customer" in a), agents[0])
        comp_agent = next((a for a in agents if "compliance" in a), agents[1 % len(agents)])
        bill_agent = next((a for a in agents if "billing" in a),   agents[2 % len(agents)])
        now = datetime.now(timezone.utc)
        logger.info("demo_mode_conflict_scan", run_id=run_id)
        return [
            ConflictSummary(
                conflict_id=str(uuid.uuid4()),
                run_id=run_id,
                conflict_type=ConflictType.SEMANTIC,
                status=ConflictStatus.RESOLVED,
                agents_involved=[cs_agent, comp_agent],
                detected_at=now,
                root_cause_type=RootCauseType.INSTRUCTION_AMBIGUITY,
                resolution_strategy=ResolutionStrategy.SYNTHESIZE,
            ),
            ConflictSummary(
                conflict_id=str(uuid.uuid4()),
                run_id=run_id,
                conflict_type=ConflictType.PROCEDURAL,
                status=ConflictStatus.RESOLVED,
                agents_involved=[cs_agent, bill_agent],
                detected_at=now,
                root_cause_type=RootCauseType.TOOL_RESULT_DIVERGENCE,
                resolution_strategy=ResolutionStrategy.ARBITRATE,
            ),
            ConflictSummary(
                conflict_id=str(uuid.uuid4()),
                run_id=run_id,
                conflict_type=ConflictType.TEMPORAL,
                status=ConflictStatus.RESOLVED,
                agents_involved=[bill_agent, cs_agent],
                detected_at=now,
                root_cause_type=RootCauseType.STALE_STATE,
                resolution_strategy=ResolutionStrategy.RESET,
            ),
        ]
    # ── Live pipeline ──────────────────────────────────────────────────────

    graph = await store.get_trace(run_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Trace {run_id} not found")

    conflicts = await sensor.scan(graph)

    if conflicts:
        conflicts = await classifier.batch_classify(conflicts, graph)
        for c in conflicts:
            _conflict_store[c.conflict_id] = c
            await memory.record(c)
            await memory.upsert_pattern(c)

    logger.info(
        "conflict_scan_complete",
        run_id=run_id,
        found=len(conflicts),
    )

    return [
        ConflictSummary(
            conflict_id=c.conflict_id,
            run_id=c.run_id,
            conflict_type=c.conflict_type,
            status=c.status,
            agents_involved=c.agents_involved,
            detected_at=c.detected_at,
            root_cause_type=c.root_cause_type,
            resolution_strategy=c.resolution_strategy,
        )
        for c in conflicts
    ]


@router.get("", response_model=List[ConflictSummary])
async def list_conflicts(
    run_id: Optional[str]  = Query(None),
    status: Optional[str]  = Query(None),
    limit:  int            = Query(default=50, ge=1, le=200),
) -> List[ConflictSummary]:
    """List conflicts with optional filters."""
    conflicts = list(_conflict_store.values())

    if run_id:
        conflicts = [c for c in conflicts if c.run_id == run_id]

    if status:
        try:
            status_enum = ConflictStatus(status)
            conflicts = [c for c in conflicts if c.status == status_enum]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    conflicts = sorted(conflicts, key=lambda c: c.detected_at, reverse=True)[:limit]

    return [
        ConflictSummary(
            conflict_id=c.conflict_id,
            run_id=c.run_id,
            conflict_type=c.conflict_type,
            status=c.status,
            agents_involved=c.agents_involved,
            detected_at=c.detected_at,
            root_cause_type=c.root_cause_type,
            resolution_strategy=c.resolution_strategy,
        )
        for c in conflicts
    ]


@router.get("/patterns", response_model=List[ConflictPattern])
async def get_patterns(
    memory: ConflictMemoryStore = Depends(get_conflict_memory),
) -> List[ConflictPattern]:
    """Return all known conflict patterns, most frequent first."""
    return await memory.get_all_patterns()


@router.get("/recommendations", response_model=List[str])
async def get_recommendations(
    memory: ConflictMemoryStore = Depends(get_conflict_memory),
) -> List[str]:
    """Return actionable fix recommendations for recurring patterns."""
    return await memory.get_recommendations()


@router.get("/{conflict_id}", response_model=Conflict)
async def get_conflict(conflict_id: str) -> Conflict:
    """Get full conflict detail including root cause and resolution."""
    conflict = _conflict_store.get(conflict_id)
    if conflict is None:
        raise HTTPException(status_code=404, detail=f"Conflict {conflict_id} not found")
    return conflict


@router.post("/{conflict_id}/resolve", response_model=Conflict)
async def resolve_conflict(
    conflict_id: str,
    store: TraceStore = Depends(get_store),
    strategist: ResolutionStrategist = Depends(get_strategist),
) -> Conflict:
    """Manually trigger the Resolution Strategist on a specific conflict."""
    conflict = _conflict_store.get(conflict_id)
    if conflict is None:
        raise HTTPException(status_code=404, detail=f"Conflict {conflict_id} not found")

    if conflict.status in (ConflictStatus.RESOLVED, ConflictStatus.ESCALATED):
        return conflict  # Already resolved — return as-is

    graph = await store.get_trace(conflict.run_id)
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail=f"Source trace {conflict.run_id} not found — cannot resolve"
        )

    resolved = await strategist.resolve(conflict, graph)
    _conflict_store[conflict_id] = resolved

    logger.info(
        "conflict_manually_resolved",
        conflict_id=conflict_id,
        strategy=resolved.resolution_strategy.value if resolved.resolution_strategy else "none",
    )

    return resolved
