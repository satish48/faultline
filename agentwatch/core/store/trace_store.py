"""
AgentWatch — Trace Store
=========================
Async SQLite-backed persistence for TraceGraph objects.
Uses SQLAlchemy 2.0 async + aiosqlite.

Design decisions:
- Full graph stored as JSON blob — avoids complex relational schema for a v1
- Indexed on run_id (PK), session_id, and created_at for fast dashboard queries
- Async throughout — never blocks the event loop
- init_db() is idempotent — safe to call on every startup
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

import structlog
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, Index
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import select, delete

from agentwatch.core.models.schemas import TraceGraph

logger = structlog.get_logger(__name__)


# ── ORM 

class Base(DeclarativeBase):
    pass


class TraceRecord(Base):
    __tablename__ = "traces"

    run_id         = Column(String, primary_key=True)
    session_id     = Column(String, nullable=False, index=True)
    created_at     = Column(DateTime, nullable=False, index=True)
    completed_at   = Column(DateTime, nullable=True)
    success        = Column(Boolean, nullable=False, default=True)
    total_cost_usd = Column(Float, nullable=True)
    total_latency_ms = Column(Float, nullable=True)
    total_tokens   = Column(Integer, nullable=True)
    agent_count    = Column(Integer, nullable=True)
    event_count    = Column(Integer, nullable=True)
    graph_json     = Column(Text, nullable=False)   # full TraceGraph serialized


# ── Store ─────────────────────────────────────────────

class TraceStore:
    """
    Async persistence layer for TraceGraph objects.
    One instance shared across the app lifetime.
    """

    def __init__(self, database_url: str = "sqlite+aiosqlite:///agentwatch.db"):
        self._engine = create_async_engine(
            database_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        self._initialized = False

    async def init_db(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._initialized = True
        logger.info("trace_store_initialized")

    async def save_trace(self, graph: TraceGraph) -> None:
        """Persist a TraceGraph. Upserts on run_id."""
        record = TraceRecord(
            run_id=graph.run_id,
            session_id=graph.session_id,
            created_at=graph.created_at,
            completed_at=graph.completed_at,
            success=graph.success,
            total_cost_usd=graph.total_cost_usd,
            total_latency_ms=graph.total_latency_ms,
            total_tokens=graph.total_tokens,
            agent_count=len(graph.agents_involved),
            event_count=len(graph.events),
            graph_json=graph.model_dump_json(),
        )
        async with self._session_factory() as session:
            async with session.begin():
                # merge = upsert on PK
                await session.merge(record)

        logger.info(
            "trace_saved",
            run_id=graph.run_id,
            events=len(graph.events),
            cost=graph.total_cost_usd,
        )

    async def get_trace(self, run_id: str) -> Optional[TraceGraph]:
        """Retrieve a TraceGraph by run_id. Returns None if not found."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(TraceRecord).where(TraceRecord.run_id == run_id)
            )
            record = result.scalar_one_or_none()

        if record is None:
            return None

        return TraceGraph.model_validate_json(record.graph_json)

    async def list_traces(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[TraceGraph]:
        """List most-recent traces for a session."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(TraceRecord)
                .where(TraceRecord.session_id == session_id)
                .order_by(TraceRecord.created_at.desc())
                .limit(limit)
            )
            records = result.scalars().all()

        return [TraceGraph.model_validate_json(r.graph_json) for r in records]

    async def delete_trace(self, run_id: str) -> bool:
        """Delete a trace. Returns True if deleted, False if not found."""
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    delete(TraceRecord).where(TraceRecord.run_id == run_id)
                )
        deleted = result.rowcount > 0
        if deleted:
            logger.info("trace_deleted", run_id=run_id)
        return deleted

    async def close(self) -> None:
        await self._engine.dispose()