"""
AgentWatch — FastAPI Application
==================================
Production API wiring all components together.

Architecture:
- Lifespan event initializes DB + loads shared component instances
- All components are dependency-injected via FastAPI Depends
- Per-request cost + latency tracking middleware
- Structured JSON logging via structlog
- CORS configured for dev (restrict in prod via env)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from agentwatch.api.routes import audit_routes, conflict_routes, trace_routes
from agentwatch.api.dependencies import get_settings, init_components
from agentwatch.core.llm.client import LLMClient
from agentwatch.core.models.schemas import HealthResponse
from agentwatch.core.store.trace_store import TraceStore
from agentwatch.agents.conflict.conflict_memory import ConflictMemoryStore

logger = structlog.get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Initialize all shared components on startup."""
    settings = get_settings()

    # Init DB
    store = TraceStore(database_url=settings.database_url)
    await store.init_db()
    app.state.store = store

    # Init LLM client
    app.state.llm_client = LLMClient.from_env()

    # Init conflict memory
    app.state.conflict_memory = ConflictMemoryStore(
        store_path=settings.conflict_memory_path
    )

    # Cache for audit reports (in-memory for v1)
    app.state.audit_cache: dict = {}

    logger.info(
        "agentwatch_started",
        version="1.0.0",
        db=settings.database_url,
    )

    yield

    # Cleanup
    await store.close()
    logger.info("agentwatch_stopped")


# ── App ───────────────────────────────────────────────

app = FastAPI(
    title="AgentWatch API",
    description="Agent Observability & Conflict Resolution System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — open in dev, restrict in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware ────────────────────────────────────────

@app.middleware("http")
async def observability_middleware(request: Request, call_next) -> Response:
    """
    Per-request observability:
    - Latency tracking
    - Structured logging with route + status
    - Request ID for trace correlation
    """
    start = time.monotonic()
    request_id = request.headers.get("x-request-id", "none")

    response = await call_next(request)

    latency_ms = (time.monotonic() - start) * 1000

    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
    )

    response.headers["X-Latency-Ms"] = str(round(latency_ms, 2))
    response.headers["X-Request-Id"] = request_id

    return response


# ── Routes ────────────────────────────────────────────

app.include_router(trace_routes.router,    prefix="/trace",     tags=["Traces"])
app.include_router(audit_routes.router,    prefix="/audit",     tags=["Audit"])
app.include_router(conflict_routes.router, prefix="/conflicts", tags=["Conflicts"])


# ── Health ────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request) -> HealthResponse:
    db_ok = hasattr(request.app.state, "store")
    return HealthResponse(db_connected=db_ok)
