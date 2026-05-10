"""
AgentWatch — FastAPI Dependencies
=================================
Shared dependency injection for all routes.
LLMClient auto-detects provider from environment — Groq preferred.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings
from fastapi import Request

from agentwatch.core.llm.client import LLMClient
from agentwatch.core.store.trace_store import TraceStore

from agentwatch.agents.auditor.compliance_reporter import ComplianceReporter
from agentwatch.agents.auditor.counterfactual_explorer import CounterfactualExplorer
from agentwatch.agents.auditor.decision_decomposer import DecisionDecomposer
from agentwatch.agents.auditor.nl_explainer import NLExplainer

from agentwatch.agents.conflict.conflict_memory import ConflictMemoryStore
from agentwatch.agents.conflict.conflict_sensor import ConflictSensor
from agentwatch.agents.conflict.resolution_strategist import ResolutionStrategist
from agentwatch.agents.conflict.root_cause_classifier import RootCauseClassifier


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    database_url: str = "sqlite+aiosqlite:///agentwatch.db"
    conflict_memory_path: str = "conflict_memory.json"
    log_level: str = "INFO"
    cost_per_1k_tokens: float = 0.0001
    loop_detection_threshold: int = 3

    class Config:
        env_file = ".env"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()


def init_components(app_state) -> None:
    settings = get_settings()

    app_state.settings = settings
    app_state.llm_client = LLMClient.from_env()
    app_state.store = TraceStore(database_url=settings.database_url)
    app_state.conflict_memory = ConflictMemoryStore(
        store_path=settings.conflict_memory_path
    )
    app_state.audit_cache = {}


def get_store(request: Request) -> TraceStore:
    return request.app.state.store


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_conflict_memory(request: Request) -> ConflictMemoryStore:
    return request.app.state.conflict_memory


def get_audit_cache(request: Request) -> dict:
    return request.app.state.audit_cache


def get_decomposer(request: Request) -> DecisionDecomposer:
    return DecisionDecomposer(client=get_llm_client(request))


def get_explainer(request: Request) -> NLExplainer:
    return NLExplainer(client=get_llm_client(request))


def get_cf_explorer(request: Request) -> CounterfactualExplorer:
    return CounterfactualExplorer(client=get_llm_client(request))


def get_reporter(request: Request) -> ComplianceReporter:
    return ComplianceReporter(client=get_llm_client(request))


def get_sensor(request: Request) -> ConflictSensor:
    return ConflictSensor(client=get_llm_client(request))


def get_classifier(request: Request) -> RootCauseClassifier:
    return RootCauseClassifier(client=get_llm_client(request))


def get_strategist(request: Request) -> ResolutionStrategist:
    return ResolutionStrategist(client=get_llm_client(request))