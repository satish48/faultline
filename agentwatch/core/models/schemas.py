"""
AgentWatch — Core Data Models
==============================
All shared Pydantic v2 schemas. Every component in the system imports from here.
Nothing else defines data structures — single source of truth.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentEventType(str, Enum):
    LLM_CALL      = "llm_call"
    TOOL_CALL     = "tool_call"
    TOOL_RESULT   = "tool_result"
    DECISION      = "decision"
    HANDOFF       = "handoff"
    STATE_UPDATE  = "state_update"
    ERROR         = "error"
    START         = "start"
    END           = "end"


class ConflictType(str, Enum):
    SEMANTIC    = "semantic"
    PROCEDURAL  = "procedural"
    TEMPORAL    = "temporal"


class ResolutionStrategy(str, Enum):
    ARBITRATE  = "arbitrate"
    SYNTHESIZE = "synthesize"
    ESCALATE   = "escalate"
    RESET      = "reset"


class ConflictStatus(str, Enum):
    DETECTED   = "detected"
    RESOLVING  = "resolving"
    RESOLVED   = "resolved"
    ESCALATED  = "escalated"


class RiskLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class RootCauseType(str, Enum):
    INSTRUCTION_AMBIGUITY         = "instruction_ambiguity"
    CONTEXT_WINDOW_MISMATCH       = "context_window_mismatch"
    TOOL_RESULT_DIVERGENCE        = "tool_result_divergence"
    LOGICAL_CONTRADICTION         = "logical_contradiction"
    STALE_STATE                   = "stale_state"
    PERMISSION_BOUNDARY_VIOLATION = "permission_boundary_violation"
    UNKNOWN                       = "unknown"



# TRACE MODELS


class AgentEvent(BaseModel):
    event_id:    str            = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:      str
    agent_name:  str
    event_type:  AgentEventType
    timestamp:   datetime       = Field(default_factory=datetime.utcnow)
    sequence_no: int            = Field(..., ge=1)
    input_context: Dict[str, Any] = Field(default_factory=dict)
    output:        Optional[Any]  = None
    tool_name:   Optional[str]  = None
    tool_args:   Optional[Dict[str, Any]] = None
    tool_result: Optional[Any]  = None
    latency_ms: float | None = Field(default=None, ge=0)
    tokens_used: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    state_hash:   Optional[str]   = None
    metadata:     Dict[str, Any]  = Field(default_factory=dict)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class DecisionNode(BaseModel):
    node_id:          str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:           str
    agent_name:       str
    sequence_no:      int
    context_snapshot: Dict[str, Any] = Field(default_factory=dict)
    decision_type:    str
    options_available: List[str]     = Field(default_factory=list)
    chosen_option:    str
    confidence:       Optional[float] = Field(None, ge=0.0, le=1.0)
    alternative_option:  Optional[str] = None
    alternative_outcome: Optional[str] = None
    is_critical:         bool          = False
    source_event_ids:    List[str]     = Field(default_factory=list)


class TraceGraph(BaseModel):
    run_id:      str      = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id:  str
    created_at:  datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    agents_involved: List[str]          = Field(default_factory=list)
    events:          List[AgentEvent]   = Field(default_factory=list)
    decision_nodes:  List[DecisionNode] = Field(default_factory=list)
    edges:           Dict[str, List[str]] = Field(default_factory=dict)
    total_latency_ms: Optional[float] = Field(None, ge=0)
    total_cost_usd:   Optional[float] = Field(None, ge=0)
    total_tokens:     Optional[int]   = Field(None, ge=0)
    success:          bool            = True
    error_message:    Optional[str]   = None
    seen_state_hashes: List[str]      = Field(default_factory=list)

    @field_validator("events")
    @classmethod
    def events_ordered(cls, v: List[AgentEvent]) -> List[AgentEvent]:
        return sorted(v, key=lambda e: e.sequence_no)

    def get_events_by_agent(self, agent_name: str) -> List[AgentEvent]:
        return [e for e in self.events if e.agent_name == agent_name]

    def get_events_by_type(self, event_type: AgentEventType) -> List[AgentEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def get_handoffs(self) -> List[AgentEvent]:
        return self.get_events_by_type(AgentEventType.HANDOFF)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}



# AUDIT MODELS


class CounterfactualResult(BaseModel):
    node_id:                str
    run_id:                 str
    counterfactual_input:   Dict[str, Any]
    original_outcome:       str
    counterfactual_outcome: str
    divergence_point:       str
    outcome_changed:        bool
    sensitivity_score:      Optional[float] = Field(None, ge=0.0, le=1.0)
    explanation:            str


class AuditStep(BaseModel):
    step_no:          int
    agent_name:       str
    action:           str
    reasoning:        str
    outcome:          str
    risk_level:       RiskLevel
    source_event_ids: List[str] = Field(default_factory=list)


class AuditReport(BaseModel):
    report_id:    str      = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:       str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    steps:              List[AuditStep]
    critical_decisions: List[DecisionNode]     = Field(default_factory=list)
    counterfactuals:    List[CounterfactualResult] = Field(default_factory=list)
    summary:            str
    key_finding:        str
    overall_risk:       RiskLevel
    risk_factors:       List[str]              = Field(default_factory=list)
    compliance_data:    Dict[str, Any]         = Field(default_factory=dict)

    @model_validator(mode="after")
    def escalate_overall_risk(self) -> "AuditReport":
        step_risks = [s.risk_level for s in self.steps]
        if RiskLevel.HIGH in step_risks:
            self.overall_risk = RiskLevel.HIGH
        elif RiskLevel.MEDIUM in step_risks and self.overall_risk == RiskLevel.LOW:
            self.overall_risk = RiskLevel.MEDIUM
        return self

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}



# CONFLICT MODELS


class Conflict(BaseModel):
    conflict_id:  str      = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:       str
    detected_at:  datetime = Field(default_factory=datetime.utcnow)
    conflict_type: ConflictType
    status:        ConflictStatus = ConflictStatus.DETECTED
    agents_involved:     List[str]      = Field(default_factory=list, min_length=2)
    conflicting_outputs: Dict[str, Any] = Field(default_factory=dict)
    root_cause_type:       Optional[RootCauseType] = None
    root_cause_detail:     Optional[str]           = None
    root_cause_confidence: Optional[float]         = Field(None, ge=0.0, le=1.0)
    resolution_strategy:  Optional[ResolutionStrategy] = None
    resolution_output:    Optional[Any]                = None
    resolution_reasoning: Optional[str]               = None
    resolved_at:          Optional[datetime]           = None
    triggering_event_ids: List[str] = Field(default_factory=list)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class ConflictPattern(BaseModel):
    pattern_id:       str      = Field(default_factory=lambda: str(uuid.uuid4()))
    first_seen_at:    datetime = Field(default_factory=datetime.utcnow)
    last_seen_at:     datetime = Field(default_factory=datetime.utcnow)
    occurrence_count: int      = Field(default=1, ge=1)
    conflict_type:    ConflictType
    root_cause_type:  Optional[RootCauseType] = None
    agents_involved:  List[str]               = Field(default_factory=list)
    trigger_description: str
    fix_recommendation:  str
    fix_specificity:     str
    example_conflict_ids: List[str] = Field(default_factory=list)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class LoopDetectionResult(BaseModel):
    run_id:              str
    loop_detected:       bool
    loop_start_event_id: Optional[str] = None
    repeated_state_hash: Optional[str] = None
    loop_length:         Optional[int] = Field(None, ge=2)
    forced_resolution:   Optional[str] = None



# API MODELS


class TraceRequest(BaseModel):
    session_id:   str
    agent_events: List[AgentEvent] = Field(..., min_length=1)
    metadata:     Dict[str, Any]   = Field(default_factory=dict)

    @field_validator("agent_events")
    @classmethod
    def events_same_run_id(cls, v: List[AgentEvent]) -> List[AgentEvent]:
        run_ids = {e.run_id for e in v}
        if len(run_ids) > 1:
            raise ValueError(f"All events must share one run_id. Found: {run_ids}")
        return v


class TraceResponse(BaseModel):
    run_id:      str
    session_id:  str
    status:      str
    event_count: int
    created_at:  datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class AuditRequest(BaseModel):
    run_id:                  str
    include_counterfactuals: bool = True
    compliance_format:       bool = False
    max_counterfactuals:     int  = Field(default=5, ge=1, le=20)


class ConflictSummary(BaseModel):
    conflict_id:         str
    run_id:              str
    conflict_type:       ConflictType
    status:              ConflictStatus
    agents_involved:     List[str]
    detected_at:         datetime
    root_cause_type:     Optional[RootCauseType]      = None
    resolution_strategy: Optional[ResolutionStrategy] = None

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class HealthResponse(BaseModel):
    status:       str  = "ok"
    version:      str  = "1.0.0"
    db_connected: bool
    timestamp:    datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class ErrorResponse(BaseModel):
    error:  str
    detail: Optional[str] = None
    run_id: Optional[str] = None
