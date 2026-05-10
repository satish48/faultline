"""
AgentWatch — Demo Agent System
================================
A deliberately imperfect 3-agent customer service pipeline.
Designed to generate all three conflict types in a single run.

Agents:
  CustomerServiceAgent  — approves refunds based on customer tier
  ComplianceAgent       — blocks refunds above risk threshold
  BillingAgent          — processes using stale order data

Conflicts generated:
  SEMANTIC   — CustomerService approves, Compliance blocks same request
  PROCEDURAL — Both agents call process_refund within 3 sequence steps
  TEMPORAL   — BillingAgent uses hardcoded stale timestamp

This is NOT a production system. It is a controlled demo environment
designed to showcase AgentWatch's detection capabilities.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog

from agentwatch.core.tracer.execution_tracer import ExecutionTracer
from agentwatch.core.models.schemas import AgentEventType

logger = structlog.get_logger(__name__)


# ── Shared state passed between agents ───────────────

class OrderContext:
    """Simulated order context with deliberately stale timestamp."""

    def __init__(self, order_id: str, customer_id: str, amount: float, tier: str):
        self.order_id    = order_id
        self.customer_id = customer_id
        self.amount      = amount
        self.tier        = tier
        # Deliberately stale — 3 hours old
        self.updated_at  = (datetime.utcnow() - timedelta(hours=3)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id":    self.order_id,
            "customer_id": self.customer_id,
            "amount":      self.amount,
            "tier":        self.tier,
            "updated_at":  self.updated_at,
        }


# ══════════════════════════════════════════════════════
# Agent 1 — CustomerServiceAgent
# Generates: SEMANTIC conflict (approves what Compliance blocks)
# ══════════════════════════════════════════════════════

class CustomerServiceAgent:
    """
    Approves refunds for standard and premium tier customers.
    Has no knowledge of the compliance risk threshold.
    This is the SEMANTIC conflict source.
    """

    TIER_LIMITS = {
        "standard": 100.0,
        "premium":  500.0,
        "vip":      float("inf"),
    }

    def __init__(self, tracer: ExecutionTracer):
        self.tracer = tracer
        self.name   = "customer_service_agent"

    async def handle_refund_request(
        self, order: OrderContext
    ) -> Dict[str, Any]:
        """
        Decide whether to approve or escalate a refund request.
        Decision is based purely on customer tier — ignores risk signals.
        """
        ctx = {
            **order.to_dict(),
            "agent": self.name,
            "policy": "approve_by_tier",
        }

        async with self.tracer.trace_step(
            self.name,
            AgentEventType.DECISION,
            input_context=ctx,
        ) as ev:
            limit = self.TIER_LIMITS.get(order.tier, 0.0)
            approved = order.amount <= limit

            decision = "approve_refund" if approved else "escalate_to_manager"
            ev.output    = decision
            ev.tokens_used = 180  # simulated

            logger.info(
                "cs_decision",
                order_id=order.order_id,
                amount=order.amount,
                tier=order.tier,
                decision=decision,
            )

        # Simulate also calling process_refund tool (PROCEDURAL conflict source)
        async with self.tracer.trace_step(
            self.name,
            AgentEventType.TOOL_CALL,
            input_context={"order_id": order.order_id, "amount": order.amount},
            tool_name="process_refund",
            tool_args={"order_id": order.order_id, "amount": order.amount},
        ) as tool_ev:
            tool_ev.output    = {"status": "queued", "ref": "CS-001"}
            tool_ev.tokens_used = 0  # tool call, no tokens

        return {
            "agent":    self.name,
            "decision": decision,
            "approved": approved,
            "amount":   order.amount,
            "tier":     order.tier,
        }


# ══════════════════════════════════════════════════════
# Agent 2 — ComplianceAgent
# Generates: SEMANTIC conflict (blocks what CS approved)
#            PROCEDURAL conflict (also calls process_refund)
# ══════════════════════════════════════════════════════

class ComplianceAgent:
    """
    Blocks refunds that exceed the risk threshold regardless of tier.
    Has no knowledge of what CustomerService decided.
    This is the SEMANTIC + PROCEDURAL conflict source.
    """

    RISK_THRESHOLD = 100.0  # blocks anything above this

    def __init__(self, tracer: ExecutionTracer):
        self.tracer = tracer
        self.name   = "compliance_agent"

    async def review_action(
        self, order: OrderContext, cs_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Independently reviews the refund for compliance risk.
        Does NOT coordinate with CustomerService — this causes the conflict.
        """
        ctx = {
            **order.to_dict(),
            "agent":           self.name,
            "risk_threshold":  self.RISK_THRESHOLD,
            "cs_decision":     cs_decision.get("decision"),
            # Note: compliance agent sees the cs decision but ignores it
        }

        async with self.tracer.trace_step(
            self.name,
            AgentEventType.DECISION,
            input_context=ctx,
        ) as ev:
            high_risk = order.amount > self.RISK_THRESHOLD
            decision  = "block_refund" if high_risk else "approve_refund"

            ev.output      = decision
            ev.tokens_used = 210

            logger.info(
                "compliance_decision",
                order_id=order.order_id,
                amount=order.amount,
                threshold=self.RISK_THRESHOLD,
                decision=decision,
                conflicts_with_cs=decision != cs_decision.get("decision"),
            )

        # Also calls process_refund → PROCEDURAL conflict with CS
        async with self.tracer.trace_step(
            self.name,
            AgentEventType.TOOL_CALL,
            input_context={"order_id": order.order_id, "action": decision},
            tool_name="process_refund",
            tool_args={"order_id": order.order_id, "action": decision},
        ) as tool_ev:
            tool_ev.output    = {"status": "compliance_hold", "ref": "COMP-001"}
            tool_ev.tokens_used = 0

        return {
            "agent":    self.name,
            "decision": decision,
            "blocked":  high_risk,
            "reason":   f"Amount ${order.amount} exceeds risk threshold ${self.RISK_THRESHOLD}",
        }


# ══════════════════════════════════════════════════════
# Agent 3 — BillingAgent
# Generates: TEMPORAL conflict (uses stale order data)
# ══════════════════════════════════════════════════════

class BillingAgent:
    """
    Processes the final billing action using order context.
    Uses the stale updated_at timestamp from OrderContext — TEMPORAL conflict.
    """

    def __init__(self, tracer: ExecutionTracer):
        self.tracer = tracer
        self.name   = "billing_agent"

    async def process(
        self, order: OrderContext, action: str
    ) -> Dict[str, Any]:
        """
        Execute the billing action.
        The stale updated_at in order context is the TEMPORAL conflict trigger.
        """
        # Uses stale context — this is the temporal conflict
        ctx = {
            **order.to_dict(),   # contains stale updated_at
            "agent":  self.name,
            "action": action,
            # Fresh timestamp the billing system actually has
            "billing_system_updated_at": datetime.utcnow().strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        }

        async with self.tracer.trace_step(
            self.name,
            AgentEventType.DECISION,
            input_context=ctx,
        ) as ev:
            # Processes based on stale data — wrong decision possible
            decision = f"execute_{action}"
            ev.output      = decision
            ev.tokens_used = 150

        async with self.tracer.trace_step(
            self.name,
            AgentEventType.TOOL_CALL,
            input_context={"order_id": order.order_id, "action": action},
            tool_name="billing_execute",
            tool_args={
                "order_id":   order.order_id,
                "amount":     order.amount,
                "action":     action,
                "as_of":      order.updated_at,  # stale!
            },
        ) as tool_ev:
            tool_ev.output    = {"status": "executed", "ref": "BILL-001"}
            tool_ev.tokens_used = 0

        logger.info(
            "billing_processed",
            order_id=order.order_id,
            action=action,
            using_stale_data=True,
        )

        return {
            "agent":    self.name,
            "decision": decision,
            "executed": True,
            "stale_context_used": order.updated_at,
        }
