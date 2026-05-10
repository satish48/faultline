# AgentWatch

**Autonomous Agent Observability & Conflict Resolution System**

A production-grade system that instruments any multi-agent pipeline,
audits decisions with causal explainability, and detects + resolves
inter-agent conflicts autonomously.

## Quick Start

```bash
cp .env.example .env        # add your API keys
pip install -r requirements.txt
uvicorn agentwatch.api.main:app --reload   # API on :8000
streamlit run agentwatch/dashboard/app.py  # Dashboard on :8501
python agentwatch/demo/run_demo.py --pretty  # Run demo
```

## Architecture

Two pillars on a shared instrumentation layer:

- **Pillar A — Decision Auditor**: Execution Tracer → Decision Decomposer
  → Counterfactual Explorer → NL Explainer → Compliance Reporter
- **Pillar B — Conflict Detector**: Conflict Sensor → Root Cause Classifier
  → Resolution Strategist → Conflict Memory Store

## Structure

```
agentwatch/
  core/          # Tracer, Store, Models
  agents/        # Auditor + Conflict agents
  api/           # FastAPI backend
  dashboard/     # Streamlit UI
  demo/          # Demo system
  tests/         # pytest suite
  docs/          # Technical documentation
```
