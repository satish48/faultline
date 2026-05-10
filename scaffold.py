"""
AgentWatch — Project Scaffold
Run this once to create the entire folder + file structure.
Usage: python scaffold.py
"""

import os

# ── Folders ──────────────────────────────────────────
folders = [
    "agentwatch/core/models",
    "agentwatch/core/tracer",
    "agentwatch/core/store",
    "agentwatch/agents/auditor",
    "agentwatch/agents/conflict",
    "agentwatch/api/routes",
    "agentwatch/dashboard",
    "agentwatch/demo",
    "agentwatch/tests",
    "agentwatch/docs",
]

# ── Files with starter content ────────────────────────
files = {
    # Init files
    "agentwatch/__init__.py": '"""AgentWatch — Agent Observability & Conflict Resolution."""\n__version__ = "1.0.0"\n',
    "agentwatch/core/__init__.py": "",
    "agentwatch/core/models/__init__.py": "from agentwatch.core.models.schemas import *\n",
    "agentwatch/core/tracer/__init__.py": "from agentwatch.core.tracer.execution_tracer import ExecutionTracer\n",
    "agentwatch/core/store/__init__.py": "from agentwatch.core.store.trace_store import TraceStore\n",
    "agentwatch/agents/__init__.py": "",
    "agentwatch/agents/auditor/__init__.py": "",
    "agentwatch/agents/conflict/__init__.py": "",
    "agentwatch/api/__init__.py": "",
    "agentwatch/api/routes/__init__.py": "",
    "agentwatch/tests/__init__.py": "",

    # Core source files (empty — CC will fill these)
    "agentwatch/core/models/schemas.py": "# TODO: Pydantic v2 models — filled by CC Prompt 1\n",
    "agentwatch/core/tracer/execution_tracer.py": "# TODO: ExecutionTracer — filled by CC Prompt 2\n",
    "agentwatch/core/store/trace_store.py": "# TODO: TraceStore — filled by CC Prompt 2\n",

    # Auditor agents
    "agentwatch/agents/auditor/decision_decomposer.py": "# TODO: DecisionDecomposer — filled by CC Prompt 3\n",
    "agentwatch/agents/auditor/nl_explainer.py": "# TODO: NLExplainer — filled by CC Prompt 3\n",
    "agentwatch/agents/auditor/counterfactual_explorer.py": "# TODO: CounterfactualExplorer — filled by CC Prompt 4\n",
    "agentwatch/agents/auditor/compliance_reporter.py": "# TODO: ComplianceReporter — filled by CC Prompt 4\n",

    # Conflict agents
    "agentwatch/agents/conflict/conflict_sensor.py": "# TODO: ConflictSensor — filled by CC Prompt 5\n",
    "agentwatch/agents/conflict/root_cause_classifier.py": "# TODO: RootCauseClassifier — filled by CC Prompt 5\n",
    "agentwatch/agents/conflict/resolution_strategist.py": "# TODO: ResolutionStrategist — filled by CC Prompt 6\n",
    "agentwatch/agents/conflict/loop_killer.py": "# TODO: LoopKiller — filled by CC Prompt 6\n",
    "agentwatch/agents/conflict/conflict_memory.py": "# TODO: ConflictMemoryStore — filled by CC Prompt 6\n",

    # API
    "agentwatch/api/main.py": "# TODO: FastAPI app — filled by CC Prompt 7\n",
    "agentwatch/api/routes/trace_routes.py": "# TODO: Trace routes — filled by CC Prompt 7\n",
    "agentwatch/api/routes/audit_routes.py": "# TODO: Audit routes — filled by CC Prompt 7\n",
    "agentwatch/api/routes/conflict_routes.py": "# TODO: Conflict routes — filled by CC Prompt 7\n",

    # Dashboard + Demo
    "agentwatch/dashboard/app.py": "# TODO: Streamlit dashboard — filled by CC Prompt 8\n",
    "agentwatch/demo/demo_agents.py": "# TODO: Demo agents — filled by CC Prompt 9\n",
    "agentwatch/demo/run_demo.py": "# TODO: Demo runner — filled by CC Prompt 9\n",

    # Tests
    "agentwatch/tests/conftest.py": "# TODO: Shared fixtures — filled progressively\n",
    "agentwatch/tests/test_tracer.py": "# TODO: Tracer tests — filled by CC Prompt 2\n",
    "agentwatch/tests/test_auditor.py": "# TODO: Auditor tests — filled by CC Prompt 3 & 4\n",
    "agentwatch/tests/test_conflict.py": "# TODO: Conflict tests — filled by CC Prompt 5 & 6\n",

    # Config files
    "agentwatch/requirements.txt": """\
# Core LLM
anthropic>=0.25.0
openai>=1.30.0
langgraph>=0.1.0
langchain>=0.2.0
langchain-anthropic>=0.1.0

# API
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0
pydantic-settings>=2.2.0
python-multipart>=0.0.9

# Storage
networkx>=3.3
sqlalchemy>=2.0.30
aiosqlite>=0.20.0

# Observability
opentelemetry-sdk>=1.24.0
opentelemetry-exporter-otlp>=1.24.0
langfuse>=2.30.0

# DS / ML (your moat)
numpy>=1.26.0
scipy>=1.13.0
scikit-learn>=1.5.0
pandas>=2.2.0

# Utilities
python-dotenv>=1.0.0
structlog>=24.1.0
tenacity>=8.3.0
httpx>=0.27.0

# Dashboard
streamlit>=1.35.0
plotly>=5.22.0

# Testing
pytest>=8.2.0
pytest-asyncio>=0.23.0
pytest-cov>=5.0.0
""",

    "agentwatch/.env.example": """\
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
DATABASE_URL=sqlite+aiosqlite:///agentwatch.db
LOG_LEVEL=INFO
COST_PER_1K_TOKENS=0.003
LOOP_DETECTION_THRESHOLD=3
""",

    "agentwatch/pyproject.toml": """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "agentwatch"
version = "1.0.0"
description = "Agent Observability and Conflict Resolution System"
requires-python = ">=3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
""",

    "agentwatch/README.md": """\
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
""",
}


def scaffold():
    base = os.path.dirname(os.path.abspath(__file__))

    # Create folders
    for folder in folders:
        path = os.path.join(base, folder)
        os.makedirs(path, exist_ok=True)
        print(f"  [dir]  {folder}")

    # Create files
    for rel_path, content in files.items():
        path = os.path.join(base, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(content)
            print(f"  [file] {rel_path}")
        else:
            print(f"  [skip] {rel_path} (already exists)")

    print("\nDone. Your AgentWatch scaffold is ready.")
    print("Next: open Claude Code and run Prompt 1.")


if __name__ == "__main__":
    scaffold()