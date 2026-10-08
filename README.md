# Databricks Tokenomics

Track and analyze the **token economics** of GenAI workloads on Databricks — the
token consumption, cost (DBU / dollars), and efficiency of model-serving,
Foundation Model APIs, and agent workloads — so you can attribute spend, spot
regressions, and optimize prompts and models.

> **Status:** starter scaffold. Sections marked _TODO_ are placeholders to fill in
> as the project takes shape.

## What you get

_TODO — describe the deliverable._ For example:

- **Usage ingestion** — pull token/usage and billing data (e.g. `system.billing.usage`,
  model-serving logs, MLflow traces) into Unity Catalog tables.
- **Cost attribution** — join usage to workloads, teams, and use cases to answer
  "who spent what, on which model, for which feature."
- **Dashboards & alerts** — AI/BI dashboards and thresholds for spend and token trends.

## Prerequisites

- A Databricks workspace with Unity Catalog and access to the relevant billing /
  usage system tables and model-serving endpoints.
- The [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) (`databricks`)
  authenticated to that workspace.
- [`uv`](https://docs.astral.sh/uv/) installed (for Python tooling).

```bash
# Authenticate the CLI (creates a profile you'll select later)
databricks auth login --profile my-workspace
```

> Never auto-select a profile — always pass `--profile <name>` explicitly.

## Setup

_TODO — replace with the real setup steps once the project has an entry point._

```bash
# Clone and install
git clone https://github.com/bcheng004/databricks-tokenomics.git
cd databricks-tokenomics
uv sync
```

## Project structure

_TODO — update as the layout is created._

```
databricks-tokenomics/
├── AGENTS.md          # Instructions for AI coding agents
├── README.md          # This file
├── .gitignore
└── ...
```

## Configuration

_TODO — document the environment variables / bundle variables the project reads._

Copy `.env.example` to `.env` and fill in the values (do not commit `.env`).

## Concepts

- **Token** — the unit of input/output consumed by an LLM request; the basis for
  most GenAI billing on Databricks.
- **Tokenomics** — the accounting and optimization of token usage and its cost
  (tokens → DBUs → dollars) across models, endpoints, and workloads.
- **Attribution** — mapping usage back to a team, use case, or feature so spend
  is explainable and actionable.

## References

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/)
- [System tables: billing usage](https://docs.databricks.com/admin/system-tables/billing.html)
- [Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)
- [Foundation Model APIs](https://docs.databricks.com/machine-learning/foundation-models/index.html)
- [MLflow tracing](https://mlflow.org/docs/latest/tracing/)
