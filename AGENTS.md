# AGENTS.md

Instructions for AI coding agents working in **databricks-tokenomics** — a project
for tracking and analyzing the token economics (usage, cost, efficiency) of GenAI
workloads on Databricks.

## Deployments — do not deploy without explicit instruction

UNDER NO CIRCUMSTANCES deploy without an explicit request. This includes
`databricks bundle deploy` / `databricks bundle run` and any other deploy/publish
commands. Making code changes does NOT imply permission to deploy — even if the
change is small or the previous step was a deploy. Make and validate the edits
(compile / type / signature checks), then stop and wait for an explicit deploy
request.

## Databricks conventions

- **Never auto-select a profile.** Pass `--profile <name>` on every `databricks`
  command and let the user choose. Do not hardcode a profile in scripts.
- Treat Unity Catalog objects (catalogs, schemas, tables) and warehouse IDs as
  configuration — read them from env / bundle variables, never hardcode.
- Keep secrets out of the repo. Use `.env` (gitignored) locally and Databricks
  secrets / app resources in the workspace; commit only `.env.example`.

## Setup & tooling

- Python dependencies are managed with [`uv`](https://docs.astral.sh/uv/):
  `uv sync` to install, `uv run <cmd>` to run.
- The `uv.lock` file is intentionally gitignored (see `.gitignore`).

## Working agreements

- **Explore before large changes.** For multi-file work, read the relevant code
  first and plan; prefer the smallest change that solves the problem.
- **Verify before claiming completion.** Run the relevant checks (tests, type
  checks, a dry run) and report the actual output. If something was skipped or
  failed, say so plainly.
- **Match the surrounding code** — naming, structure, comment density, and idiom.
- **Confirm irreversible or outward-facing actions** (deploys, deletes, anything
  that writes to the workspace or an external service) before proceeding.

## Project structure

_TODO — keep this in sync as the layout is created._

```
databricks-tokenomics/
├── AGENTS.md          # This file
├── README.md          # Project overview and setup
├── pyproject.toml     # Project metadata and dependencies (uv)
├── .gitignore
├── scripts/           # Standalone `uv run` scripts (PEP 723 inline dependencies)
└── ...
```

## References

- Project overview and setup: `README.md`
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/)
- [System tables: billing usage](https://docs.databricks.com/admin/system-tables/billing.html)
