#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "databricks-sdk==0.94.0",
#     # Pinned: the notebook's trace-location API (UnityCatalog +
#     # set_experiment(trace_location=...)) is missing from the newest
#     # mlflow-skinny[databricks] resolution; re-verify before bumping.
#     "mlflow-skinny[databricks]==3.16.1",
# ]
# ///
"""Provision an MLflow experiment with a Unity Catalog-managed trace location.

Python port of ``notebooks/00-mlflow-experiment-uc-managed.ipynb`` from
github.com/bcheng004/databricks-genie-agents-mlflow. uv builds an isolated
environment from the dependencies above, replacing the notebook's
``%pip install`` + ``dbutils.library.restartPython()`` preamble:

    uv run scripts/setup_mlflow_uc_experiment.py

or, via the console script registered in ``pyproject.toml``:

    uv run setup-mlflow-uc-experiment

Configuration is read from terminal prompts, command-line arguments, and the
environment:

- Experiment name (prompted for and created as ``/Shared/<name>``; use
  ``--experiment-name`` or ``EXPERIMENT_NAME`` for non-interactive runs)
- Schema name (prompted for and created in the selected catalog; use
  ``--schema`` or ``UC_SCHEMA`` for non-interactive runs)
- ``UC_TABLE_PREFIX`` (default ``evals``)
- ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` (default: auto-discovered
  "Serverless Starter Warehouse")

The Databricks profile is never auto-selected. It is resolved from the
``--profile`` flag, then ``DATABRICKS_CONFIG_PROFILE``, then an
interactive prompt listing the profiles in ``~/.databrickscfg``.

The target catalog is likewise resolved from ``--catalog``, then
``UC_CATALOG``, then an interactive prompt listing the catalogs visible
to you; a catalog that is not visible can be created after confirmation.

On success the script also updates the mlflow-codex destination file
``.codex/mlflow-tracing.json`` (override the path with
``--tracing-config``); restart Codex afterwards to pick it up.

To start over, drop the schema first (the notebook's commented-out
``%sql`` cell)::

    DROP SCHEMA <catalog>.<schema> CASCADE
"""

import argparse
import configparser
import json
import os
import sys
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    BadRequest,
    PermissionDenied,
    ResourceAlreadyExists,
)
from mlflow.entities.trace_location import UnityCatalog

EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME")
UC_SCHEMA = os.environ.get("UC_SCHEMA")
UC_TABLE_PREFIX = os.environ.get("UC_TABLE_PREFIX", "evals")


def find_default_warehouse(w: WorkspaceClient) -> str | None:
    """Return the ID of the first serverless "Serverless Starter Warehouse"."""
    return next(
        (
            wh.id
            for wh in w.warehouses.list()
            if "Serverless Starter Warehouse" in wh.name and wh.enable_serverless_compute
        ),
        None,
    )


def create_schema(w: WorkspaceClient, catalog_name: str, schema_name: str) -> None:
    """Create a new schema and refuse to reuse an existing one."""
    full_name = f"{catalog_name}.{schema_name}"
    try:
        w.schemas.create(catalog_name=catalog_name, name=schema_name)
    except (BadRequest, ResourceAlreadyExists) as error:
        if "already exists" not in str(error).lower():
            raise
        sys.exit(f"Schema '{full_name}' already exists; choose a new schema name.")
    except PermissionDenied:
        sys.exit(
            f"Permission denied creating schema '{full_name}'; "
            f"request CREATE SCHEMA on catalog '{catalog_name}'."
        )
    print(f"Created schema {full_name}")


def list_profiles() -> list[str]:
    """Parse the profile names from ~/.databrickscfg."""
    path = Path.home() / ".databrickscfg"
    if not path.exists():
        return []
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path)
    # __settings__ is a reserved CLI section, not an auth profile.
    return [s for s in parser.sections() if s != "__settings__"]


def resolve_profile(explicit: str | None) -> str:
    """--profile flag > DATABRICKS_CONFIG_PROFILE > interactive prompt."""
    profile = explicit or os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if profile:
        print(f"Using Databricks profile: {profile}")
        return profile
    profiles = list_profiles()
    if not profiles:
        sys.exit(
            "No profiles found in ~/.databrickscfg; run "
            "'databricks auth login --profile <name>' first."
        )
    if not sys.stdin.isatty():
        sys.exit(
            "No Databricks profile given and stdin is not interactive; "
            "pass --profile or set DATABRICKS_CONFIG_PROFILE."
        )
    print("Databricks profiles in ~/.databrickscfg:")
    for i, name in enumerate(profiles, start=1):
        print(f"  {i}. {name}")
    while True:
        choice = input("Select profile (number or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(profiles):
            return profiles[int(choice) - 1]
        if choice in profiles:
            return choice
        print(f"  Not one of: {', '.join(profiles)}")


def resolve_catalog(w: WorkspaceClient, explicit: str | None) -> str:
    """--catalog flag > UC_CATALOG env > interactive prompt."""
    visible = sorted(c.name for c in w.catalogs.list())
    catalog = explicit or os.environ.get("UC_CATALOG")
    if catalog:
        print(f"Using catalog: {catalog}")
        if catalog not in visible:
            sys.exit(
                f"Catalog '{catalog}' does not exist or is not visible. Visible catalogs:\n  "
                + "\n  ".join(visible)
                + "\nOmit --catalog / UC_CATALOG to choose one interactively."
            )
        return catalog
    if not sys.stdin.isatty():
        sys.exit(
            "No catalog given and stdin is not interactive; "
            "pass --catalog or set UC_CATALOG."
        )
    print("Unity Catalog catalogs:")
    for i, name in enumerate(visible, start=1):
        print(f"  {i}. {name}")
    while True:
        choice = input("Select catalog (number or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(visible):
            return visible[int(choice) - 1]
        if choice:
            if choice in visible:
                return choice
            yn = input(f"'{choice}' is not visible to you. Create it? [y/N] ").strip().lower()
            if yn != "y":
                continue
            try:
                w.catalogs.create(name=choice)
            except (BadRequest, ResourceAlreadyExists) as e:
                if "already exists" not in str(e).lower():
                    raise
                sys.exit(
                    f"Catalog '{choice}' already exists but is not visible to you; "
                    "check your permissions."
                )
            except PermissionDenied:
                sys.exit(
                    f"Permission denied creating catalog '{choice}'; "
                    "ask a metastore admin or pick an existing catalog."
                )
            print(f"Created catalog {choice}")
            return choice
        print("  Enter a number or name.")


def resolve_experiment_name(explicit: str | None) -> str:
    """Resolve a name and place the experiment directly under /Shared."""
    name = explicit or EXPERIMENT_NAME
    if name is not None:
        name = name.strip()
    if not name and not sys.stdin.isatty():
        sys.exit(
            "No experiment name given and stdin is not interactive; "
            "pass --experiment-name or set EXPERIMENT_NAME."
        )
    while not name:
        name = input("Experiment name (created under /Shared): ").strip()
        if not name:
            print("  Enter an experiment name.")
    if name in {".", ".."} or "/" in name:
        sys.exit(
            f"Experiment name '{name}' must be a name, not a path; "
            "the experiment is always created directly under /Shared."
        )
    path = f"/Shared/{name}"
    print(f"Creating experiment: {path}")
    return path


def resolve_schema_name(explicit: str | None, catalog_name: str) -> str:
    """Resolve a new schema name inside the selected catalog."""
    name = explicit or UC_SCHEMA
    if name is not None:
        name = name.strip()
    if not name and not sys.stdin.isatty():
        sys.exit(
            "No schema name given and stdin is not interactive; "
            "pass --schema or set UC_SCHEMA."
        )
    while not name:
        name = input(f"New schema name (created in {catalog_name}): ").strip()
        if not name:
            print("  Enter a schema name.")
    if name in {".", ".."} or "." in name or "/" in name:
        sys.exit(
            f"Schema name '{name}' must be a name, not a qualified name or path; "
            f"the schema is always created in catalog '{catalog_name}'."
        )
    print(f"Creating schema: {catalog_name}.{name}")
    return name


def write_tracing_config(path: Path, profile: str, experiment_id: str, trace_location: str) -> None:
    """Update the mlflow-codex destination file, preserving unknown keys."""
    config: dict = {}
    if path.exists():
        try:
            config = json.loads(path.read_text())
        except json.JSONDecodeError:
            sys.exit(f"{path} is not valid JSON; fix or remove it and re-run.")
    config.update(
        trackingUri=f"databricks://{profile}",
        experimentId=experiment_id,
        traceLocation=trace_location,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Updated {path} (restart Codex to pick it up)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        help="Databricks CLI profile to authenticate with (prompted for if omitted)",
    )
    parser.add_argument(
        "--catalog",
        help="Unity Catalog to store traces in (prompted for if omitted)",
    )
    parser.add_argument(
        "--experiment-name",
        help="Name for the new /Shared experiment (prompted for if omitted)",
    )
    parser.add_argument(
        "--schema",
        help="Name for the new schema in the selected catalog (prompted for if omitted)",
    )
    parser.add_argument(
        "--tracing-config",
        default=".codex/mlflow-tracing.json",
        help="mlflow-codex destination file to update on success "
        "(relative to the current directory)",
    )
    args = parser.parse_args()

    profile = resolve_profile(args.profile)
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile

    w = WorkspaceClient()
    current_user = w.current_user.me()
    print(f"Authenticated as {current_user.user_name} ({w.config.host})")

    experiment_name = resolve_experiment_name(args.experiment_name)
    mlflow.set_tracking_uri(f"databricks://{profile}")
    if mlflow.get_experiment_by_name(experiment_name) is not None:
        sys.exit(
            f"Experiment '{experiment_name}' already exists. Choose a new name with "
            "--experiment-name or EXPERIMENT_NAME; this setup command does not reuse experiments."
        )

    catalog = resolve_catalog(w, args.catalog)
    schema = resolve_schema_name(args.schema, catalog)

    # Specify the ID of a SQL warehouse you have access to.
    warehouse_id = os.environ.get("MLFLOW_TRACING_SQL_WAREHOUSE_ID")
    if not warehouse_id:
        warehouse_id = find_default_warehouse(w)
        print(f"{warehouse_id=}")
    if not warehouse_id:
        names = sorted(wh.name for wh in w.warehouses.list())
        sys.exit(
            "No 'Serverless Starter Warehouse' found. Available warehouses:\n  "
            + "\n  ".join(names)
            + "\nSet MLFLOW_TRACING_SQL_WAREHOUSE_ID to choose one explicitly."
        )

    create_schema(w, catalog, schema)

    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id

    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        trace_location=UnityCatalog(
            catalog_name=catalog,
            schema_name=schema,
            table_prefix=UC_TABLE_PREFIX,  # defaults to experiment id if not provided
        ),
    )
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    print(f"Created experiment {experiment.name} ({experiment.experiment_id})")
    print(experiment.trace_location.full_otel_spans_table_name)

    write_tracing_config(
        Path(args.tracing_config),
        profile,
        experiment.experiment_id,
        f"{catalog}.{schema}.{UC_TABLE_PREFIX}",
    )


if __name__ == "__main__":
    main()
