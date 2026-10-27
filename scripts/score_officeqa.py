"""Score existing OfficeQA traces using UID-matched expected facts."""

import argparse
import csv
import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
from dotenv import dotenv_values
from mlflow.genai.scorers import Correctness
from mlflow.tracing.constant import AssessmentMetadataKey

from scripts.run_officeqa import write_json


def load_references(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"uid", "question", "answer"}.issubset(reader.fieldnames or []):
            raise ValueError("Reference CSV must contain uid, question, and answer")
        references = {}
        for row in reader:
            uid = row["uid"]
            if not uid or uid in references or not row["answer"].strip():
                raise ValueError(f"Duplicate uid or empty reference answer: {uid!r}")
            references[uid] = {key: row[key] for key in ("question", "answer")}
    if not references:
        raise ValueError("Reference CSV is empty")
    return references


def evaluation_row(trace, references: dict[str, dict]) -> dict:
    reference = references[trace.info.tags["uid"]]
    root = trace.data._get_root_span()
    if root is None:
        raise ValueError(f"Missing root span: {trace.info.trace_id}")
    response = root.outputs
    if isinstance(response, str):
        try:
            decoded = json.loads(response)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, str):
            response = decoded
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Missing text response: {trace.info.trace_id}")
    return {"trace": trace, "inputs": {"question": reference["question"]},
            "outputs": response, "expectations": {"expected_facts": [reference["answer"]]}}


def successful_assessment(info, expected_facts: list[str], run_id: str, judge_model: str):
    assessments = info.assessments or []
    expected = any(assessment.name == "expected_facts" and assessment.value == expected_facts
                   for assessment in assessments)
    if not expected:
        return None
    return next((assessment for assessment in assessments
                 if assessment.name == "correctness" and assessment.error is None
                 and assessment.value in ("yes", "no")
                 and assessment.source is not None
                 and assessment.source.source_id == judge_model
                 and (assessment.metadata or {}).get(AssessmentMetadataKey.SOURCE_RUN_ID) == run_id),
                None)


def latest_traces_by_uid(traces: list) -> list:
    latest = {}
    for trace in traces:
        uid = trace.info.tags["uid"]
        previous = latest.get(uid)
        if previous is None or (trace.info.timestamp_ms, trace.info.trace_id) > (
            previous.info.timestamp_ms, previous.info.trace_id,
        ):
            latest[uid] = trace
    return [latest[uid] for uid in sorted(latest)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--warehouse-id")
    parser.add_argument("--tracing-config", type=Path, default=Path(".codex/mlflow-tracing.json"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/officeqa-pro-v2"))
    parser.add_argument("--revision", help="Pinned dataset commit when snapshot.json is unavailable")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/officeqa-correctness-opus-5-5"))
    parser.add_argument("--judge-model", default="databricks:/databricks-claude-opus-5-5",
                        help="Explicit judge model URI (default: %(default)s)")
    parser.add_argument("--model", help="Optional filter on the answer-generation model tag")
    parser.add_argument("--latest-per-uid", action="store_true",
                        help="Score only the latest successful trace for each question UID")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    config = json.loads(args.tracing_config.read_text())
    if config["trackingUri"] != f"databricks://{args.profile}":
        parser.error("Explicit --profile does not match the tracing config")
    warehouse = (args.warehouse_id or os.environ.get("MLFLOW_TRACING_SQL_WAREHOUSE_ID")
                 or dotenv_values(".env").get("MLFLOW_TRACING_SQL_WAREHOUSE_ID"))
    if config.get("traceLocation") and not warehouse:
        parser.error("UC traces require --warehouse-id or MLFLOW_TRACING_SQL_WAREHOUSE_ID")
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    if warehouse:
        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(args.workers)
    snapshot_path = args.dataset_dir / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {
        "dataset": "databricks/officeqa-pro-v2", "revision": args.revision,
    }
    if not snapshot.get("revision"):
        parser.error("Pass --revision with the downloaded dataset commit; snapshot.json is missing")
    if args.revision and args.revision != snapshot["revision"]:
        parser.error("Explicit --revision does not match snapshot.json")
    references = load_references(args.dataset_dir / "officeqa_pro_v2.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".score.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another evaluation is using this output directory")
        manifest = {"dataset": snapshot["dataset"], "revision": snapshot["revision"],
                    "tracking_uri": config["trackingUri"], "experiment_id": str(config["experimentId"]),
                    "judge_model": args.judge_model, "model_filter": args.model}
        if args.latest_per_uid:
            manifest["latest_per_uid"] = True
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            parser.error("Output directory belongs to a different dataset, judge, or destination")
        write_json(manifest_path, manifest)
        summary_path = args.output_dir / "summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        completed = previous.get("assessments", {})
        mlflow.set_tracking_uri(config["trackingUri"])
        mlflow.set_experiment(experiment_id=str(config["experimentId"]))

        def search(include_spans=False):
            traces = mlflow.search_traces(
                locations=[str(config["experimentId"])],
                filter_string=f"tags.dataset = '{snapshot['dataset']}'",
                max_results=10000, return_type="list", include_spans=include_spans,
            )
            if len(traces) == 10000:
                raise ValueError("Trace search reached its result limit; narrow the selection")
            return traces

        candidates = []
        for trace in search():
            tags = trace.info.tags
            if tags.get("execution_client") != "codex_cli" or str(trace.info.state) != "OK":
                continue
            if args.model and tags.get("model") != args.model:
                continue
            if tags.get("dataset_revision") != snapshot["revision"] or tags.get("uid") not in references:
                raise ValueError(f"Unmatched reference dataset or uid: {trace.info.trace_id}")
            candidates.append(trace)
        candidates.sort(key=lambda trace: (trace.info.tags["uid"], trace.info.trace_id))
        if args.latest_per_uid:
            candidates = latest_traces_by_uid(candidates)
            candidate_ids = {trace.info.trace_id for trace in candidates}
            completed = {trace_id: assessment for trace_id, assessment in completed.items()
                         if trace_id in candidate_ids}
        pending = [trace for trace in candidates if trace.info.trace_id not in completed]
        selected = pending[:args.limit] if args.limit else pending
        print(f"Matched {len(candidates)} traces; already scored={len(candidates) - len(pending)}, "
              f"selected={len(selected)}, judge={args.judge_model}", flush=True)
        if not candidates:
            raise ValueError("No matching OfficeQA traces found")
        if not selected:
            print("All selected traces already have verified correctness assessments", flush=True)
            return
        print(f"Loading {len(selected)} traces for evaluation", flush=True)
        if len(selected) > 10:
            fetched = {trace.info.trace_id: trace for trace in search(include_spans=True)}
            full = [fetched.get(trace.info.trace_id) for trace in selected]
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                full = list(executor.map(lambda trace: mlflow.get_trace(trace.info.trace_id), selected))
        if any(trace is None for trace in full):
            raise ValueError("A selected trace could not be retrieved")
        rows = [evaluation_row(trace, references) for trace in full]
        with mlflow.start_run(
            experiment_id=str(config["experimentId"]), run_name="officeqa-correctness",
            tags={"dataset": snapshot["dataset"], "dataset_revision": snapshot["revision"],
                  "execution_client": "codex_cli", "judge_model": args.judge_model},
        ):
            result = mlflow.genai.evaluate(data=rows, scorers=[Correctness(model=args.judge_model)])
        run_path = args.output_dir / f"run-{result.run_id}.json"
        write_json(run_path, {"run_id": result.run_id, "metrics": result.metrics,
                             "rows": json.loads(result.result_df.drop(columns=["trace"], errors="ignore")
                                                .to_json(orient="records", default_handler=str))})
        expected = {row["trace"].info.trace_id: row["expectations"]["expected_facts"] for row in rows}
        deadline = time.monotonic() + 180
        outstanding = set(expected)
        while outstanding:
            for trace in search():
                trace_id = trace.info.trace_id
                if trace_id not in outstanding:
                    continue
                feedback = successful_assessment(
                    trace.info, expected[trace_id], result.run_id, args.judge_model,
                )
                if feedback is not None:
                    completed[trace_id] = {"uid": trace.info.tags["uid"],
                                           "model": trace.info.tags.get("model", "provider-default"),
                                           "value": feedback.value, "rationale": feedback.rationale,
                                           "judge_model": feedback.source.source_id,
                                           "assessment_id": feedback.assessment_id,
                                           "expected_facts": expected[trace_id], "run_id": result.run_id}
                    outstanding.remove(trace_id)
            if outstanding:
                if time.monotonic() >= deadline:
                    break
                time.sleep(5)
        summary = {**manifest, "available_traces": len(candidates), "scored": len(completed),
                   "correct": sum(row["value"] == "yes" for row in completed.values()),
                   "incorrect": sum(row["value"] == "no" for row in completed.values()),
                   "unverified_trace_ids": sorted(outstanding), "assessments": completed}
        write_json(summary_path, summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "assessments"}), flush=True)
        if outstanding:
            raise RuntimeError(f"{len(outstanding)} assessments failed or could not be verified; "
                               f"inspect {run_path}")


if __name__ == "__main__":
    main()
