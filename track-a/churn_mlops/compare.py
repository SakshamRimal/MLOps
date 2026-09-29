"""Export MLflow run comparisons and registry state to versioned files.

Writes:
    artifacts/run_comparison.md     - side-by-side metrics table for every training run
    artifacts/run_comparison.csv    - the same table as CSV
    artifacts/registry_status.md    - registered model versions + stages
    artifacts/drift_check_history.md- drift-check runs and their verdicts

Usage:
    uv run python -m churn_mlops.compare
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from churn_mlops import common as C

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "fit_seconds",
]


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def export_training_comparison() -> pd.DataFrame:
    import mlflow

    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(C.TRAINING_EXPERIMENT)
    if experiment is None:
        raise SystemExit("no training experiment yet - run `uv run python -m churn_mlops.train`")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
    )
    # keep only model training runs (the registry-promotion run has no model metrics)
    runs = runs[runs["tags.mlflow.runName"] != "registry-promotion"].copy()
    runs = runs[runs["metrics.f1"].notna()].copy()

    param_cols = [
        c
        for c in runs.columns
        if c.startswith("params.")
        and c.removeprefix("params.") != "model_run_name"  # duplicates run_name
        and runs[c].notna().any()  # drop params only set on other runs
    ]
    table = runs[["run_id", "tags.mlflow.runName"] + param_cols + [
        f"metrics.{m}" for m in METRIC_COLUMNS
    ]].copy()
    table.columns = (
        ["run_id", "run_name"]
        + [c.removeprefix("params.") for c in param_cols]
        + METRIC_COLUMNS
    )
    table = table.sort_values("f1", ascending=False).reset_index(drop=True)

    best = table.iloc[0]
    table.insert(
        1,
        "registered",
        ["YES <- best (f1, tie-break roc_auc)" if r == best["run_name"] else "" for r in table["run_name"]],
    )

    C.EXPORTS_DIR.mkdir(exist_ok=True)
    table.to_csv(C.EXPORTS_DIR / "run_comparison.csv", index=False)

    md = [
        "# Track A - MLflow run comparison (telco-churn-training)",
        "",
        f"All runs ranked by {C.SELECTION_METRIC} (tie-break {C.TIE_BREAK_METRIC}).",
        "",
        _markdown_table(table.round(4)),
        "",
        f"**Registered:** `{best['run_name']}` (run `{best['run_id']}`) "
        f"-> `{C.MODEL_NAME}` Staging -> Production.",
        "",
        "Reproduce: `uv run mlflow ui --port 5000` then open the `telco-churn-training` experiment.",
    ]
    (C.EXPORTS_DIR / "run_comparison.md").write_text("\n".join(md))
    print(f"wrote {C.EXPORTS_DIR / 'run_comparison.md'} ({len(table)} runs)")
    return table


def export_registry_status() -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{C.MODEL_NAME}'")
    except Exception as exc:  # registry empty before the first registration
        versions = []
        print(f"registry unavailable: {exc}")

    rows = []
    for v in sorted(versions, key=lambda x: int(x.version)):
        rows.append(
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status,
                "creation_timestamp": pd.to_datetime(v.creation_timestamp, unit="ms").isoformat()
                if v.creation_timestamp
                else "",
            }
        )
    frame = pd.DataFrame(rows, columns=["version", "stage", "run_id", "status", "creation_timestamp"])

    transition_file = C.EXPORTS_DIR / "registry_transitions.json"
    transitions = json.loads(transition_file.read_text()) if transition_file.exists() else {}

    md = [
        "# Track A - Model registry status",
        "",
        f"Model: `{C.MODEL_NAME}`",
        "",
        _markdown_table(frame) if len(frame) else "_no versions registered yet_",
        "",
        "## Recorded stage transitions (from training run)",
        "",
        "```json",
        json.dumps(transitions, indent=2),
        "```",
    ]
    C.EXPORTS_DIR.mkdir(exist_ok=True)
    (C.EXPORTS_DIR / "registry_status.md").write_text("\n".join(md))
    print(f"wrote {C.EXPORTS_DIR / 'registry_status.md'} ({len(frame)} versions)")


def export_drift_history() -> None:
    import mlflow

    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(C.DRIFT_EXPERIMENT)
    if experiment is None:
        return
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
    )
    if not len(runs):
        return

    time_col = next(
        (c for c in ("start_time", "attributes.start_time") if c in runs.columns), None
    )
    keep = [
        c
        for c in [
            "tags.mlflow.runName",
            time_col,
            "metrics.drift_share",
            "metrics.n_drifted_columns",
            "metrics.churn_rate_shift",
            "metrics.target_drift_detected",
            "metrics.mean_monthly_charges_shift",
            "metrics.mtm_churn_rate_shift",
            "metrics.custom_metric_tests_passed",
            "metrics.retrain_recommended",
            "params.synthetic_drift_injected",
        ]
        if c and c in runs.columns
    ]
    table = runs[keep].copy()
    table.columns = [
        c.removeprefix("metrics.")
        .removeprefix("params.")
        .removeprefix("tags.mlflow.")
        .removeprefix("attributes.")
        for c in keep
    ]
    if time_col:
        renamed_time = time_col.removeprefix("attributes.")
        table[renamed_time] = pd.to_datetime(table[renamed_time], unit="ms", utc=True).dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        table = table.rename(columns={renamed_time: "started"})

    md = [
        "# Track A - Drift check history (telco-churn-drift-monitoring)",
        "",
        _markdown_table(table),
    ]
    C.EXPORTS_DIR.mkdir(exist_ok=True)
    (C.EXPORTS_DIR / "drift_check_history.md").write_text("\n".join(md))
    print(f"wrote {C.EXPORTS_DIR / 'drift_check_history.md'} ({len(table)} runs)")


def main() -> None:
    export_training_comparison()
    export_registry_status()
    export_drift_history()


if __name__ == "__main__":
    main()
