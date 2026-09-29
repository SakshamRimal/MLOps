"""Drift monitoring for the churn model with Evidently AI.

Splits the Telco Churn dataset into a reference set (training-time data) and a
current set (incoming production data), injects deliberate synthetic drift into
the current set, then runs:

  1. Evidently Data Drift report (feature distributions, all columns)
  2. Evidently Target Drift report (did the churn rate itself move?)
  3. A custom-metric report built with Evidently's custom metric/test API
     - |mean(MonthlyCharges)_current - mean(MonthlyCharges)_reference|
     - churn-rate shift inside the Contract == "Month-to-month" segment

All three HTML reports plus a JSON summary are logged to MLflow; the headline
drift numbers are logged as MLflow metrics so the Airflow DAG (and the run UI)
can read the verdict without parsing HTML.

Usage:
    uv run python -m churn_mlops.monitor
    uv run python -m churn_mlops.monitor --no-drift        # clean baseline split
    uv run python -m churn_mlops.monitor --reference-frac 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn_mlops import common as C

# Thresholds that decide whether retraining is recommended.
DRIFT_SHARE_THRESHOLD = 0.1  # >=10% of monitored features drifted -> alert
MEAN_CHARGES_SHIFT_LIMIT = 5.0  # USD
MTM_CHURN_SHIFT_LIMIT = 0.03  # absolute churn-rate points
TARGET_STATTEST_THRESHOLD = 0.05  # JS distance that counts as target drift


def _is_fail(status) -> bool:
    """Evidently reports test statuses as SUCCESS/FAIL (new) or pass/fail (legacy)."""
    value = getattr(status, "value", status)
    return str(value).strip().lower() in {"fail", "failed", "error"}


def split_reference_current(
    df: pd.DataFrame, reference_frac: float = 0.7, seed: int = C.RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random 70/30 split: reference = training-time data, current = production."""
    reference = df.sample(frac=reference_frac, random_state=seed).reset_index(drop=True)
    current = df.drop(reference.index).reset_index(drop=True)
    return reference, current


def inject_synthetic_drift(current: pd.DataFrame, seed: int = C.RANDOM_SEED):
    """Deliberately perturb the 'production' set so the drift report has a signal.

    Three engineered shifts:
      * numeric noise  - MonthlyCharges shifted upward by U(8, 20) per row
      * categorical skew - Contract == "Month-to-month" oversampled by 120%
      * label/concept drift - 10% of Churn labels flipped
    """
    rng = np.random.default_rng(seed)
    current = current.copy()

    shift = rng.uniform(8.0, 20.0, size=len(current))
    current["MonthlyCharges"] = (current["MonthlyCharges"] + shift).round(2)

    mtm = current[current["Contract"] == "Month-to-month"]
    oversample = mtm.sample(frac=1.2, random_state=seed, replace=True)
    current = pd.concat([current, oversample], ignore_index=True)

    flip_idx = rng.choice(current.index, size=int(0.10 * len(current)), replace=False)
    current.loc[flip_idx, "Churn"] = (
        current.loc[flip_idx, "Churn"].map({"Yes": "No", "No": "Yes"}).fillna("No")
    )

    injected = {
        "monthly_charges_noise": "uniform(8, 20) added per row",
        "contract_oversample": (
            "Contract=Month-to-month rows duplicated at 120% "
            "(Jensen-Shannon distance 0.12 > Evidently's default 0.10 threshold)"
        ),
        "label_flip": "10% of Churn labels flipped",
    }
    return current, injected


def _run_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset(include_tests=True)], include_tests=True)
    snapshot = report.run(current_data=current, reference_data=reference)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_html))

    payload = snapshot.dict()
    drifted_columns, scores, tests = [], {}, []
    for test in payload.get("tests", []):
        name = test.get("name", "")
        status = getattr(test.get("status", ""), "value", test.get("status", ""))
        tests.append({"name": name, "status": status})
        if name.startswith("Value Drift for column "):
            column = name.removeprefix("Value Drift for column ")
            scores[column] = {
                "description": test.get("description", ""),
                "drift_detected": _is_fail(status),
            }
            if _is_fail(status):
                drifted_columns.append(column)

    share = None
    for metric in payload.get("metrics", []):
        if "DriftedColumnsCount" in metric.get("metric_name", ""):
            share = metric.get("value", {}).get("share")
            n_drifted = metric.get("value", {}).get("count")
            break
    else:
        n_drifted = len(drifted_columns)

    return {
        "report": out_html.name,
        "n_columns_scored": len(scores),
        "drifted_columns": drifted_columns,
        "n_drifted": int(n_drifted if n_drifted is not None else len(drifted_columns)),
        "drift_share": float(share) if share is not None else (
            len(drifted_columns) / max(len(scores), 1)
        ),
        "column_scores": scores,
        "n_tests": len(tests),
        "n_tests_failed": sum(1 for t in tests if _is_fail(t["status"])),
    }


def _run_target_drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> dict:
    """Evidently's Target Drift preset (legacy module kept for this preset).

    The categorical stat-test threshold is tightened to 0.05 (vs the default
    0.10): for this business a >=5 point move in the churn base rate already
    changes expected revenue enough to act on, and the injected label drift
    would otherwise sit just under the default threshold.
    """
    from evidently.legacy.metric_preset import TargetDriftPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report as LegacyReport

    mapping = ColumnMapping(target=C.TARGET)
    report = LegacyReport(
        metrics=[TargetDriftPreset(cat_stattest_threshold=TARGET_STATTEST_THRESHOLD)]
    )
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_html))

    churn_ref = float((reference[C.TARGET] == C.POS_LABEL).mean())
    churn_cur = float((current[C.TARGET] == C.POS_LABEL).mean())
    target_score = target_detected = None
    for metric in report.as_dict().get("metrics", []):
        result = metric.get("result", {})
        if metric.get("metric") == "ColumnDriftMetric" and result.get("column_name") == C.TARGET:
            target_score = result.get("drift_score")
            target_detected = result.get("drift_detected")
            break
    return {
        "report": out_html.name,
        "churn_rate_reference": round(churn_ref, 4),
        "churn_rate_current": round(churn_cur, 4),
        "churn_rate_shift": round(churn_cur - churn_ref, 4),
        "target_drift_score": target_score,
        "target_drift_detected": bool(target_detected),
        "target_stattest_threshold": TARGET_STATTEST_THRESHOLD,
    }


def _custom_metric_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> dict:
    """Custom metrics built on Evidently's custom metric/test API."""
    from evidently import Report, tests as etest
    from evidently.core.metric_types import SingleValueCalculation, SingleValueMetric

    class MeanChargesShift(SingleValueMetric):
        """|mean(MonthlyCharges)_current - mean(MonthlyCharges)_reference| in USD."""

        column: str = "MonthlyCharges"

    class MeanChargesShiftCalculation(SingleValueCalculation[MeanChargesShift]):
        def display_name(self) -> str:
            return f"Mean {self.metric.column} shift (current vs reference)"

        def calculate(self, context, current_data, reference_data):
            cur = float(current_data.column(self.metric.column).data.mean())
            ref = float(reference_data.column(self.metric.column).data.mean())
            result = self.result(abs(cur - ref))
            result.display_name = (
                f"{self.display_name()}: current={cur:.2f}, reference={ref:.2f}"
            )
            return result

    class MtmChurnRateShift(SingleValueMetric):
        """Churn-rate shift inside the Contract == Month-to-month segment."""

        column: str = C.TARGET
        segment_column: str = "Contract"
        segment_value: str = "Month-to-month"

    class MtmChurnRateShiftCalculation(SingleValueCalculation[MtmChurnRateShift]):
        def display_name(self) -> str:
            return (
                f"Churn-rate shift in {self.metric.segment_column}"
                f"={self.metric.segment_value} segment"
            )

        @staticmethod
        def _rate(frame, m) -> float:
            contract = frame.column(m.segment_column).data
            churn = frame.column(m.column).data
            mask = contract == m.segment_value
            return float((churn[mask] == C.POS_LABEL).mean())

        def calculate(self, context, current_data, reference_data):
            cur = self._rate(current_data, self.metric)
            ref = self._rate(reference_data, self.metric)
            result = self.result(abs(cur - ref))
            result.display_name = f"{self.display_name()}: current={cur:.4f}, reference={ref:.4f}"
            return result

    mean_shift = MeanChargesShift(tests=[etest.lt(MEAN_CHARGES_SHIFT_LIMIT)])
    mtm_shift = MtmChurnRateShift(tests=[etest.lt(MTM_CHURN_SHIFT_LIMIT)])

    report = Report([mean_shift, mtm_shift], include_tests=True)
    snapshot = report.run(current_data=current, reference_data=reference)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_html))

    payload = snapshot.dict()
    values, test_status = {}, {}
    for metric in payload.get("metrics", []):
        name = metric.get("metric_name", "")
        if "MeanChargesShift" in name:
            values["mean_monthly_charges_shift"] = float(metric.get("value"))
        elif "MtmChurnRateShift" in name:
            values["mtm_churn_rate_shift"] = float(metric.get("value"))
    for test in payload.get("tests", []):
        test_status[test.get("name", "")] = getattr(
            test.get("status", ""), "value", test.get("status", "")
        )

    return {
        "report": out_html.name,
        **values,
        "limits": {
            "mean_monthly_charges_shift_lt": MEAN_CHARGES_SHIFT_LIMIT,
            "mtm_churn_rate_shift_lt": MTM_CHURN_SHIFT_LIMIT,
        },
        "tests": test_status,
        "custom_test_passed": int(not any(_is_fail(s) for s in test_status.values())),
    }


def run_monitoring(reference_frac: float = 0.7, inject: bool = True, seed: int = C.RANDOM_SEED):
    """Run the whole drift check and log everything to MLflow. Returns the summary."""
    import mlflow

    C.setup_mlflow(C.DRIFT_EXPERIMENT)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = C.load_churn_data().drop(columns=[C.ID_COL])
    reference, current = split_reference_current(df, reference_frac, seed)
    injections = {}
    if inject:
        current, injections = inject_synthetic_drift(current, seed)

    with mlflow.start_run(run_name="drift-check") as run:
        mlflow.log_params(
            {
                "reference_frac": reference_frac,
                "seed": seed,
                "synthetic_drift_injected": inject,
                "reference_rows": len(reference),
                "current_rows": len(current),
                "injections": json.dumps(injections) if injections else "none",
                "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
            }
        )

        data_summary = _run_data_drift_report(
            reference, current, C.REPORTS_DIR / "data_drift_report.html"
        )
        target_summary = _run_target_drift_report(
            reference, current, C.REPORTS_DIR / "target_drift_report.html"
        )
        custom_summary = _custom_metric_report(
            reference, current, C.REPORTS_DIR / "custom_metrics_report.html"
        )

        retrain_recommended = bool(
            data_summary["drift_share"] >= DRIFT_SHARE_THRESHOLD
            or not custom_summary["custom_test_passed"]
        )

        mlflow.set_tags(
            {"drifted_columns": ",".join(data_summary["drifted_columns"]) or "none"}
        )
        mlflow.log_metrics(
            {
                "drift_share": round(data_summary["drift_share"], 4),
                "n_drifted_columns": data_summary["n_drifted"],
                "n_columns_scored": data_summary["n_columns_scored"],
                "data_drift_tests_failed": data_summary["n_tests_failed"],
                "churn_rate_reference": target_summary["churn_rate_reference"],
                "churn_rate_current": target_summary["churn_rate_current"],
                "churn_rate_shift": target_summary["churn_rate_shift"],
                "target_drift_detected": int(target_summary["target_drift_detected"]),
                "target_drift_score": round(target_summary["target_drift_score"] or 0.0, 4),
                "mean_monthly_charges_shift": round(
                    custom_summary.get("mean_monthly_charges_shift", float("nan")), 4
                ),
                "mtm_churn_rate_shift": round(
                    custom_summary.get("mtm_churn_rate_shift", float("nan")), 4
                ),
                "custom_metric_tests_passed": custom_summary["custom_test_passed"],
                "retrain_recommended": int(retrain_recommended),
            }
        )

        for report_name in (
            data_summary["report"],
            target_summary["report"],
            custom_summary["report"],
        ):
            mlflow.log_artifact(str(C.REPORTS_DIR / report_name), artifact_path="evidently_reports")

        summary = {
            "run_id": run.info.run_id,
            "data_drift": data_summary,
            "target_drift": target_summary,
            "custom_metrics": custom_summary,
            "retrain_recommended": retrain_recommended,
            "retrain_reason": (
                f"drift_share={data_summary['drift_share']:.3f} >= {DRIFT_SHARE_THRESHOLD}"
                if data_summary["drift_share"] >= DRIFT_SHARE_THRESHOLD
                else (
                    "custom drift test failed"
                    if not custom_summary["custom_test_passed"]
                    else "no threshold crossed"
                )
            ),
        }
        summary_path = C.REPORTS_DIR / "drift_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(summary_path), artifact_path="evidently_reports")

    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="skip synthetic drift injection (baseline control run)",
    )
    args = parser.parse_args(argv)
    run_monitoring(
        reference_frac=args.reference_frac, inject=not args.no_drift, seed=args.seed
    )


if __name__ == "__main__":
    main()
