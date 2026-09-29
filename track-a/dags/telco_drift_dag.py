"""Airflow DAG (bonus) - scheduled churn drift check with retrain recommendation.

Schedule : weekly, Monday 06:00 (``@weekly``), catchup off.
Tasks    : run_drift_check -> evaluate_verdict -> branch -> [trigger_retrain ->
           retrain_pipeline | drift_healthy]
           (run_drift_check is the PythonOperator equivalent of
           ``uv run python -m churn_mlops.monitor``)

Trigger condition
    The drift check logs ``retrain_recommended`` (0/1) to the
    ``telco-churn-drift-monitoring`` MLflow experiment. It is 1 when either
      * share of drifted features >= DRIFT_SHARE_THRESHOLD (0.10), or
      * one of the custom Evidently drift tests failed
        (mean MonthlyCharges shift >= $5 or month-to-month churn-rate shift >= 3pp).

What happens on a positive signal
    ``trigger_retrain`` re-runs the full training pipeline
    (``uv run python -m churn_mlops.train``), which retrains all model specs,
    registers the new best model and promotes it Staging -> Production
    (the previous Production version is archived by the registry transition).
    On a negative signal the DAG just records the healthy verdict in the task
    log/XCom, so the weekly history stays auditable.

Not installed locally (Airflow is an optional bonus): validate with
    python -m py_compile dags/telco_drift_dag.py
and drop ``dags/`` into an Airflow ``dags_folder``.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

TRACK_A_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TRACK_A_DIR not in sys.path:  # make churn_mlops importable from the dags folder
    sys.path.insert(0, TRACK_A_DIR)

RETRAIN_COMMAND = f"cd {TRACK_A_DIR} && uv run python -m churn_mlops.train"
RETRAIN_RECOMMENDED_METRIC = "retrain_recommended"


def run_drift_check(**context) -> None:
    """Execute the Evidently drift check as a normal Python task (no shell)."""
    from churn_mlops.monitor import run_monitoring

    summary = run_monitoring()
    context["ti"].xcom_push(key="retrain_recommended", value=int(summary["retrain_recommended"]))
    context["ti"].xcom_push(key="retrain_reason", value=summary["retrain_reason"])


def evaluate_verdict(**context) -> str:
    """Read the newest drift run from MLflow and pick the follow-up task."""
    import mlflow

    from churn_mlops import common as C

    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(C.DRIFT_EXPERIMENT)
    if experiment is None:
        return "drift_healthy"

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
    )
    if not len(runs):
        return "drift_healthy"

    latest = runs.iloc[0]
    recommended = int(latest.get(f"metrics.{RETRAIN_RECOMMENDED_METRIC}", 0) or 0)
    drift_share = latest.get("metrics.drift_share")
    print(
        f"latest drift run={latest['run_id']} drift_share={drift_share} "
        f"retrain_recommended={recommended}"
    )
    context["ti"].xcom_push(key="drift_share", value=drift_share)
    return "trigger_retrain" if recommended else "drift_healthy"


def log_healthy(**context) -> None:
    drift_share = context["ti"].xcom_pull(task_ids="evaluate_verdict", key="drift_share")
    print(f"no action needed (drift_share={drift_share}); next check in 1 week")


def log_retrain(**context) -> None:
    reason = context["ti"].xcom_pull(task_ids="run_drift_check", key="retrain_reason")
    print(f"DRIFT SIGNAL: retraining triggered. reason={reason}")


default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="telco_churn_drift_monitor",
    description="Weekly Evidently drift check on Telco Churn data; retrain on signal",
    schedule="@weekly",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "drift", "evidently", "track-a"],
) as dag:
    task_drift_check = PythonOperator(
        task_id="run_drift_check",
        python_callable=run_drift_check,
    )
    task_evaluate = PythonOperator(
        task_id="evaluate_verdict",
        python_callable=evaluate_verdict,
    )
    task_branch = BranchPythonOperator(
        task_id="branch_on_verdict",
        python_callable=lambda **context: context["ti"].xcom_pull(
            task_ids="evaluate_verdict"
        ),
    )
    task_trigger_retrain = PythonOperator(
        task_id="trigger_retrain",
        python_callable=log_retrain,
    )
    task_retrain_pipeline = BashOperator(
        task_id="retrain_pipeline",
        bash_command=RETRAIN_COMMAND,
    )
    task_drift_healthy = PythonOperator(
        task_id="drift_healthy",
        python_callable=log_healthy,
    )

    task_drift_check >> task_evaluate >> task_branch
    task_branch >> [task_trigger_retrain, task_drift_healthy]
    task_trigger_retrain >> task_retrain_pipeline
