"""Airflow DAG (bonus) - nightly LLM regression suite for the assistant.

Schedule : daily (``@daily``), catchup off.
Tasks    : run_regression_suite -> evaluate_verdict -> branch ->
           [notify_failure | regression_healthy]

What it runs
    ``run_regression_suite`` calls ``mlops.regression.main`` in-process, which:
      1. runs the 6-query golden set through the agentic loop with the pinned
         production prompt (``PROMPT_VERSION``),
      2. grades it with the W16 harness graders,
      3. runs the two Evidently LLM-as-judge checks (faithfulness of the
         answer to the evidence, answer actually addresses the question) and
         logs ``pct_tests_passed`` + the HTML report to MLflow
         (experiment ``agentify-the-assistant``, tag ``kind=llm_regression``).

Trigger condition
    ``evaluate_verdict`` reads the newest ``kind=llm_regression`` run and
    branches to ``notify_failure`` when
    ``metrics.pct_tests_passed < params.fail_under`` (default 1.0, i.e. both
    judge tests must pass their share threshold), else to ``regression_healthy``.

Not installed locally (Airflow is an optional bonus): validate with
    python -m py_compile dags/nightly_regression_dag.py
and drop ``dags/`` into an Airflow ``dags_folder``.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

TRACK_B_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TRACK_B_DIR not in sys.path:  # make mlops/ importable from the dags folder
    sys.path.insert(0, TRACK_B_DIR)

PROMPT_VERSION = "v3"  # production prompt pinned by the W17 experiments
FAIL_UNDER = 1.0


def run_regression_suite(**context) -> None:
    """Execute the golden-set + Evidently judge suite as a Python task."""
    from mlops.regression import main

    exit_code = main(["--prompt", PROMPT_VERSION])
    context["ti"].xcom_push(key="exit_code", value=int(exit_code))
    if exit_code != 0:
        raise RuntimeError(f"regression suite returned non-zero exit code {exit_code}")


def evaluate_verdict(**context) -> str:
    """Read the newest LLM regression run from MLflow and pick the follow-up."""
    import mlflow

    from mlops.tracking import EXPERIMENT_NAME, TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return "regression_healthy"

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kind = 'llm_regression'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not len(runs):
        return "regression_healthy"

    latest = runs.iloc[0]
    pct = float(latest.get("metrics.pct_tests_passed", 0.0) or 0.0)
    fail_under = float(latest.get("params.fail_under", FAIL_UNDER) or FAIL_UNDER)
    print(f"latest regression run={latest['run_id']} pct_tests_passed={pct} (>= {fail_under}?)")
    context["ti"].xcom_push(key="pct_tests_passed", value=pct)
    return "regression_healthy" if pct >= fail_under else "notify_failure"


def log_failure(**context) -> None:
    pct = context["ti"].xcom_pull(task_ids="evaluate_verdict", key="pct_tests_passed")
    print(
        f"REGRESSION FAILED: pct_tests_passed={pct}. "
        "Inspect the newest kind=llm_regression MLflow run "
        "(judge_test_status.json + reports/) and compare prompt versions "
        "before promoting a prompt to production."
    )


def log_healthy(**context) -> None:
    pct = context["ti"].xcom_pull(task_ids="evaluate_verdict", key="pct_tests_passed")
    print(f"all judge tests passed (pct_tests_passed={pct}); next run in 1 day")


default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="assistant_nightly_regression",
    description="Nightly golden-set + Evidently LLM judge regression for the research agent",
    schedule="@daily",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "evidently", "llm-judge", "regression", "track-b"],
) as dag:
    task_run_suite = PythonOperator(
        task_id="run_regression_suite",
        python_callable=run_regression_suite,
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
    task_notify_failure = PythonOperator(
        task_id="notify_failure",
        python_callable=log_failure,
    )
    task_healthy = PythonOperator(
        task_id="regression_healthy",
        python_callable=log_healthy,
    )

    task_run_suite >> task_evaluate >> task_branch
    task_branch >> [task_notify_failure, task_healthy]
