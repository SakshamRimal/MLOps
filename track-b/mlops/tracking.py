"""Shared MLflow tracking setup for Track B (W17).

One sqlite-backed tracking URI per project so runs persist in version control
just like Track A: ``track-b/mlflow.db`` (backend) + ``track-b/mlruns/`` (artifacts).

All Track B runs live in a single experiment and are distinguished by the
``kind`` tag: ``prompt_experiment`` (experiments.py) and ``llm_regression``
(regression.py).
"""
from __future__ import annotations

from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "agentify-the-assistant"

# mlflow >= 2.17 logs model versions into the "models" table of the tracking DB;
# the sqlite file must therefore exist (it is created by MLflow on first use).
TRACKING_URI = f"sqlite:///{MLFLOW_DB}"


def init_tracking(experiment_name: str = EXPERIMENT_NAME) -> str:
    """Point MLflow at this project's sqlite backend and resolve the experiment."""
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    return experiment_name
