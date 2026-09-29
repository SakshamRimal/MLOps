"""Shared configuration, MLflow setup and data helpers for Track A."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

TRACK_A_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = TRACK_A_DIR / "data" / "Telco-Customer-Churn.csv"
MLFLOW_DB = TRACK_A_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB}")
ARTIFACT_ROOT = TRACK_A_DIR / "mlruns"
REPORTS_DIR = TRACK_A_DIR / "reports"
EXPORTS_DIR = TRACK_A_DIR / "artifacts"
DAGS_DIR = TRACK_A_DIR / "dags"

TRAINING_EXPERIMENT = "telco-churn-training"
DRIFT_EXPERIMENT = "telco-churn-drift-monitoring"
MODEL_NAME = "TelcoChurnClassifier"

RANDOM_SEED = 42
TARGET = "Churn"
ID_COL = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
POS_LABEL = "Yes"

# The metric that decides which run gets registered (documented in README):
# the target is ~27% positive and accuracy is misleading on it, so we rank by
# F1 (precision/recall balance on the churn class) with ROC-AUC as tie-break.
SELECTION_METRIC = "f1"
TIE_BREAK_METRIC = "roc_auc"


def setup_mlflow(experiment: str = TRAINING_EXPERIMENT) -> None:
    """Point MLflow at this project's sqlite backend + local artifact root.

    MLflow >= 3 refuses new filesystem tracking backends, so the tracking
    store is a sqlite database (`mlflow.db`) while artifacts (models, plots,
    HTML reports) are written to `mlruns/` next to it. Both paths are
    absolute, so scripts behave the same regardless of the CWD.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    existing = client.get_experiment_by_name(experiment)
    if existing is None:
        client.create_experiment(experiment, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(experiment)


def load_churn_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the Telco Churn CSV with the cleaning the pipeline actually needs."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = df[TARGET].astype(str)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (TARGET, ID_COL)]


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split so the churn rate matches in both frames."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )


def build_preprocessor(features: list[str]):
    """ColumnTransformer: scale numerics, one-hot encode categoricals."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c not in NUMERIC_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Accuracy + the imbalance-aware metrics required by the assignment."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_bin = (pd.Series(y_true) == POS_LABEL).astype(int).to_numpy()
    pred_bin = (pd.Series(y_pred) == POS_LABEL).astype(int).to_numpy()
    return {
        "accuracy": float(accuracy_score(y_bin, pred_bin)),
        "precision": float(precision_score(y_bin, pred_bin, zero_division=0)),
        "recall": float(recall_score(y_bin, pred_bin, zero_division=0)),
        "f1": float(f1_score(y_bin, pred_bin, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_bin, y_proba)),
        "average_precision": float(average_precision_score(y_bin, y_proba)),
    }


def confusion_matrix_figure(y_true, y_pred):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_bin = (pd.Series(y_true) == POS_LABEL).astype(int).to_numpy()
    pred_bin = (pd.Series(y_pred) == POS_LABEL).astype(int).to_numpy()
    cm = confusion_matrix(y_bin, pred_bin)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(
        title="Confusion matrix (churn = Yes is the positive class)",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=[0, 1],
        yticks=[0, 1],
    )
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, int(value), ha="center", va="center", color="black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def roc_curve_figure(y_true, y_proba, label: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    y_bin = (pd.Series(y_true) == POS_LABEL).astype(int).to_numpy()
    fpr, tpr, _ = roc_curve(y_bin, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], lw=1, ls="--", color="grey", label="chance")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve (held-out test set)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
