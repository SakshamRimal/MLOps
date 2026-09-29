"""Train several churn models, track every run in MLflow, register the best one.

Usage:
    uv run python -m churn_mlops.train
    uv run python -m churn_mlops.train --no-registry
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from churn_mlops import common as C


@dataclass(frozen=True)
class ModelSpec:
    """One tracked training run. `params` are the genuinely varied hyperparameters."""

    run_name: str
    family: str
    params: dict
    build: object  # callable(**params) -> estimator
    tags: dict = field(default_factory=dict)


def _logreg(**params):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**params)


def _forest(**params):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**params, n_jobs=-1, random_state=C.RANDOM_SEED)


def _hgb(**params):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(**params, random_state=C.RANDOM_SEED)


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        run_name="logreg_C0.1",
        family="logistic_regression",
        params={"C": 0.1, "max_iter": 1000, "solver": "lbfgs"},
        build=_logreg,
        tags={"regularization": "strong"},
    ),
    ModelSpec(
        run_name="logreg_C10",
        family="logistic_regression",
        params={"C": 10.0, "max_iter": 1000, "solver": "lbfgs"},
        build=_logreg,
        tags={"regularization": "weak"},
    ),
    ModelSpec(
        run_name="rf_depth6",
        family="random_forest",
        params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 10},
        build=_forest,
        tags={"depth_tuned": "shallow"},
    ),
    ModelSpec(
        run_name="rf_depth16",
        family="random_forest",
        params={"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 2},
        build=_forest,
        tags={"depth_tuned": "deep"},
    ),
    ModelSpec(
        run_name="hgb_lr0.08",
        family="hist_gradient_boosting",
        params={"max_iter": 300, "learning_rate": 0.08, "max_leaf_nodes": 31},
        build=_hgb,
        tags={"boosting": "default_leaves"},
    ),
]


def train_one(spec: ModelSpec, train_df, test_df) -> dict:
    """Fit one pipeline, log params/metrics/artifacts to MLflow, return a summary."""
    from sklearn.pipeline import Pipeline

    from churn_mlops.common import build_preprocessor, classification_metrics

    features = C.feature_columns(train_df)
    X_train, y_train = train_df[features], train_df[C.TARGET]
    X_test, y_test = test_df[features], test_df[C.TARGET]

    pipeline = Pipeline(
        [("preprocess", build_preprocessor(features)), ("model", spec.build(**spec.params))]
    )

    started = time.perf_counter()
    with mlflow.start_run(run_name=spec.run_name) as run:
        mlflow.set_tags({"model_family": spec.family, **spec.tags})
        mlflow.log_params(spec.params)
        mlflow.log_params(
            {
                "model_run_name": spec.run_name,
                "seed": C.RANDOM_SEED,
                "preprocessor": "standard_scaler+onehot",
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "positive_rate": round(float((y_train == C.POS_LABEL).mean()), 4),
            }
        )

        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, y_pred, y_proba)
        metrics["fit_seconds"] = round(fit_seconds, 3)

        from sklearn.metrics import classification_report

        report_txt = classification_report(y_test, y_pred, digits=4)

        import tempfile

        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            cm_fig = C.confusion_matrix_figure(y_test, y_pred)
            cm_path = tmp_path / "confusion_matrix.png"
            cm_fig.savefig(cm_path, dpi=120)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            roc_fig = C.roc_curve_figure(y_test, y_proba, spec.run_name)
            roc_path = tmp_path / "roc_curve.png"
            roc_fig.savefig(roc_path, dpi=120)
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

            report_path = tmp_path / "classification_report.txt"
            report_path.write_text(report_txt)
            mlflow.log_artifact(str(report_path), artifact_path="plots")

        plt.close(cm_fig)
        plt.close(roc_fig)

        mlflow.log_metrics(metrics)

        from mlflow.models import infer_signature

        signature = infer_signature(X_test, y_proba)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
        )

        summary = {
            "run_id": run.info.run_id,
            "run_name": spec.run_name,
            "family": spec.family,
            "params": dict(spec.params),
            "metrics": metrics,
            "model_uri": f"runs:/{run.info.run_id}/model",
        }

    return summary


def select_best(summaries: list[dict]) -> dict:
    """Rank by F1, tie-break on ROC-AUC (see common.SELECTION_METRIC)."""
    return sorted(
        summaries,
        key=lambda s: (
            s["metrics"][C.SELECTION_METRIC],
            s["metrics"][C.TIE_BREAK_METRIC],
        ),
        reverse=True,
    )[0]


def register_best(best: dict) -> dict:
    """Register the winning run and walk it Staging -> Production."""
    client = MlflowClient()
    mv = mlflow.register_model(best["model_uri"], C.MODEL_NAME)
    version = str(mv.version)
    transitions = []

    for stage in ("Staging", "Production"):
        moved = client.transition_model_version_stage(
            C.MODEL_NAME, version, stage, archive_existing_versions=False
        )
        transitions.append(
            {
                "model": C.MODEL_NAME,
                "version": version,
                "to_stage": stage,
                "from_run": best["run_id"],
                "final_stage": moved.current_stage,
            }
        )
        mlflow.log_param(f"registered_stage_{stage.lower()}", stage)
        print(f"  registered {C.MODEL_NAME} v{version} -> {stage}")

    return {
        "model_name": C.MODEL_NAME,
        "version": version,
        "run_id": best["run_id"],
        "run_name": best["run_name"],
        "selection_metric": C.SELECTION_METRIC,
        "selection_value": best["metrics"][C.SELECTION_METRIC],
        "transitions": transitions,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-registry", action="store_true", help="skip model registration")
    parser.add_argument(
        "--only", nargs="*", default=None, help="train only these run names"
    )
    args = parser.parse_args(argv)

    C.setup_mlflow(C.TRAINING_EXPERIMENT)

    df = C.load_churn_data()
    train_df, test_df = C.stratified_split(df)
    print(f"dataset: {len(df)} rows | train {len(train_df)} | test {len(test_df)}")

    specs = MODEL_SPECS
    if args.only:
        specs = [s for s in MODEL_SPECS if s.run_name in args.only]
        if not specs:
            raise SystemExit(f"no run named {args.only}")

    summaries = [train_one(spec, train_df, test_df) for spec in specs]

    best = select_best(summaries)
    print(
        f"best run: {best['run_name']} "
        f"({C.SELECTION_METRIC}={best['metrics'][C.SELECTION_METRIC]:.4f}, "
        f"{C.TIE_BREAK_METRIC}={best['metrics'][C.TIE_BREAK_METRIC]:.4f})"
    )

    registration = None
    if not args.no_registry:
        with mlflow.start_run(run_name="registry-promotion") as promote_run:
            mlflow.log_param("promoted_run_id", best["run_id"])
            mlflow.log_param("promoted_run_name", best["run_name"])
            mlflow.log_param("model_name", C.MODEL_NAME)
            registration = register_best(best)
            out = C.EXPORTS_DIR / "registry_transitions.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(registration, indent=2))
            mlflow.log_artifact(str(out), artifact_path="registry")
        print(f"wrote {out}")

    C.EXPORTS_DIR.mkdir(exist_ok=True)
    (C.EXPORTS_DIR / "training_summaries.json").write_text(
        json.dumps(summaries, indent=2)
    )
    print(f"trained {len(summaries)} models; best={best['run_name']}")


if __name__ == "__main__":
    main()
