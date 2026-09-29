"""Serve the registered Production-stage churn model behind a small FastAPI API.

Usage:
    uv run uvicorn churn_mlops.serve:app --host 0.0.0.0 --port 8080
    uv run python -m churn_mlops.serve            # same thing, via __main__

Override the loaded model with MODEL_URI, e.g.
    MODEL_URI="models:/TelcoChurnClassifier/Staging" ...
    MODEL_URI="runs:/<run_id>/model" ...
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn_mlops import common as C

MODEL_URI = os.environ.get("MODEL_URI", f"models:/{C.MODEL_NAME}/Production")

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    _state["model"] = mlflow.sklearn.load_model(MODEL_URI)
    _state["features"] = C.feature_columns(C.load_churn_data())
    try:
        from mlflow.tracking import MlflowClient

        if MODEL_URI.startswith("models:/"):
            name, _, stage = MODEL_URI.removeprefix("models:/").partition("/")
            versions = MlflowClient().get_latest_versions(name, stages=[stage or "Production"])
            _state["model_version"] = versions[0].version if versions else None
            _state["model_stage"] = stage or "Production"
    except Exception:  # registry metadata is nice-to-have, not load-critical
        _state["model_version"] = None
        _state["model_stage"] = None
    yield
    _state.clear()


app = FastAPI(title="Telco Churn Scoring API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(
        ..., description="Raw customer records (every feature column except Churn)"
    )


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_stage": _state.get("model_stage"),
        "model_version": _state.get("model_version"),
        "n_features": len(_state.get("features", [])),
    }


@app.post("/predict", response_model=dict)
def predict(request: PredictRequest) -> dict[str, Any]:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    features = _state["features"]
    frame = pd.DataFrame(request.instances)
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing feature columns: {missing}")
    frame = frame[features]
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce").fillna(0.0)

    started = time.perf_counter()
    proba = model.predict_proba(frame)[:, 1]
    latency_ms = (time.perf_counter() - started) * 1000

    predictions = [
        Prediction(
            churn_probability=round(float(p), 4),
            churn_prediction=C.POS_LABEL if p >= 0.5 else "No",
        )
        for p in proba
    ]
    return {
        "model_uri": MODEL_URI,
        "model_version": _state.get("model_version"),
        "n_scored": len(predictions),
        "latency_ms": round(latency_ms, 2),
        "predictions": [p.model_dump() for p in predictions],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("APP_HOST", "0.0.0.0"), port=8080)
