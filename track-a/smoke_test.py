"""End-to-end smoke test for the serving layer (no server process needed).

Loads the Production model from the registry in-process through FastAPI's
TestClient, scores one high-risk and one low-risk customer, and checks that
malformed input is rejected.

Usage:
    uv run python smoke_test.py
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from churn_mlops.serve import app

HIGH_RISK = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

LOW_RISK = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 34,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 56.95,
    "TotalCharges": 1889.5,
}


def main() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        body = health.json()
        assert body["status"] == "ok", body
        print(f"health: {body}")

        response = client.post("/predict", json={"instances": [HIGH_RISK, LOW_RISK]})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["n_scored"] == 2, payload
        high, low = payload["predictions"]
        assert 0.0 <= high["churn_probability"] <= 1.0
        assert high["churn_probability"] > low["churn_probability"], (high, low)
        print(f"predictions: {payload['predictions']} ({payload['latency_ms']} ms)")

        bad = client.post("/predict", json={"instances": [{"gender": "Female"}]})
        assert bad.status_code == 400, bad.text
        print(f"validation rejected bad payload: {bad.json()['detail']}")

    print("smoke test passed")


if __name__ == "__main__":
    main()
