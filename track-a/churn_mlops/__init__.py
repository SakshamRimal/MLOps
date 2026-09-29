"""Track A - Telco Customer Churn MLOps pipeline.

Modules:
    common  - shared config, MLflow setup, data loading
    train   - experiment tracking (MLflow) + model registry
    serve   - model serving API (FastAPI)
    monitor - drift monitoring (Evidently AI)
    compare - run comparison / registry exports
"""
