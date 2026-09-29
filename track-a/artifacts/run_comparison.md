# Track A - MLflow run comparison (telco-churn-training)

All runs ranked by f1 (tie-break roc_auc).

| run_id | registered | run_name | test_rows | preprocessor | seed | C | max_iter | positive_rate | train_rows | solver | learning_rate | max_leaf_nodes | min_samples_leaf | n_estimators | max_depth | accuracy | precision | recall | f1 | roc_auc | average_precision | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13a66838ef43479981a710ed49f0b2e5 | YES <- best (f1, tie-break roc_auc) | logreg_C10 | 1409 | standard_scaler+onehot | 42 | 10.0 | 1000 | 0.2654 | 5634 | lbfgs | nan | nan | nan | nan | nan | 0.8048 | 0.6552 | 0.5588 | 0.6032 | 0.8412 | 0.6281 | 0.167 |
| f564c67ba1d24f74b286ad5a71c6a439 |  | logreg_C0.1 | 1409 | standard_scaler+onehot | 42 | 0.1 | 1000 | 0.2654 | 5634 | lbfgs | nan | nan | nan | nan | nan | 0.7999 | 0.6456 | 0.5455 | 0.5913 | 0.841 | 0.6337 | 0.475 |
| c63ff04e272d4769bece1273c51d3b17 |  | hgb_lr0.08 | 1409 | standard_scaler+onehot | 42 | nan | 300 | 0.2654 | 5634 | nan | 0.08 | 31 | nan | nan | nan | 0.7842 | 0.6067 | 0.5321 | 0.567 | 0.8215 | 0.6141 | 6.682 |
| 24122108134f4d99b28ad27340f6dc66 |  | rf_depth6 | 1409 | standard_scaler+onehot | 42 | nan | nan | 0.2654 | 5634 | nan | nan | nan | 10 | 200 | 6 | 0.7999 | 0.6769 | 0.4706 | 0.5552 | 0.843 | 0.656 | 1.375 |
| b9cd041a3c984071851dba9667dc8325 |  | rf_depth16 | 1409 | standard_scaler+onehot | 42 | nan | nan | 0.2654 | 5634 | nan | nan | nan | 2 | 200 | 16 | 0.7899 | 0.6373 | 0.484 | 0.5502 | 0.8343 | 0.6406 | 1.086 |

**Registered:** `logreg_C10` (run `13a66838ef43479981a710ed49f0b2e5`) -> `TelcoChurnClassifier` Staging -> Production.

Reproduce: `uv run mlflow ui --port 5000` then open the `telco-churn-training` experiment.