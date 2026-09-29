# Track A — Data Science MLOps: Telco Customer Churn

Production-style MLOps for a churn prediction pipeline: reproducible environment (**uv**),
experiment tracking + model registry (**MLflow**), a serving API, drift monitoring
(**Evidently AI**), and a scheduled drift/retrain DAG (**Airflow**, bonus).

Dataset: IBM Telco Customer Churn (~7,043 customers, 19 features + binary `Churn`),
committed at [`data/Telco-Customer-Churn.csv`](data/Telco-Customer-Churn.csv).

---

## 1. Repository layout

```
track-a/
├── pyproject.toml              # uv project definition (all deps declared here)
├── uv.lock                     # committed lockfile — exact versions for every run
├── data/Telco-Customer-Churn.csv
├── churn_mlops/
│   ├── common.py               # paths, MLflow setup, data cleaning, metrics, plots
│   ├── train.py                # 5 tracked training runs + registry promotion
│   ├── serve.py                # FastAPI server loading models:/…​/Production
│   ├── monitor.py              # Evidently data/target drift + custom metrics
│   └── compare.py              # exports run-comparison / registry / drift tables
├── smoke_test.py               # in-process end-to-end check of the serving layer
├── dags/telco_drift_dag.py     # Airflow DAG (bonus, not installed locally)
├── reports/                    # Evidently HTML reports + drift_summary.json
├── artifacts/                  # exported MLflow comparisons (md/csv/json)
├── mlruns/                     # MLflow artifact store (models, plots, HTML reports)
├── mlflow.db                   # MLflow tracking + model-registry backend (sqlite)
└── README.md                   # this file
```

## 2. Quickstart (one command each)

```bash
uv sync                              # create .venv from uv.lock (exact pinned versions)
uv run python -m churn_mlops.train   # train 5 runs, log to MLflow, register best -> Production
uv run python -m churn_mlops.monitor # Evidently drift check, HTML reports -> MLflow
uv run python -m churn_mlops.compare # export run-comparison + registry tables to artifacts/
uv run python smoke_test.py          # score customers through the Production model
uv run mlflow ui --port 5000         # run comparison UI  -> http://localhost:5000
uv run uvicorn churn_mlops.serve:app --port 8080   # serving API -> :8080/docs
```

The repository ships a populated `mlflow.db` + `mlruns/`, so on a clean clone you can
skip straight to `uv run mlflow ui` or `uv run python -m churn_mlops.serve` and inspect
the already-registered model; re-running the commands above regenerates everything.

---

## 3. Environment & Reproducibility (uv) — *documentation requirement (a)*

**What was broken before uv:** the pipeline inherited an unpinned `requirements.txt`
style workflow (`pandas`, `scikit-learn`, `matplotlib` all floating), a project-local
venv created from a *different* project's Python, and no record of which transitive
versions produced a given metric. MLflow results were therefore not reproducible:
a metric logged on one machine could silently change after a reinstall, and
`pip freeze` snapshots drift from what CI/another machine actually resolves.

**What uv fixes for this project specifically:**

| Problem | uv solution in this repo |
| --- | --- |
| Floating versions change metrics between machines | `uv.lock` pins every direct **and** transitive package (pandas, scikit-learn, matplotlib, mlflow, evidently, fastapi…) to exact wheels + hashes |
| "Works on my machine" venvs | `.venv` is created *by uv, for this project only*, from `requires-python = ">=3.11"` — no inherited/foreign virtualenv state |
| Slow, non-deterministic installs | `uv sync` resolves + installs the lockfile in one step (seconds, content-addressed cache) |
| No declared project metadata | `pyproject.toml` is the single source of truth; `requirements.txt` is not needed |

**One-command reproduction path (verified from a clean state):**

```bash
rm -rf .venv
uv sync          # installs exactly the uv.lock versions, nothing else
uv run python -m churn_mlops.train && uv run python -m churn_mlops.monitor
```

`uv run` guarantees commands execute inside the locked environment (it re-syncs if the
lockfile changed), so every script in this repo runs with the same dependency graph
that produced the committed metrics.

---

## 4. Experiment Tracking Strategy (MLflow) — *documentation requirement (b)*

**Experiment:** `telco-churn-training` (backend: `mlflow.db`, artifacts: `mlruns/`).

### What was varied and why

Five runs, three model families, hyperparameters that genuinely differ (not seeds):

| run_name | family | hyperparameters varied | why |
| --- | --- | --- | --- |
| `logreg_C0.1` | logistic regression | `C=0.1` (strong L2) | strong-regularization baseline, interpretable coefficients |
| `logreg_C10` | logistic regression | `C=10` (weak L2) | same model, less shrinkage — isolates the effect of `C` |
| `rf_depth6` | random forest | `n_estimators=200, max_depth=6, min_samples_leaf=10` | shallow bagging, high bias/low variance |
| `rf_depth16` | random forest | `n_estimators=200, max_depth=16, min_samples_leaf=2` | deep bagging — tests whether capacity overfits 5.6k rows |
| `hgb_lr0.08` | hist. gradient boosting | `max_iter=300, learning_rate=0.08` | sequential boosting, usually best tabular accuracy |

Identical across runs (controlled): stratified 80/20 split with `seed=42`, the same
`ColumnTransformer` (StandardScaler on numerics, OneHotEncoder `handle_unknown="ignore"`
on categoricals), same 5,634-row train / 1,409-row test split, positive rate 26.5%.

### What was logged per run

* **Params** — every hyperparameter above, plus seed, preprocessor, row counts, positive rate.
* **Metrics** — `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `average_precision`
  (all computed on the held-out test set, churn = positive class), plus `fit_seconds`.
  Accuracy alone is meaningless on this 27%-positive target, which is why the ranking
  metric is F1 (tie-break ROC-AUC).
* **Artifacts** — the full sklearn `Pipeline` (`model/`, with MLmodel signature + input
  example), `plots/confusion_matrix.png`, `plots/roc_curve.png`,
  `plots/classification_report.txt`.

### Run comparison — all five runs side by side

Exported live from MLflow by `uv run python -m churn_mlops.compare`
([`artifacts/run_comparison.md`](artifacts/run_comparison.md),
[`artifacts/run_comparison.csv`](artifacts/run_comparison.csv)); this is the same table
the MLflow UI shows under *telco-churn-training → Compare runs*:

| run | accuracy | precision | recall | **f1** | roc_auc | avg precision | fit (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **logreg_C10** ← registered | 0.8048 | 0.6552 | **0.5588** | **0.6032** | 0.8412 | 0.6281 | **0.17** |
| logreg_C0.1 | 0.7999 | 0.6456 | 0.5455 | 0.5913 | 0.8410 | 0.6337 | 0.48 |
| hgb_lr0.08 | 0.7842 | 0.6067 | 0.5321 | 0.5670 | 0.8215 | 0.6141 | 6.68 |
| rf_depth6 | 0.7999 | **0.6769** | 0.4706 | 0.5552 | **0.8430** | **0.6560** | 1.38 |
| rf_depth16 | 0.7899 | 0.6373 | 0.4840 | 0.5502 | 0.8343 | 0.6406 | 1.09 |

*(numbers from the committed MLflow run `13a66838…` et al.; open the UI to reproduce)*

### Which model was registered, and why the table supports it

`logreg_C10` (F1 **0.6032**) was registered and promoted
`TelcoChurnClassifier` **v1: Staging → Production** (transitions recorded in
[`artifacts/registry_transitions.json`](artifacts/registry_transitions.json) and
[`artifacts/registry_status.md`](artifacts/registry_status.md)).

The decision is not "it had the best accuracy" — three runs beat or nearly tied it on
other metrics, and the table above is what drove the choice:

* **`rf_depth6` had the highest ROC-AUC (0.8430), highest precision (0.6769) and highest
  average precision (0.6560)** — but its recall collapsed to **0.4706**: it would miss
  more than half of the customers who actually churn. On a retention use-case a missed
  churner costs more than a false alarm (a discount offer to a staying customer), so
  recall matters more than the small precision edge.
* **`logreg_C10` wins F1 (0.6032) via recall 0.5588 at precision 0.6552** — it identifies
  8.8pp more churners than `rf_depth6` while giving up only 2.2pp of precision. It also
  trains ~8× faster than the forests and ~40× faster than boosting (0.17 s vs 1.38 s /
  6.68 s) and its coefficients are explainable to the retention team.
* **`hgb_lr0.08` under-performed across the board** (F1 0.5670, AUC 0.8215) — boosting's
  extra capacity does not pay off on 5.6k rows with a linear-ish signal.
* **`logreg_C0.1` vs `logreg_C10` isolates the `C` effect**: weaker regularization raises
  recall 0.5455 → 0.5588 and F1 0.5913 → 0.6032 at identical accuracy, confirming the
  decision boundary benefits from less shrinkage on the scaled+one-hot features.
* **`rf_depth16` vs `rf_depth6`**: extra depth *hurt* (F1 0.5502 vs 0.5552, AUC 0.8343 vs
  0.8430) — the classic overfitting signal; capacity is not the bottleneck here.

**Trade-off accepted:** ~0.2pp of ROC-AUC (0.8412 vs 0.8430) in exchange for +8.8pp
recall, +4.8pp F1, 8× faster training, and an interpretable model. If the business
later optimizes for ranking quality (e.g., a top-K calling list), `rf_depth6` is one
`MODEL_URI` switch away — the registry keeps it available as a run artifact.

### Registry stages

`mlflow.register_model(...)` → `transition_model_version_stage(..., "Staging")` →
`transition_model_version_stage(..., "Production")` (MLflow 3 still supports stages but
deprecates them in favour of aliases; stages are used here because the assignment
requires stage transitions, and the deprecation warning is expected).

---

## 5. Model Serving — *documentation requirement (workflow: …registry → serving…)*

`churn_mlops/serve.py` loads `models:/TelcoChurnClassifier/Production` at FastAPI
startup (override with `MODEL_URI=models:/…/Staging` or `runs:/<run_id>/model`).

```bash
uv run uvicorn churn_mlops.serve:app --port 8080
curl -s localhost:8080/health
curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' \
  -d '{"instances": [{ …customer fields… }]}'
```

```json
{"status":"ok","model_uri":"models:/TelcoChurnClassifier/Production",
 "model_stage":"Production","model_version":1,"n_features":19}
```

Scoring example (one high-risk month-to-month customer, one low-risk one-year customer):

```json
{"n_scored":2,"latency_ms":86.9,
 "predictions":[{"churn_probability":0.6189,"churn_prediction":"Yes"},
                {"churn_probability":0.0440,"churn_prediction":"No"}]}
```

`uv run python smoke_test.py` runs exactly this path in-process (TestClient): health,
two predictions (asserting the high-risk customer scores higher), and a 400 on a
payload missing feature columns.

---

## 6. Monitoring & Drift Strategy (Evidently AI) — *documentation requirement (c)*

Implemented in `churn_mlops/monitor.py`; every check runs inside an MLflow run in the
`telco-churn-drift-monitoring` experiment, so metrics are queryable and the HTML
reports are MLflow artifacts.

### Reference vs current in this setup

| set | rows | meaning |
| --- | --- | --- |
| **reference** | 70% random sample (seed 42), 4,930 rows | "training-time" data — what the registered model learned from; distributions are the expected baseline |
| **current** | remaining 30%, 2,113 rows → 4,159 after skew | "incoming production" data — what the monitor scores each cycle |

Into **current** we inject deliberate synthetic drift (nothing touches reference):

1. **numeric noise** — `MonthlyCharges += U(8, 20)` per row (billing change),
2. **categorical skew** — `Contract == "Month-to-month"` rows oversampled by 120%
   (mix moves 55.5% → 72.0%; Jensen-Shannon distance **0.12**, above Evidently's
   default 0.10 categorical threshold),
3. **label/concept drift** — 10% of `Churn` labels flipped.

`--no-drift` runs the same split unperturbed as a control.

### Metrics monitored

* **Data Drift report** (`DataDriftPreset`, all 20 feature columns, per-column
  Wasserstein/Jensen-Shannon tests at threshold 0.10, plus the drifted-columns-share test)
  → [`reports/data_drift_report.html`](reports/data_drift_report.html)
* **Target Drift report** (`TargetDriftPreset`, churn as target; categorical stat-test
  threshold tightened to **0.05** — a ≥5pp base-rate move already changes expected
  revenue enough to act on) → [`reports/target_drift_report.html`](reports/target_drift_report.html)
* **Custom metrics** built on Evidently's custom metric/test API
  (`SingleValueMetric` + `SingleValueCalculation` + bound `tests`) →
  [`reports/custom_metrics_report.html`](reports/custom_metrics_report.html):
  * `MeanChargesShift` = |mean(MonthlyCharges)current − mean(… )reference| with test `< $5`
  * `MtmChurnRateShift` = |churn-rate current − reference| **inside the
    `Contract = Month-to-month` segment**, with test `< 0.03`
* Headline values are logged as **MLflow metrics** (`drift_share`, `n_drifted_columns`,
  `churn_rate_shift`, `target_drift_detected`, `mean_monthly_charges_shift`,
  `mtm_churn_rate_shift`, `custom_metric_tests_passed`, `retrain_recommended`) so a DAG
  can read the verdict without parsing HTML.

### What the report showed (committed run)

| check | result | verdict |
| --- | --- | --- |
| Data drift, per column | flagged **MonthlyCharges (0.46)**, **Contract (0.12)**, plus **tenure (0.21)** and **TotalCharges (0.15)** | ✅ both engineered columns detected |
| Share of drifted columns | 4/20 = **0.20** ≥ 0.10 alert threshold | ⚠️ alert |
| Target drift | churn rate 26.96% → 38.04% (**+11.08pp**), JS score **0.084 ≥ 0.05** target threshold | ✅ label drift detected |
| Custom `MeanChargesShift` | **$13.84** vs $5 limit → test **FAIL** | ✅ catches what the default per-column tests already see, with a business-unit limit |
| Custom `MtmChurnRateShift` | 2.57pp vs 3pp limit → test **SUCCESS** | segment churn mix shifted less than the overall rate |

**Interpretation.** The report flags precisely the columns we perturbed, plus two honest
*side effects*: oversampling month-to-month customers also skews `tenure` and
`TotalCharges`, because that contract type has a different tenure/charges profile —
feature drift propagates, and Evidently correctly followed the causal chain. In
production this would mean the model is being asked to score a population it never saw
(higher bills, shorter tenure, more month-to-month contracts) *and* that the base rate
it was calibrated against has moved +11pp — its predicted churn probabilities would be
systematically wrong (under-predicting churn), precision/recall on the next batch would
degrade, and segment-level fairness (month-to-month customers) would shift first.

### Action on crossing a threshold

`retrain_recommended = 1` when `drift_share ≥ 0.10` **or** any custom test fails.
Action ladder (implemented in `dags/telco_drift_dag.py`):

1. **< threshold** — record the drift metrics, keep serving; next check on schedule.
2. **≥ threshold (this run)** — automated retrain: re-run the full training experiment,
   register the new best model, promote Staging → Production (previous version is
   archived by the transition). If the new F1/ROC-AUC falls below the incumbent by more
   than a tolerance, the promotion step fails and the old Production model keeps serving
   (manual review).
3. **Target drift ≥ 0.05 with feature drift < 0.10** — concept drift: recalibrate the
   decision threshold / re-label a sample before retraining, since features look healthy
   but the decision boundary moved.

---

## 7. Orchestration (Airflow bonus) — *documentation requirement (d)*

[`dags/telco_drift_dag.py`](dags/telco_drift_dag.py) — DAG id `telco_churn_drift_monitor`.

```
run_drift_check ──▶ evaluate_verdict ──▶ branch_on_verdict ──┬─▶ trigger_retrain ──▶ retrain_pipeline
  (Evidently)        (reads MLflow)                         └─▶ drift_healthy
```

| aspect | value |
| --- | --- |
| **Schedule** | `@weekly` (Monday 06:00), `catchup=False`, owner `ml-engineering`, 1 retry |
| **Trigger condition** | `evaluate_verdict` reads the newest run in `telco-churn-drift-monitoring` and branches on the logged `retrain_recommended` metric (1 when `drift_share ≥ 0.10` **or** a custom Evidently test failed) |
| **On a positive signal** | `trigger_retrain` logs the reason to XCom/task log, then `retrain_pipeline` executes `uv run python -m churn_mlops.train` from `track-a/` — retraining + registry promotion happen automatically |
| **On a negative signal** | `drift_healthy` records `drift_share` and stops; nothing is retrained, the weekly history stays auditable |

Airflow is an optional bonus and is **not installed in this environment** (≈600 MB of
extra dependencies unrelated to the pipeline). The DAG file is validated with
`python -m py_compile dags/telco_drift_dag.py` plus an import test against a stubbed
`airflow` module; to run it for real, place `dags/` in an Airflow `dags_folder` with
`uv` on the worker's PATH (Airflow 2.8+/3.x).

---

## 8. Standard workflow map

| stage | entrypoint | artifact |
| --- | --- | --- |
| data | `data/Telco-Customer-Churn.csv` → `common.load_churn_data()` | cleaned frame (TotalCharges coerced) |
| training | `uv run python -m churn_mlops.train` | 5 MLflow runs, `plots/*`, `model/` |
| tracking | MLflow `telco-churn-training` | `artifacts/run_comparison.{md,csv}` |
| registry | same command, `register_best()` | `TelcoChurnClassifier` v1 Staging → Production |
| serving | `uv run uvicorn churn_mlops.serve:app` | `/health`, `/predict` |
| monitoring | `uv run python -m churn_mlops.monitor` | `reports/*.html`, MLflow drift metrics |
| retraining | Airflow `trigger_retrain` → `churn_mlops.train` | new registered version |

## 9. Deliverables checklist

* [x] **uv** — `pyproject.toml` + committed `uv.lock`, one-command `uv sync`
* [x] **MLflow run comparison** for all ≥3 models — `artifacts/run_comparison.md` / `.csv`
      (5 runs), viewable in `uv run mlflow ui`
* [x] **Model registry** — `TelcoChurnClassifier` v1 registered, Staging → Production
      (`artifacts/registry_status.md`, `artifacts/registry_transitions.json`)
* [x] **Serving** — FastAPI wrapper + `smoke_test.py`
* [x] **Evidently reports (HTML)** — data drift, target drift, custom metrics in
      `reports/`, also logged as MLflow artifacts
* [x] **Custom metric/test** — `MeanChargesShift` & `MtmChurnRateShift` via Evidently's
      custom metric API with bound tests
* [x] **Airflow DAG (bonus)** — `dags/telco_drift_dag.py`
* [x] **Documentation** — this README (sections a–d above)
