# MLOps

Two independent, production-style tracks in one repository:

| | Track A — [`track-a/`](track-a/) | Track B — [`track-b/`](track-b/) |
|---|---|---|
| **Problem** | Telco customer churn prediction (tabular ML) | Production-ready AI assistant with RAG + agentic research loop |
| **Stack** | uv, scikit-learn, MLflow, FastAPI, Evidently AI, Airflow | uv/venv, FastAPI, Streamlit, ChromaDB, OpenAI, MLflow, Evidently AI, Docker, Airflow |
| **Lifecycle covered** | Environment → training → tracking → registry → serving → drift monitoring → retrain DAG | App → reliability patterns → RAG → agentic loop → eval harness → prompt experiments → nightly regression gate |
| **Full docs** | [`track-a/README.md`](track-a/README.md) | [`track-b/README.md`](track-b/README.md) |

Each track is self-contained (its own environment, its own MLflow backend, its own
commands) — nothing in one track is required to run the other.

---

## Repository layout

```
MLOps/
├── README.md                  # this file
├── track-a/                   # Track A — Data Science MLOps: churn prediction
│   ├── pyproject.toml + uv.lock
│   ├── data/Telco-Customer-Churn.csv
│   ├── churn_mlops/           # train / serve / monitor / compare
│   ├── dags/telco_drift_dag.py
│   ├── reports/  artifacts/   # Evidently HTML, run comparisons, registry state
│   ├── mlflow.db + mlruns/    # populated tracking + registry backend
│   └── smoke_test.py
└── track-b/                   # Track B — Agentic AI assistant & RAG system
    ├── app/                   # FastAPI gateway, LLM client, resilience, RAG, agent/
    ├── ui/                    # Streamlit frontend
    ├── eval/                  # built-from-scratch harness (13 queries × 2 modes)
    ├── mlops/                 # MLflow tracking, prompt experiments, nightly regression
    ├── prompts/               # prompt_v1/v2/v3.txt
    ├── dags/nightly_regression_dag.py
    ├── tests/                 # 39 unit/integration tests (14 deterministic agent tests)
    ├── Dockerfile + docker-compose.yml
    ├── mlflow.db + mlruns/    # populated tracking backend
    └── ONNX.md, ARCHITECTURE.txt
```

---

## Track A — Data Science MLOps: Telco Customer Churn

Production-style MLOps for a churn pipeline on the IBM Telco dataset
(~7,043 customers, 19 features, binary `Churn`, 26.5% positive rate).

**What has been done:**

1. **Reproducible environment (uv)** — `pyproject.toml` + committed `uv.lock` pins every
   direct *and* transitive dependency; `uv sync` reproduces the exact environment that
   produced the committed metrics.
2. **Experiment tracking (MLflow)** — experiment `telco-churn-training` with **5 runs
   across 3 model families** (logistic regression ×2, random forest ×2, histogram
   gradient boosting), logging params, 6 test-set metrics (`accuracy`, `precision`,
   `recall`, `f1`, `roc_auc`, `average_precision`), fit time, confusion-matrix/ROC plots
   and the sklearn `Pipeline` itself.
3. **Run comparison & model selection** — exported to
   [`track-a/artifacts/run_comparison.md`](track-a/artifacts/run_comparison.md)/`.csv`.
   `logreg_C10` won on **F1 (0.6032)** via recall 0.5588; `rf_depth6` had the best
   ROC-AUC (0.8430) but recall collapsed to 0.4706 — the retention use-case weights
   missed churners higher than false alarms, which is why the logistic model was chosen.
4. **Model registry** — `TelcoChurnClassifier` v1 registered and promoted
   **Staging → Production** ([`artifacts/registry_status.md`](track-a/artifacts/registry_status.md),
   `registry_transitions.json`).
5. **Serving (FastAPI)** — `churn_mlops/serve.py` loads
   `models:/TelcoChurnClassifier/Production`, exposes `/health` and `/predict`;
   `smoke_test.py` asserts a high-risk customer scores above a low-risk one and that a
   malformed payload returns 400.
6. **Drift monitoring (Evidently AI)** — reference (70%) vs synthetic-drifted "current"
   split (MonthlyCharges noise, month-to-month oversampling, 10% label flips) → data
   drift, target drift and **two custom metrics** (`MeanChargesShift`,
   `MtmChurnRateShift`) with bound tests, all reported as HTML + MLflow metrics in
   [`track-a/reports/`](track-a/reports/).
7. **Orchestration (Airflow bonus)** — `dags/telco_drift_dag.py`: weekly drift check →
   verdict read from MLflow → branch to automated retrain + registry promotion, or to a
   "healthy" no-op. Validated with `py_compile`; Airflow itself is not installed locally.

**Quickstart:**

```bash
cd track-a
uv sync
uv run python -m churn_mlops.train      # 5 tracked runs + registry promotion
uv run python -m churn_mlops.monitor    # Evidently drift reports
uv run python -m churn_mlops.compare    # export comparison tables
uv run mlflow ui --port 5000            # http://localhost:5000
uv run uvicorn churn_mlops.serve:app --port 8080   # http://localhost:8080/docs
```

---

## Track B — Production-Ready AI Assistant & RAG System

A conversational assistant with a FastAPI gateway, Streamlit UI, ChromaDB-backed RAG,
tool calling, and a bounded agentic research loop.

**What has been done:**

1. **Application core** — FastAPI backend (`app/main.py`) + Streamlit UI (`ui/app.py`)
   with endpoints `POST /chat`, `/chat/tools`, `/chat/rag`, `/chat/structured`,
   `/chat/research`, plus `GET /health` and `/rag/documents`.
2. **Reliability patterns** — exponential backoff with jitter (3 attempts),
   sliding-window rate limiting (30 req/min/IP → 429), circuit breaker (5 failures →
   30 s cooldown), automatic fallback to a local vLLM/Ollama provider, thread-safe LRU
   response cache (256 entries, 5-min TTL), async offload of ChromaDB queries,
   `X-Request-ID`/`X-Response-Time` telemetry and structured error responses.
3. **RAG pipeline** — PDF/TXT/MD ingestion with tiktoken chunking, `all-MiniLM-L6-v2`
   embeddings (ONNX export utility in `app/rag/onnx_export.py`), ChromaDB vector store,
   cosine-distance retrieval with citations. ONNX feasibility analysis in
   [`track-b/ONNX.md`](track-b/ONNX.md).
4. **Agentic research loop (W16)** — `POST /chat/research` lets the model decide after
   every tool result whether to re-search, switch tools, ask the user, or answer;
   `mode=single` (in-context self-check) vs `mode=multi` (isolated verifier agent).
   Context engineering via persistent evidence notes + clearing of old tool results;
   citations validated against actually-retrieved chunks (no fabrication).
5. **Evaluation harness (built from scratch)** — `eval/harness.py` + `eval/queries.json`,
   13 queries × 2 modes through the real HTTP endpoint: task completion, tool-call
   correctness, trajectory length, token accounting and a hard/soft/cascading failure
   taxonomy → [`eval/results/report.md`](track-b/eval/results/report.md). Result:
   12/13 completion in both modes, 100% tool correctness, +33% tokens for multi-agent;
   6/6 failure-injection runs answered with 0 fabricated citations.
6. **Testing** — 39 unit/integration tests (`tests/`), including 14 deterministic
   agent-loop tests driven by a fake LLM.
7. **MLOps tracking & regression (W17)** — `mlops/tracking.py` points MLflow at
   `track-b/mlflow.db` (experiment `agentify-the-assistant`, runs tagged by `kind`);
   `mlops/experiments.py` runs the full 13-query suite in-process, one MLflow run per
   **prompt version** (`prompts/prompt_v1..v3.txt`, params include prompt sha256),
   logging completion rate, step/token aggregates, failure counts, verification pass
   rate, per-query traces and markdown reports; `mlops/regression.py` is a nightly
   **gated regression suite** — 6 golden queries graded by the harness graders plus two
   **Evidently LLM-as-judge checks** (faithfulness, answers_question), logging
   `pct_tests_passed` + an HTML judge report ([`track-b/reports/`](track-b/reports/)) and
   exiting non-zero below `--fail-under`.
8. **Orchestration (Airflow bonus)** — `dags/nightly_regression_dag.py`: daily
   regression run → verdict read from MLflow → branch to `notify_failure` or
   `regression_healthy`.
9. **Packaging & deployment** — multi-stage `Dockerfile`, `docker-compose.yml`
   (standard / `ollama` / `vllm` profiles), plus Azure Container Apps, AWS ECS Fargate
   and Google Cloud Run deployment guides in the track README.

**Quickstart:**

```bash
cd track-b
cp .env.example .env                       # add OPENAI_API_KEY
docker compose up --build -d               # API :8000, UI :8501
# or without Docker:
pip install -r requirements.txt -r requirements-ui.txt
uvicorn app.main:app --port 8000 &
BACKEND_URL=http://localhost:8000 streamlit run ui/app.py

python3 eval/harness.py --modes single,multi     # evaluation suite
uv run python -m mlops.experiments --versions v1 v2 v3   # prompt experiments -> MLflow
uv run python -m mlops.regression --prompt v3 --gate     # nightly gate
```

---

## What's done at a glance

- [x] **Reproducible environments** — uv lockfiles in both tracks
- [x] **Experiment tracking (MLflow)** — 5 churn training runs (Track A); prompt-version
      experiments + LLM regression runs (Track B); populated `mlflow.db` + `mlruns/`
- [x] **Model registry with stage transitions** — Track A: Staging → Production
- [x] **Serving** — FastAPI scoring API (Track A), full assistant API (Track B)
- [x] **Drift monitoring (Evidently)** — data/target drift, custom metrics with bound
      tests (Track A)
- [x] **LLM quality gating (Evidently)** — LLM-as-judge faithfulness/answer checks with
      a pass-rate gate (Track B)
- [x] **Evaluation harness** — Track B: 13-query suite, failure taxonomy, token costs
- [x] **Agentic AI** — bounded research loop, single vs multi-agent verification
- [x] **Reliability engineering** — retry, rate limit, circuit breaker, cache, fallback
- [x] **Airflow DAGs (bonus, validated but not installed locally)** — drift-triggered
      retrain (Track A), nightly LLM regression gate (Track B)
- [x] **Tests** — Track B: 39 tests incl. 14 deterministic agent-loop tests;
      Track A: `smoke_test.py` end-to-end serving check
- [x] **Documentation** — this README plus the detailed per-track READMEs
