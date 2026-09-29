#!/usr/bin/env python3
"""W17 nightly LLM regression suite - Track B.

Pipeline (all in-process, no server needed):

1. Golden set: 6 representative queries from eval/queries.json covering KB fact
   lookup, cross-tool reasoning, unanswerable refusal, clarification and a
   failure injection.
2. Run them through the agentic loop with a pinned prompt version (default v3).
3. Grade them with the W16 harness graders (task completion) AND with two
   Evidently LLM-as-judge checks:
     - faithfulness: answer grounded in the evidence notes/tool outputs
     - answers_question: the answer actually answers the question (PASS/FAIL)
   Both checks are row-level tests aggregated by Evidently's RowTestSummary:
   the share of passing rows must be >= --min-success-rate.
4. Log everything to MLflow (kind=llm_regression): pct_tests_passed, harness
   metrics, per-row judge verdicts+reasoning, HTML report. Exit non-zero below
   --fail-under when --gate is passed (Airflow/CI use this).

Usage:
    uv run python -m mlops.regression --prompt v3
    uv run python -m mlops.regression --prompt v3 --gate
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  # noqa: E402
from evidently import Dataset, Report, tests  # noqa: E402
from evidently.descriptors import FaithfulnessLLMEval, LLMEval  # noqa: E402
from evidently.llm.templates import BinaryClassificationPromptTemplate  # noqa: E402
from evidently.metrics import RowTestSummary  # noqa: E402

from app.agent.prompts import PROMPTS_DIR, load_prompt_bundle  # noqa: E402
from app.config import settings  # noqa: E402
from eval.harness import aggregate  # noqa: E402
from mlops.experiments import SPECS, run_one, version_metrics  # noqa: E402
from mlops.tracking import init_tracking  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"

# Golden set: one (or two) queries per behavioural category.
GOLDEN_IDS = [
    "kb_fact_list_comprehension",      # KB fact + citation
    "kb_compare_lists_vs_tuples",      # cross-source comparison
    "cross_tool_weather_percent",      # weather + calculator
    "unanswerable_mars_boiling_point", # must refuse honestly
    "ambiguous_needs_clarification",   # must ask the user
    "injection_kb_down",               # tool outage must not be papered over
]

ANSWERS_QUESTION_CRITERIA = (
    "The text has a QUESTION line, an ANSWER line and a SOURCES line. "
    "PASS if the ANSWER directly addresses the QUESTION, including when the "
    "ANSWER is: an honest refusal (no information available); a clarifying "
    "question asked back to the user because only the user can supply the "
    "missing information; or a request for permission to answer via an "
    "alternative route (e.g. from general knowledge) because the knowledge "
    "base or a tool is unavailable. "
    "When the QUESTION asks to cite sources: citations are delivered "
    "structurally on the SOURCES line, not inside the ANSWER text - a "
    "non-empty SOURCES line satisfies the citation request. SOURCES 'none' is "
    "correct whenever the facts come from other tools (weather, calculator, "
    "current time) rather than the knowledge base - never FAIL an answer for "
    "having SOURCES 'none' in that case. "
    "FAIL only if the ANSWER is irrelevant to the QUESTION, evasive, "
    "contradicts itself, or silently ignores what was asked."
)


def _category_test(values: list[str], column: str):
    """Bind an is-in test to one specific judge output column."""
    descriptor_test = tests.is_in(values).descriptor
    descriptor_test.column = column
    return descriptor_test


def build_evidence(record: dict) -> str:
    """Reconstruct the evidence available to the judge from the step trace."""
    parts = [f"[user_question] {record.get('query', '')}"]
    for entry in record.get("trace", []):
        if entry.get("action") == "tool" and entry.get("tool"):
            parts.append(f"[{entry['tool']}] {entry.get('result', '')}")
        elif entry.get("action") == "implicit_submit":
            parts.append(f"[researcher] {entry.get('reasoning', '')}")
    text = "\n".join(parts).strip()
    return (text or "no tool evidence was collected")[:6000]


def judge_answer(record: dict) -> str:
    answer = (record.get("answer") or "").strip()
    if not answer:
        # Clarification/refusal runs produce no final answer text; the judge
        # must see what the agent did instead.
        question = ""
        for entry in record.get("trace", []):
            if entry.get("action") == "ask_user" and entry.get("question"):
                question = entry["question"]
                break
        if question:
            return f"(CLARIFYING QUESTION instead of an answer:) {question}"
        return "(NO ANSWER: the agent reported it could not answer.)"
    return answer


def run_golden(prompt_version: str, mode: str, temperature: float, max_steps: int) -> list[dict]:
    bundle = load_prompt_bundle(prompt_version)
    specs = [s for s in SPECS if s["id"] in GOLDEN_IDS]
    missing = set(GOLDEN_IDS) - {s["id"] for s in specs}
    if missing:
        raise SystemExit(f"golden ids not found in eval/queries.json: {sorted(missing)}")

    async def _run() -> list[dict]:
        records = []
        for spec in specs:
            record = await run_one(
                spec, mode=mode, bundle=bundle, temperature=temperature, max_steps=max_steps
            )
            records.append(record)
            print(
                f"  [golden] {record['id']:<34} {record['classification']:<21} "
                f"stop={record['stop_reason']}",
                flush=True,
            )
        return records

    return asyncio.run(_run())


def run_judges(
    records: list[dict], model: str, min_success_rate: float
) -> tuple[pd.DataFrame, object, dict]:
    df = pd.DataFrame(
        [
            {
                "id": r["id"],
                "query": r["query"],
                "answer": judge_answer(r),
                # the relevance judge must see both sides of the exchange plus
                # the structured citations the harness graded
                "judge_input": (
                    f"QUESTION: {r['query']}\n"
                    f"ANSWER: {judge_answer(r)}\n"
                    f"SOURCES: {json.dumps(r['sources']) if r['sources'] else 'none'}"
                ),
                "evidence": build_evidence(r),
                "stop_reason": r["stop_reason"],
                "classification": r["classification"],
                "fabrics": len(r["fabricated_citations"]),
                "steps": r["steps"],
            }
            for r in records
        ]
    )
    faithfulness = FaithfulnessLLMEval(
        column_name="answer",
        context="evidence",
        provider="openai",
        model=model,
        alias="faithfulness",
        tests=[_category_test(["FAITHFUL"], "faithfulness")],
    )
    answers_question = LLMEval(
        column_name="judge_input",
        provider="openai",
        model=model,
        template=BinaryClassificationPromptTemplate(
            criteria=ANSWERS_QUESTION_CRITERIA,
            target_category="FAIL",
            non_target_category="PASS",
            include_reasoning=True,
        ),
        alias="answers_question",
        tests=[_category_test(["PASS"], "answers_question")],
    )
    dataset = Dataset.from_pandas(df, descriptors=[faithfulness, answers_question])
    report = Report([RowTestSummary(min_success_rate=min_success_rate)], include_tests=True)
    snapshot = report.run(dataset)
    return dataset.as_dataframe(), snapshot, _summarize(snapshot)


def _summarize(snapshot) -> dict:
    """pct_tests_passed over the judge tests (row-count guard excluded)."""
    data = snapshot.dict()
    statuses = []
    for item in data.get("tests", []):
        name = str(item.get("name", ""))
        if name.startswith("Row count"):
            continue
        status = item.get("status")
        statuses.append((name, str(getattr(status, "value", status))))
    passed = sum(1 for _, s in statuses if s == "SUCCESS")
    return {
        "tests": statuses,
        "n_tests": len(statuses),
        "n_passed": passed,
        "pct_tests_passed": round(passed / len(statuses), 3) if statuses else 0.0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track B Evidently LLM regression suite")
    parser.add_argument("--prompt", default="v3", help="prompt version under test")
    parser.add_argument("--mode", default="multi", choices=["single", "multi"])
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="researcher temperature (0.0 by default: regression favours reproducibility)",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.9,
        help="min share of golden rows each judge must pass (per-row test)",
    )
    parser.add_argument("--fail-under", type=float, default=1.0, help="min pct_tests_passed")
    parser.add_argument("--gate", action="store_true", help="exit 1 below --fail-under")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.VLLM_MODEL
    init_tracking()

    print(f"== golden set ({len(GOLDEN_IDS)} queries, prompt {args.prompt}) ==", flush=True)
    records = run_golden(args.prompt, args.mode, args.temperature, args.max_steps)

    print("== Evidently LLM judges ==", flush=True)
    judged_df, snapshot, summary = run_judges(records, model, args.min_success_rate)
    harness_metrics = version_metrics(records)
    print(
        f"  judges: {summary['n_passed']}/{summary['n_tests']} tests passed "
        f"({summary['pct_tests_passed']:.0%})",
        flush=True,
    )
    for name, status in summary["tests"]:
        print(f"    [{status}] {name}", flush=True)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"llm_regression_{args.prompt}_{stamp}.html"
    snapshot.save_html(str(html_path))

    metrics = {
        "pct_tests_passed": summary["pct_tests_passed"],
        "judge_tests_total": summary["n_tests"],
        "judge_tests_passed": summary["n_passed"],
        "faithfulness_pass_share": float(judged_df["faithfulness in list {'FAITHFUL'}"].mean()),
        "answers_pass_share": float(judged_df["answers_question in list {'PASS'}"].mean()),
        "golden_queries": len(records),
        "harness_success_rate": harness_metrics["success_rate"],
        "harness_completion_rate": harness_metrics["completion_rate"],
        "mean_steps": harness_metrics["mean_steps"],
        "fabricated_citations_total": harness_metrics["fabricated_citations_total"],
    }

    with mlflow.start_run(run_name=f"regression_{args.prompt}"):
        mlflow.set_tags(
            {
                "kind": "llm_regression",
                "prompt_version": args.prompt,
                "suite": "golden_set",
                "gate": str(args.gate).lower(),
            }
        )
        mlflow.log_params(
            {
                "prompt_version": args.prompt,
                "mode": args.mode,
                "temperature": args.temperature,
                "max_steps": args.max_steps,
                "golden_ids": ",".join(GOLDEN_IDS),
                "min_success_rate": args.min_success_rate,
                "fail_under": args.fail_under,
                "model": model,
                "judge_1": "evidently FaithfulnessLLMEval (answer vs evidence)",
                "judge_2": "evidently LLMEval BinaryClassification (answers question)",
                "graders": "eval/harness.py (W16, in-process)",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(summary["tests"], "judge_test_status.json")
        mlflow.log_dict(
            [
                {
                    "id": r["id"],
                    "answer": r["answer"],
                    "stop_reason": r["stop_reason"],
                    "classification": r["classification"],
                    "trace": r["trace"],
                    "termination": r["termination"],
                }
                for r in records
            ],
            "golden_traces.json",
        )
        mlflow.log_text(
            (PROMPTS_DIR / f"prompt_{args.prompt}.txt").read_text(encoding="utf-8"),
            "prompt.txt",
        )
        mlflow.log_artifact(str(html_path), artifact_path="reports")
        judged_df.drop(columns=["evidence"], errors="ignore").to_csv(
            REPORTS_DIR / f"judge_outputs_{args.prompt}_{stamp}.csv", index=False
        )
        mlflow.log_artifact(
            str(REPORTS_DIR / f"judge_outputs_{args.prompt}_{stamp}.csv"),
            artifact_path="reports",
        )

    print("\n== metrics ==")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.gate and summary["pct_tests_passed"] < args.fail_under:
        print(
            f"GATE FAILED: pct_tests_passed {summary['pct_tests_passed']} "
            f"< {args.fail_under}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
