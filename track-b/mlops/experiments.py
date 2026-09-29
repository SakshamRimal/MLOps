#!/usr/bin/env python3
"""W17 prompt/trace experiment runner - Track B.

Runs the full W16 evaluation suite (``eval/queries.json``, 13 queries with the
built-from-scratch graders from ``eval/harness.py``) in-process against the
agentic loop - no server needed - and logs one MLflow run per prompt version:

- params: prompt version + sha256, mode, max_steps, temperature, model, ...
- metrics: completion rate, tool correctness, step/token/latency aggregates,
  failure-taxonomy counts, verification pass rate
- artifacts: the exact prompt text, the full step-by-step trace of every query,
  representative success/failure traces, a per-version markdown report

Usage:
    uv run python -m mlops.experiments --versions v1
    uv run python -m mlops.experiments --versions v1 v2 v3 --mode multi
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  # noqa: E402

from app.agent.orchestrator import run_research  # noqa: E402
from app.agent.prompts import PROMPTS_DIR, load_prompt_bundle  # noqa: E402
from app.config import settings  # noqa: E402
from eval.harness import (  # noqa: E402
    aggregate,
    check_tool_arguments,
    classify,
    evaluate_expectations,
)
from mlops.tracking import init_tracking  # noqa: E402

QUERIES_FILE = PROJECT_ROOT / "eval" / "queries.json"
SPECS = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))["queries"]


# --------------------------------------------------------------------------- runner
async def run_one(
    spec: dict,
    *,
    mode: str,
    bundle: dict,
    temperature: float,
    max_steps: int,
) -> dict:
    """One query through the agentic loop, graded by the W16 harness graders.

    Mirrors ``eval/harness.run_one`` (which speaks HTTP) but calls
    ``run_research`` directly, so MLflow runs need no running server.
    """
    response = await run_research(
        spec["query"],
        mode=mode,
        max_steps=max_steps,
        temperature=temperature,
        top_p=1.0,
        max_tokens=1200,
        inject_failure=spec.get("inject_failure"),
        system_prompt=bundle["researcher"],
        verifier_system=bundle["verifier"],
    )

    record = {
        "id": spec["id"],
        "category": spec["category"],
        "mode": mode,
        "query": spec["query"],
        "inject_failure": spec.get("inject_failure"),
        "max_steps": max_steps,
        "expect": spec["expect"],
        "_expected_tools": spec["expect"].get("tools") or spec["expect"].get("tools_any"),
        "answer": response.get("answer", ""),
        "sources": response.get("sources") or [],
        "fabricated_citations": response.get("fabricated_citations") or [],
        "tool_errors": response.get("tool_errors") or [],
        "tools_used": response.get("tools_used") or [],
        "trajectory": response.get("trajectory") or [],
        "trace": response.get("trace") or [],
        "termination": response.get("termination"),
        "steps": response.get("steps"),
        "stop_reason": response.get("stop_reason"),
        "verified": response.get("verified"),
        "verification": response.get("verification"),
        "tokens": response.get("tokens") or {},
        "latency_ms": response.get("latency_ms"),
        "injection": response.get("injection") or "",
        "wall_ms": response.get("latency_ms"),
    }

    completed, failed_keys, detail = evaluate_expectations(record, spec["expect"])
    classification, reasons = classify(record, failed_keys)
    record.pop("_expected_tools", None)

    record["completed"] = completed
    record["failed_keys"] = failed_keys
    record["criteria_detail"] = detail
    record["classification"] = classification
    record["failure_reasons"] = reasons
    record["tool_correctness"] = check_tool_arguments(
        {**record, "_expected_tools": spec["expect"].get("tools") or spec["expect"].get("tools_any")}
    )
    return record


async def run_version(version: str, *, mode: str, temperature: float, max_steps: int) -> tuple[str, list[dict]]:
    bundle = load_prompt_bundle(version)
    records: list[dict] = []
    for spec in SPECS:
        record = await run_one(
            spec, mode=mode, bundle=bundle, temperature=temperature, max_steps=max_steps
        )
        records.append(record)
        print(
            f"  [{version}] {record['id']:<34} {record['classification']:<21} "
            f"steps={record['steps']} stop={record['stop_reason']} "
            f"({record['latency_ms'] / 1000:.1f}s)",
            flush=True,
        )
    raw_file = (PROMPTS_DIR / f"prompt_{version}.txt").read_text(encoding="utf-8")
    return raw_file, records


# --------------------------------------------------------------------------- metrics
def version_metrics(records: list[dict]) -> dict[str, float]:
    agg = aggregate(records)
    metrics: dict[str, float] = {
        "queries": agg["queries"],
        "completion_rate": agg["completion_rate"],
        "mean_steps": agg["mean_steps"],
        "max_steps_seen": agg["max_steps_seen"],
        "hit_step_cap": agg["hit_step_cap"],
        "tool_calls": agg["tool_calls"],
        "invalid_tool_calls": agg["invalid_tool_calls"],
        "tokens_total": agg["tokens_total"],
        "llm_calls": agg["llm_calls"],
        "mean_latency_ms": agg["mean_latency_ms"],
        "success_count": sum(1 for r in records if r["classification"] == "success"),
        "hard_failure_count": sum(1 for r in records if r["classification"] == "hard_failure"),
        "soft_failure_count": sum(1 for r in records if r["classification"] == "soft_failure"),
        "cascading_failure_count": sum(
            1 for r in records if r["classification"] == "cascading_soft_failure"
        ),
        "fabricated_citations_total": sum(len(r["fabricated_citations"]) for r in records),
        "tool_errors_total": sum(len(r["tool_errors"]) for r in records),
        "verified_pass_rate": round(
            sum(1 for r in records if r.get("verified")) / len(records), 3
        ),
        "clarification_count": sum(1 for r in records if r["stop_reason"] == "clarification"),
        "max_stop_count": sum(1 for r in records if r["stop_reason"] == "max_steps"),
    }
    if agg.get("expected_tool_rate") is not None:
        metrics["expected_tool_rate"] = agg["expected_tool_rate"]
    if agg.get("valid_arg_rate") is not None:
        metrics["valid_arg_rate"] = agg["valid_arg_rate"]
    metrics["success_rate"] = round(metrics["success_count"] / len(records), 3)
    return metrics


def report_md(version: str, metrics: dict, records: list[dict]) -> str:
    lines = [
        f"# Prompt experiment - {version}",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Mode: `{records[0]['mode']}` | queries: {len(records)} | max_steps: {records[0]['max_steps']}",
        "",
        "## Aggregate metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in metrics.items()]
    lines += [
        "",
        "## Per-query results",
        "",
        "| id | classification | steps | stop | failed criteria | tool sequence |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        seq = " -> ".join(r["tool_correctness"]["tool_sequence"]) or "(no tool)"
        failed = ", ".join(r["failed_keys"]) or "-"
        lines.append(
            f"| {r['id']} | {r['classification']} | {r['steps']} | {r['stop_reason']} "
            f"| {failed} | {seq} |"
        )
    failures = [r for r in records if r["classification"] != "success"]
    lines += ["", "## Failure diagnosis (from traces)", ""]
    if not failures:
        lines.append("All queries succeeded.")
    for r in failures:
        lines.append(f"### {r['id']} - {r['classification']}")
        lines.append("")
        lines.append(f"- reasons: {', '.join(r['failure_reasons'])}")
        lines.append(f"- stop: `{r['stop_reason']}` after {r['steps']} steps")
        if r["tool_errors"]:
            lines.append(f"- tool errors: {r['tool_errors'][:3]}")
        for entry in r["trace"]:
            if entry.get("status") in ("error", "unparseable"):
                lines.append(
                    f"- step {entry['step']} `{entry['action']}` ({entry.get('tool')}): "
                    f"{entry['result'][:200]}"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def failure_diagnosis(records: list[dict]) -> list[dict]:
    """Compact per-failure summary used to decide the next prompt revision."""
    out = []
    for r in records:
        if r["classification"] == "success":
            continue
        out.append(
            {
                "id": r["id"],
                "classification": r["classification"],
                "reasons": r["failure_reasons"],
                "stop_reason": r["stop_reason"],
                "steps": r["steps"],
                "tool_errors": r["tool_errors"][:3],
                "trace": r["trace"],
            }
        )
    return out


# --------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track B prompt/trace experiments")
    parser.add_argument("--versions", nargs="+", default=["v1"], help="prompt versions, e.g. v1 v2")
    parser.add_argument("--mode", default="multi", choices=["single", "multi"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    init_tracking()

    async def _run() -> list[tuple[str, str, list[dict]]]:
        results = []
        for version in args.versions:
            print(f"== running prompt {version} ({args.mode}) ==", flush=True)
            prompt_text, records = await run_version(
                version, mode=args.mode, temperature=args.temperature, max_steps=args.max_steps
            )
            results.append((version, prompt_text, records))
        return results

    results = asyncio.run(_run())

    print("\n== logging to MLflow ==")
    summary_rows = []
    for version, prompt_text, records in results:
        metrics = version_metrics(records)
        prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]
        with mlflow.start_run(run_name=f"prompt_{version}_{args.mode}"):
            mlflow.set_tags(
                {
                    "kind": "prompt_experiment",
                    "prompt_version": version,
                    "mode": args.mode,
                    "suite": str(QUERIES_FILE.relative_to(PROJECT_ROOT)),
                }
            )
            mlflow.log_params(
                {
                    "prompt_version": version,
                    "prompt_sha256": prompt_sha,
                    "mode": args.mode,
                    "temperature": args.temperature,
                    "max_steps": args.max_steps,
                    "model": (
                        settings.OPENAI_MODEL
                        if settings.LLM_PROVIDER == "openai"
                        else settings.VLLM_MODEL
                    ),
                    "top_p": 1.0,
                    "max_tokens": 1200,
                    "n_queries": len(records),
                    "graders": "eval/harness.py (W16, in-process)",
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_text(prompt_text, "prompt.txt")
            mlflow.log_dict(
                {r["id"]: {"trace": r["trace"], "termination": r["termination"]} for r in records},
                f"traces_{version}.json",
            )
            mlflow.log_dict(
                [
                    {
                        "id": r["id"],
                        "classification": r["classification"],
                        "failed_keys": r["failed_keys"],
                        "steps": r["steps"],
                        "stop_reason": r["stop_reason"],
                        "verified": r["verified"],
                        "tool_sequence": r["tool_correctness"]["tool_sequence"],
                        "answer": r["answer"],
                        "sources": r["sources"],
                        "fabricated_citations": r["fabricated_citations"],
                        "tokens": r["tokens"],
                        "latency_ms": r["latency_ms"],
                    }
                    for r in records
                ],
                f"results_{version}.json",
            )
            mlflow.log_dict(failure_diagnosis(records), f"failures_{version}.json")

            success = next((r for r in records if r["classification"] == "success"), None)
            failure = next((r for r in records if r["classification"] != "success"), None)
            if success:
                mlflow.log_dict(
                    {"id": success["id"], "trace": success["trace"], "termination": success["termination"]},
                    "trace_representative_success.json",
                )
            if failure:
                mlflow.log_dict(
                    {"id": failure["id"], "trace": failure["trace"], "termination": failure["termination"]},
                    "trace_representative_failure.json",
                )
            mlflow.log_text(report_md(version, metrics, records), f"report_{version}.md")

        summary_rows.append({"version": version, "prompt_sha256": prompt_sha, **metrics})
        print(
            f"  {version}: success {metrics['success_count']}/{metrics['queries']} "
            f"completion_rate={metrics['completion_rate']} hard={metrics['hard_failure_count']} "
            f"soft={metrics['soft_failure_count']} mean_steps={metrics['mean_steps']}",
            flush=True,
        )

    print("\n| version | sha | success | completion | hard | soft | cascading | mean_steps | hit_cap |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in summary_rows:
        print(
            f"| {row['version']} | {row['prompt_sha256']} | {row['success_count']}/{row['queries']} "
            f"| {row['completion_rate']} | {row['hard_failure_count']} | {row['soft_failure_count']} "
            f"| {row['cascading_failure_count']} | {row['mean_steps']} | {row['hit_step_cap']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
