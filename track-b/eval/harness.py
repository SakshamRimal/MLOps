#!/usr/bin/env python3
"""Evaluation harness for the agentic research loop - built from scratch.

No evaluation framework is used: this file defines its own graders, metrics,
failure taxonomy and report generator. It talks to the running API over HTTP,
so it measures the real end-to-end behaviour of the feature.

Usage:
    python3 eval/harness.py --base-url http://localhost:8000 --modes single,multi

Outputs:
    eval/results/results.json   raw records (one per query x mode)
    eval/results/report.md      markdown report with tables
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- graders
def _submit_answer_args_ok(args: dict) -> bool:
    if not str(args.get("answer", "")).strip():
        return False
    citations = args.get("citations")
    if citations in (None, ""):
        return True
    if not isinstance(citations, list):
        return False
    for item in citations:
        chunk_id = item.get("chunk_id") if isinstance(item, dict) else item
        if not isinstance(chunk_id, str) or len(chunk_id) < 8:
            return False
    return True


ARG_RULES = {
    "search_kb": lambda a: bool(str(a.get("query", "")).strip()),
    "weather": lambda a: bool(str(a.get("city", "")).strip()),
    "calculator": lambda a: bool(str(a.get("expression", "")).strip()),
    "get_current_time": lambda a: True,
    "submit_answer": _submit_answer_args_ok,
    "ask_user": lambda a: bool(str(a.get("question", "")).strip()),
}

REFUSAL_MARKERS = [
    "not in the knowledge base", "no relevant information", "no information",
    "does not contain", "doesn't contain", "does not mention", "doesn't mention",
    "does not discuss", "doesn't discuss", "does not cover", "doesn't cover",
    "does not address", "does not provide", "doesn't provide", "does not have",
    "not mentioned", "not covered", "not discussed",
    "cannot find", "could not find", "couldn't find", "no such information",
    "unable to find", "no match", "no matching", "nothing in the knowledge base",
    "outside the knowledge base", "i don't have", "i do not have",
    "no access", "not available in", "no mention of", "not addressed",
    "no relevant", "does not include", "isn't covered", "is not covered",
    "no specific information", "unrelated content", "appears to contain",
    "trouble accessing", "having trouble", "cannot access", "could not access",
    "can't access", "difficulty accessing", "not able to access",
]

HONEST_FAILURE_MARKERS = [
    "unavailable", "could not", "couldn't", "cannot", "unable", "failed",
    "timed out", "timeout", "malformed", "no access", "not able", "broken",
    "error", "did not work", "didn't work",
]

# Criteria whose absence means the task itself was not completed (hard failure).
HARD_KEYS = {
    "tools", "tools_any", "requires_citation", "requires_refusal",
    "requires_clarification", "answer_contains_any", "answer_contains_all",
    "answer_regex", "no_fabrication", "honest_failure",
}
# Criteria that degrade quality without meaning the task was not done (soft).
SOFT_KEYS = {"min_searches"}


def used_tools(record: dict) -> set[str]:
    return {t["tool"] for t in record.get("trajectory", []) if t.get("tool")}


def ok_searches(record: dict) -> int:
    return len([
        t for t in record.get("trajectory", [])
        if t.get("tool") == "search_kb" and t.get("status") == "ok"
        and "0 chunks" not in str(t.get("summary", ""))
    ])


def evaluate_expectations(record: dict, expect: dict) -> tuple[bool, list[str], dict]:
    """Apply the declarative criteria of one query. Returns (completed, failed, detail)."""
    answer = (record.get("answer") or "").lower()
    sources = record.get("sources") or []
    fabricated = record.get("fabricated_citations") or []
    tools = used_tools(record)
    results: dict[str, bool] = {}

    if "tools" in expect:
        results["tools"] = all(t in tools for t in expect["tools"])
    if "tools_any" in expect:
        results["tools_any"] = any(t in tools for t in expect["tools_any"])
    if "min_searches" in expect:
        results["min_searches"] = ok_searches(record) >= expect["min_searches"]
    if "requires_citation" in expect:
        results["requires_citation"] = len(sources) > 0
    if "no_fabrication" in expect:
        results["no_fabrication"] = len(fabricated) == 0
    if "requires_refusal" in expect:
        # Accept either an explicit "the knowledge base does not cover this" refusal
        # or an ask_user clarification: both refuse to answer from invalid evidence.
        refused = any(marker in answer for marker in REFUSAL_MARKERS)
        or_clarified = record.get("stop_reason") == "clarification"
        results["requires_refusal"] = (refused or or_clarified) and not sources
    if "requires_clarification" in expect:
        results["requires_clarification"] = record.get("stop_reason") == "clarification"
    if "answer_contains_all" in expect:
        results["answer_contains_all"] = all(s in answer for s in expect["answer_contains_all"])
    if "answer_contains_any" in expect:
        results["answer_contains_any"] = any(s in answer for s in expect["answer_contains_any"])
    if "answer_regex" in expect:
        results["answer_regex"] = re.search(expect["answer_regex"], record.get("answer") or "") is not None
    if "honest_failure" in expect:
        # Did the agent acknowledge the broken tooling (or ask) instead of
        # producing a confident answer backed by sources it never received?
        acknowledged = any(marker in answer for marker in HONEST_FAILURE_MARKERS) or (
            record.get("stop_reason") == "clarification" and bool(record.get("tool_errors"))
        )
        results["honest_failure"] = acknowledged and not fabricated and not sources

    failed = [key for key, ok in results.items() if not ok]
    return (not failed), failed, results


def check_tool_arguments(record: dict) -> dict:
    """Tool-call correctness: valid names + non-empty required arguments."""
    calls = [t for t in record.get("trajectory", []) if t.get("tool")]
    invalid = []
    for call in calls:
        rule = ARG_RULES.get(call["tool"])
        args = call.get("args") or {}
        if rule is None:
            invalid.append({"tool": call["tool"], "reason": "unknown tool"})
        elif not rule(args):
            invalid.append({"tool": call["tool"], "args": args, "reason": "missing/empty required argument"})
    tools = used_tools(record)
    expected = None
    if record.get("_expected_tools"):
        expected = all(t in tools for t in record["_expected_tools"])
    return {
        "total_tool_calls": len(calls),
        "invalid_tool_calls": len(invalid),
        "invalid_detail": invalid,
        "expected_tool_used": expected,
        "tool_sequence": [f"{t['tool']}:{t['status']}" for t in calls],
    }


def classify(record: dict, failed_keys: list[str]) -> tuple[str, list[str]]:
    """Failure taxonomy: hard / soft / cascading soft failure."""
    expect = record.get("expect") or {}
    failed_hard, failed_soft = [], []
    for key in failed_keys:
        # A rejected citation is only task-breaking when citations were required;
        # for weather/calculator answers the system dropped it, so it degrades quality.
        if key == "no_fabrication" and not expect.get("requires_citation"):
            failed_soft.append(key)
        elif key in HARD_KEYS:
            failed_hard.append(key)
        else:
            failed_soft.append(key)

    hit_cap = record.get("stop_reason") == "max_steps"
    reasons = failed_hard + failed_soft + (["hit_max_steps"] if hit_cap else [])

    if not reasons:
        return "success", []

    # A tool call failed during the run and the final output is still defective:
    # the early soft defect propagated into the final answer.
    early_tool_defect = any(
        t.get("action") == "tool" and t.get("status") == "error"
        for t in record.get("trajectory", [])
    )
    if early_tool_defect:
        return "cascading_soft_failure", reasons
    if failed_hard:
        return "hard_failure", reasons
    return "soft_failure", reasons


# --------------------------------------------------------------------------- client
def http_post(url: str, payload: dict, timeout: float = 300, retries: int = 5) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {text[:300]}"
            if exc.code == 429:
                try:
                    retry_after = float(json.loads(text).get("retry_after", 10))
                except Exception:
                    retry_after = 10
                time.sleep(max(retry_after, 2))
                continue
            if exc.code >= 500 and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_error)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_error)
    raise RuntimeError(last_error or "request failed")


def http_get(url: str, timeout: float = 15) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------- runner
def run_one(base_url: str, spec: dict, mode: str, max_steps: int, sleep: float) -> dict:
    payload = {
        "message": spec["query"],
        "mode": mode,
        "max_steps": max_steps,
        "temperature": 0.2,
    }
    if spec.get("inject_failure"):
        payload["inject_failure"] = spec["inject_failure"]

    started = time.time()
    response = http_post(f"{base_url}/chat/research", payload)
    wall_ms = round((time.time() - started) * 1000, 1)
    time.sleep(sleep)

    record = {
        "id": spec["id"],
        "category": spec["category"],
        "mode": mode,
        "query": spec["query"],
        "inject_failure": spec.get("inject_failure"),
        "max_steps": max_steps,
        "expect": spec["expect"],
        "_expected_tools": spec["expect"].get("tools") or spec["expect"].get("tools_any"),
        "response": response,
        "wall_ms": wall_ms,
    }
    record.update({
        "answer": response.get("answer", ""),
        "sources": response.get("sources") or [],
        "fabricated_citations": response.get("fabricated_citations") or [],
        "tool_errors": response.get("tool_errors") or [],
        "tools_used": response.get("tools_used") or [],
        "trajectory": response.get("trajectory") or [],
        "steps": response.get("steps"),
        "stop_reason": response.get("stop_reason"),
        "verified": response.get("verified"),
        "verification": response.get("verification"),
        "tokens": response.get("tokens") or {},
        "latency_ms": response.get("latency_ms"),
        "injection": response.get("injection") or "",
    })

    completed, failed_keys, detail = evaluate_expectations(record, spec["expect"])
    classification, reasons = classify(record, failed_keys)
    record.pop("_expected_tools", None)

    record["completed"] = completed
    record["failed_keys"] = failed_keys
    record["criteria_detail"] = detail
    record["classification"] = classification
    record["failure_reasons"] = reasons
    record["tool_correctness"] = check_tool_arguments({
        **record,
        "_expected_tools": spec["expect"].get("tools") or spec["expect"].get("tools_any"),
    })
    return record


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    completed = [r for r in records if r["completed"]]
    tool_calls = sum(r["tool_correctness"]["total_tool_calls"] for r in records)
    invalid_calls = sum(r["tool_correctness"]["invalid_tool_calls"] for r in records)
    expected = [r for r in records if r["tool_correctness"]["expected_tool_used"] is not None]
    expected_ok = [r for r in expected if r["tool_correctness"]["expected_tool_used"]]
    steps = [r["steps"] for r in records if isinstance(r["steps"], int)]
    totals = [r["tokens"].get("total", 0) for r in records]
    counts = {}
    for r in records:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    return {
        "queries": len(records),
        "completed": len(completed),
        "completion_rate": round(len(completed) / len(records), 3),
        "expected_tool_rate": round(len(expected_ok) / len(expected), 3) if expected else None,
        "tool_calls": tool_calls,
        "invalid_tool_calls": invalid_calls,
        "valid_arg_rate": round((tool_calls - invalid_calls) / tool_calls, 3) if tool_calls else None,
        "mean_steps": round(statistics.mean(steps), 2) if steps else 0,
        "median_steps": statistics.median(steps) if steps else 0,
        "max_steps_seen": max(steps) if steps else 0,
        "hit_step_cap": sum(1 for r in records if r["stop_reason"] == "max_steps"),
        "classifications": counts,
        "tokens_total": sum(totals),
        "tokens_mean": round(statistics.mean(totals), 1) if totals else 0,
        "prompt_tokens": sum(r["tokens"].get("prompt", 0) for r in records),
        "completion_tokens": sum(r["tokens"].get("completion", 0) for r in records),
        "llm_calls": sum(r["tokens"].get("calls", 0) for r in records),
        "mean_latency_ms": round(statistics.mean([r["latency_ms"] for r in records if r["latency_ms"]]), 0),
    }


# --------------------------------------------------------------------------- report
def report_md(env: dict, aggregates: dict, records: list[dict], comparisons: dict) -> str:
    lines = []
    add = lines.append
    add("# Evaluation Report - W16 Task 3: Agentify the Assistant")
    add("")
    add(f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    add(f"- Backend: `{env.get('url')}` | provider `{env.get('provider', '?')}` | circuit `{env.get('circuit_breaker', '?')}`")
    add(f"- Endpoint under test: `POST /chat/research`")
    add(f"- Harness: built from scratch (`eval/harness.py`), no evaluation framework")
    add("")

    add("## 1. Task completion rate")
    add("")
    add("| mode | queries | completed | completion rate |")
    add("|---|---|---|---|")
    for mode, agg in aggregates.items():
        add(f"| {mode} | {agg['queries']} | {agg['completed']} | {agg['completion_rate']:.1%} |")
    add("")

    add("## 2. Tool-call correctness")
    add("")
    add("| mode | expected tool selected | valid arguments | tool calls | invalid calls |")
    add("|---|---|---|---|---|")
    for mode, agg in aggregates.items():
        exp = "n/a" if agg["expected_tool_rate"] is None else f"{agg['expected_tool_rate']:.1%}"
        varg = "n/a" if agg["valid_arg_rate"] is None else f"{agg['valid_arg_rate']:.1%}"
        add(f"| {mode} | {exp} | {varg} | {agg['tool_calls']} | {agg['invalid_tool_calls']} |")
    add("")

    add("## 3. Trajectory length (researcher iterations per query)")
    add("")
    add("| mode | mean | median | max | hit step cap (forced synthesis) |")
    add("|---|---|---|---|---|")
    for mode, agg in aggregates.items():
        add(f"| {mode} | {agg['mean_steps']} | {agg['median_steps']} | {agg['max_steps_seen']} | {agg['hit_step_cap']} |")
    add("")
    add("Per-query tool sequence (each entry is one model decision):")
    add("")
    add("| id | mode | steps | stop | tool sequence |")
    add("|---|---|---|---|---|")
    for r in records:
        seq = " -> ".join(r["tool_correctness"]["tool_sequence"]) or "(no tool)"
        add(f"| {r['id']} | {r['mode']} | {r['steps']} | {r['stop_reason']} | {seq} |")
    add("")

    add("## 4. Failure log")
    add("")
    add("| id | mode | classification | failed criteria | steps | note |")
    add("|---|---|---|---|---|---|")
    failures = [r for r in records if r["classification"] != "success"]
    if not failures:
        add("| - | - | (no failures) | - | - | - |")
    for r in failures:
        note = "; ".join(r["tool_errors"][:2]) or "-"
        if len(note) > 120:
            note = note[:117] + "..."
        add(f"| {r['id']} | {r['mode']} | **{r['classification']}** | {', '.join(r['failure_reasons'])} | {r['steps']} | {note} |")
    add("")
    counts = {}
    for r in records:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    add("Classification totals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    add("")
    add("Taxonomy rules used by this harness:")
    add("")
    add("- **Hard failure** - the task was not completed: required tool never called, required citation/refusal/clarification missing, required answer content missing, or a fabricated citation on a query where citations were required.")
    add("- **Soft failure** - the task completed but degraded: fewer successful searches than the query needs, a citation attached to a non-KB answer that the system rejected, or the step budget exhausted (forced synthesis).")
    add("- **Cascading soft failure** - at least one tool call failed during the run *and* the final output is still defective, i.e. the early defect propagated into the final answer.")
    add("")

    add("## 5. Token and cost accounting")
    add("")
    add("| id | mode | prompt | completion | total | LLM calls | latency (ms) |")
    add("|---|---|---|---|---|---|---|")
    for r in records:
        t = r["tokens"]
        add(
            f"| {r['id']} | {r['mode']} | {t.get('prompt', 0)} | {t.get('completion', 0)} | "
            f"{t.get('total', 0)} | {t.get('calls', 0)} | {r.get('latency_ms') or r.get('wall_ms')} |"
        )
    add("")
    add("Mode comparison (single-agent baseline vs multi-agent):")
    add("")
    add("| metric | single | multi | delta (multi - single) |")
    add("|---|---|---|---|")
    single, multi = aggregates.get("single", {}), aggregates.get("multi", {})
    if single and multi:
        rows = [
            ("completion rate", f"{single['completion_rate']:.1%}", f"{multi['completion_rate']:.1%}", ""),
            ("total tokens", single["tokens_total"], multi["tokens_total"], multi["tokens_total"] - single["tokens_total"]),
            ("mean tokens / query", single["tokens_mean"], multi["tokens_mean"], round(multi["tokens_mean"] - single["tokens_mean"], 1)),
            ("prompt tokens", single["prompt_tokens"], multi["prompt_tokens"], multi["prompt_tokens"] - single["prompt_tokens"]),
            ("LLM calls", single["llm_calls"], multi["llm_calls"], multi["llm_calls"] - single["llm_calls"]),
            ("mean steps", single["mean_steps"], multi["mean_steps"], round(multi["mean_steps"] - single["mean_steps"], 2)),
        ]
        for label, a, b, delta in rows:
            add(f"| {label} | {a} | {b} | {delta} |")
    add("")

    add("## 6. Failure injection")
    add("")
    add("| id | injected failure | mode | honest response? | fabricated citations | tool errors |")
    add("|---|---|---|---|---|---|")
    injections = [r for r in records if r["category"] == "failure_injection"]
    for r in injections:
        honest = "yes" if r["criteria_detail"].get("honest_failure") else "NO"
        add(
            f"| {r['id']} | {r['inject_failure']} | {r['mode']} | {honest} | "
            f"{len(r['fabricated_citations'])} | {len(r['tool_errors'])} |"
        )
    add("")
    if injections:
        add("Observed agent behaviour under injected failure:")
        add("")
        for r in injections:
            first_error = (r["tool_errors"] or ["-"])[0]
            add(f"- `{r['id']}` ({r['mode']}): tool error surfaced -> `{first_error}`; answer starts: \"{(r['answer'] or '')[:140]}\"")
        add("")

    if comparisons:
        add("## 7. Per-query outcome matrix")
        add("")
        ids = sorted({r["id"] for r in records})
        modes = sorted({r["mode"] for r in records})
        add("| id | " + " | ".join(modes) + " |")
        add("|---|" + "---|" * len(modes))
        for qid in ids:
            cells = []
            for mode in modes:
                match = [r for r in records if r["id"] == qid and r["mode"] == mode]
                if not match:
                    cells.append("-")
                    continue
                r = match[0]
                mark = "PASS" if r["completed"] else f"FAIL ({r['classification']})"
                cells.append(mark)
            add(f"| {qid} | " + " | ".join(cells) + " |")
        add("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic research loop evaluation harness")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--modes", default="single,multi")
    parser.add_argument("--queries", default=str(Path(__file__).parent / "queries.json"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--sleep", type=float, default=0.5, help="pause between requests (rate limiter)")
    parser.add_argument("--only", default=None, help="comma separated query ids to run")
    args = parser.parse_args()

    spec = json.loads(Path(args.queries).read_text())
    max_steps = spec.get("max_steps", 6)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    only = set(x.strip() for x in args.only.split(",")) if args.only else None

    queries = [q for q in spec["queries"] if only is None or q["id"] in only]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    health = http_get(f"{args.base_url}/health")
    print(f"[harness] backend={args.base_url} provider={health.get('provider')} queries={len(queries)} modes={modes}")

    records: list[dict] = []
    for mode in modes:
        for index, query in enumerate(queries, start=1):
            print(f"[harness] ({mode}) {index}/{len(queries)} {query['id']}", flush=True)
            try:
                record = run_one(args.base_url, query, mode, max_steps, args.sleep)
            except Exception as exc:  # a crashed request is itself a hard failure
                record = {
                    "id": query["id"], "category": query["category"], "mode": mode,
                    "query": query["query"], "inject_failure": query.get("inject_failure"),
                    "max_steps": max_steps, "expect": query["expect"],
                    "answer": "", "sources": [], "fabricated_citations": [],
                    "tool_errors": [f"request failed: {exc}"], "tools_used": [],
                    "trajectory": [], "steps": 0, "stop_reason": "request_error",
                    "verified": None, "verification": None, "tokens": {},
                    "latency_ms": None, "wall_ms": None, "injection": "",
                    "completed": False, "failed_keys": ["request_error"],
                    "criteria_detail": {}, "classification": "hard_failure",
                    "failure_reasons": ["request_error"],
                    "tool_correctness": check_tool_arguments({"trajectory": []}),
                }
                print(f"[harness]   ERROR: {exc}", flush=True)
            records.append(record)

    aggregates = {mode: aggregate([r for r in records if r["mode"] == mode]) for mode in modes}
    environment = {
        "url": args.base_url,
        "provider": health.get("provider"),
        "model": health.get("provider"),
        "circuit_breaker": health.get("circuit_breaker"),
        "generated": datetime.now(timezone.utc).isoformat(),
        "max_steps": max_steps,
        "modes": modes,
    }

    (out_dir / "results.json").write_text(json.dumps(
        {"environment": environment, "aggregates": aggregates, "records": records},
        indent=2, ensure_ascii=False,
    ))
    (out_dir / "report.md").write_text(report_md(environment, aggregates, records, aggregates))

    print("\n[harness] summary")
    for mode, agg in aggregates.items():
        print(
            f"  {mode:6s} completion={agg['completion_rate']:.1%} "
            f"({agg['completed']}/{agg['queries']}) mean_steps={agg['mean_steps']} "
            f"tokens={agg['tokens_total']} failures={agg['classifications']}"
        )
    print(f"[harness] wrote {out_dir / 'results.json'}")
    print(f"[harness] wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
