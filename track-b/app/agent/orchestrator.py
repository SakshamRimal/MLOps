"""Orchestrator: drives the researcher loop and (optionally) the verifier.

mode="single" -> one agent, self-check inside the same context (baseline)
mode="multi"  -> researcher + isolated verifier, bounded revision rounds
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.agent.failures import resolve_injection
from app.agent.prompts import VERIFIER_FEEDBACK
from app.agent.session import (
    DEFAULT_MAX_STEPS,
    ResearchSession,
    TokenTally,
    drive,
)
from app.agent.verifier import verify as verify_draft

logger = logging.getLogger(__name__)

MAX_VERIFY_ROUNDS = 2

VALID_MODES = ("single", "multi")


async def run_research(
    query: str,
    *,
    mode: str = "multi",
    max_steps: int = DEFAULT_MAX_STEPS,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 1200,
    inject_failure: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    mode = mode if mode in VALID_MODES else "multi"
    injection = resolve_injection(inject_failure)

    session = ResearchSession(
        query,
        max_steps=max_steps,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        injection=injection,
    )

    verifier_tally = TokenTally()
    verification: dict | None = None
    verified: bool | None = None
    verify_rounds = 0

    if mode == "single":
        async def on_submit(sess: ResearchSession) -> bool:
            nonlocal verified
            await sess.self_check()          # in-context self-check, no new agent
            verified = True                  # self-verification, not independent
            return True
    else:
        async def on_submit(sess: ResearchSession) -> bool:
            nonlocal verified, verify_rounds, verification
            verify_rounds += 1
            result = await verify_draft(query, sess.draft, sess.notes.render(), verifier_tally)
            verification = {**result, "round": verify_rounds}
            if result["verdict"] == "PASS":
                verified = True
                return True
            if result["verdict"] == "SKIPPED":
                verified = None
                return True
            if verify_rounds >= MAX_VERIFY_ROUNDS:
                verified = False             # out of budget: accept the last draft
                return True
            sess.inject_feedback(
                VERIFIER_FEEDBACK.format(
                    claims=", ".join(result["unsupported_claims"]) or "see comment",
                    comment=result["comment"] or "no detail given",
                )
            )
            return False

    await drive(session, on_submit)

    tokens = TokenTally()
    tokens.merge(session.usage)
    tokens.merge(verifier_tally)

    fabricated: list[dict] = []
    clarification = None
    if session.stop_reason == "clarification":
        answer = session.clarification or "Could you clarify your question?"
        sources: list[dict] = []
        confidence = None
        clarification = answer
    else:
        draft = session.draft
        answer = draft.answer if draft else "I could not produce an answer within the step budget."
        sources = draft.sources if draft else []
        confidence = draft.confidence if draft else None
        fabricated = draft.fabricated if draft else []

    if verification is not None:
        verification["llm_calls"] = verifier_tally.calls
        verification["tokens"] = verifier_tally.as_dict()

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "mode": mode,
        "steps": session.steps,
        "stop_reason": session.stop_reason or "submitted",
        "clarification": clarification,
        "verified": verified,
        "verification": verification,
        "tools_used": sorted(set(session.tools_used)),
        "trajectory": session.trajectory,
        "fabricated_citations": fabricated,
        "tool_errors": session.tool_errors,
        "tokens": tokens.as_dict(),
        "latency_ms": round((time.time() - started) * 1000, 1),
        "injection": injection,
        "max_steps": session.max_steps,
    }
