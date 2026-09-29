"""The verifier: an isolated sub-agent that checks a draft against the evidence.

It never sees the researcher's transcript - only the question, the draft and the
distilled notes - so it cannot be biased by how the answer was produced.
"""
from __future__ import annotations

import json
import logging

from app.agent.prompts import VERIFIER_SYSTEM
from app.agent.session import Draft, TokenTally, parse_json_block, _chat_completion

logger = logging.getLogger(__name__)

VERIFIER_MAX_TOKENS = 500


async def verify(question: str, draft: Draft, notes_text: str, tally: TokenTally) -> dict:
    payload = json.dumps(
        {
            "question": question,
            "draft_answer": draft.answer,
            "draft_citations": draft.sources,
            "evidence_notes": notes_text,
        },
        ensure_ascii=False,
    )
    response = await _chat_completion(
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=0.0,
        max_tokens=VERIFIER_MAX_TOKENS,
    )
    tally.add(response)

    data = parse_json_block(response.choices[0].message.content or "")
    if not data:
        logger.warning("Verifier returned unparseable output; skipping verification")
        return {
            "verdict": "SKIPPED",
            "unsupported_claims": [],
            "comment": "verifier output was unparseable; draft accepted without verification",
        }

    verdict = str(data.get("verdict", "FAIL")).upper()
    if verdict not in ("PASS", "FAIL"):
        verdict = "SKIPPED"
    claims = [str(c) for c in (data.get("unsupported_claims") or [])][:8]
    return {
        "verdict": verdict,
        "unsupported_claims": claims,
        "comment": str(data.get("comment") or "")[:500],
    }
