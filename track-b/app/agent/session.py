"""The researcher: a bounded, model-driven agentic loop.

One iteration = one LLM decision. After every tool result the model decides
whether to search again, switch tools, ask the user, or stop. The loop is
bounded by ``max_steps`` and always terminates with an answer or a question.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.agent import tool_runtime as rt
from app.agent.notes import EvidenceNotes
from app.agent.prompts import (
    FORCED_ANSWER_PROMPT,
    RESEARCH_SYSTEM,
    SELF_CHECK_PROMPT,
)
from app.llm_client import llm_client

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 6
MAX_STEPS_CAP = 10


class TokenTally:
    """Accumulates token usage across every LLM call of one request."""

    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.total = 0
        self.calls = 0

    def add(self, response: Any) -> None:
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.prompt += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion += int(getattr(usage, "completion_tokens", 0) or 0)
        self.total += int(getattr(usage, "total_tokens", 0) or 0)

    def merge(self, other: "TokenTally") -> None:
        self.prompt += other.prompt
        self.completion += other.completion
        self.total += other.total
        self.calls += other.calls

    def as_dict(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion, "total": self.total, "calls": self.calls}


@dataclass
class Draft:
    answer: str
    confidence: float
    sources: list[dict] = field(default_factory=list)
    fabricated: list[dict] = field(default_factory=list)
    evidence_sufficient: bool = True
    via: str = "submit_answer"


async def _chat_completion(**kwargs) -> Any:
    """Single seam for LLM calls (tests stub this instead of hitting the API)."""
    return await llm_client.acreate(**kwargs)


def parse_json_block(text: str) -> dict | None:
    """Extract the first JSON object from a model response, tolerating fences."""
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _clean_args(raw: str) -> dict:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _normalize_tool_call(call: Any) -> dict:
    """Accept both SDK objects and plain dicts (tests use dicts)."""
    if isinstance(call, dict):
        function = call.get("function") or {}
        raw = function.get("arguments", "{}") or "{}"
        return {
            "id": call.get("id") or "",
            "name": function.get("name") or "",
            "args": _clean_args(raw if isinstance(raw, str) else json.dumps(raw)),
            "raw": raw if isinstance(raw, str) else json.dumps(raw),
        }
    function = getattr(call, "function", None)
    raw = getattr(function, "arguments", None) or "{}"
    return {
        "id": getattr(call, "id", None) or "",
        "name": getattr(function, "name", None) or "",
        "args": _clean_args(raw if isinstance(raw, str) else json.dumps(raw)),
        "raw": raw if isinstance(raw, str) else json.dumps(raw),
    }


class ResearchSession:
    """Stateful research conversation: messages, notes, trajectory and token usage."""

    def __init__(
        self,
        query: str,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 1200,
        injection: str = "",
        system_prompt: str | None = None,
    ):
        self.query = query
        self.max_steps = max(1, min(int(max_steps), MAX_STEPS_CAP))
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.injection = injection

        self.notes = EvidenceNotes()
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt or RESEARCH_SYSTEM},
            {"role": "system", "content": self.notes.render()},  # index 1: external notes
            {"role": "user", "content": query},
        ]

        self.trajectory: list[dict] = []
        self.tool_errors: list[str] = []
        self.tools_used: list[str] = []
        self.usage = TokenTally()
        self.steps = 0
        self.stop_reason: str | None = None
        self.clarification: str | None = None
        self.draft: Draft | None = None
        self._digests: dict[str, str] = {}

    # ------------------------------------------------------------------ loop
    async def iteration(self) -> str:
        """One LLM decision. Returns 'continue' | 'submitted' | 'clarification'."""
        self.steps += 1
        response = await _chat_completion(
            messages=self.messages,
            tools=rt.RESEARCH_TOOLS,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        self.usage.add(response)

        message = response.choices[0].message
        content = getattr(message, "content", None)
        tool_calls = [_normalize_tool_call(tc) for tc in (getattr(message, "tool_calls", None) or [])]

        if not tool_calls:
            # The model answered in prose instead of calling a terminal action.
            if not (content or "").strip():
                content = "I could not produce an answer within the step budget."
            self._append_assistant(content, [])
            self.trajectory.append(
                {"step": self.steps, "action": "implicit_submit", "tool": None,
                 "args": {}, "status": "accepted", "summary": "answered without a tool call"}
            )
            self.draft = Draft(answer=content.strip(), confidence=0.4, via="implicit")
            return "submitted"

        self._append_assistant(content, tool_calls)

        outcome = "continue"
        for call in tool_calls:
            name = call["name"]
            args = call["args"]

            if name in rt.TERMINAL_ACTIONS:
                if name == "submit_answer":
                    if not str(args.get("answer") or "").strip():
                        # Reject a malformed terminal call so the model can correct it.
                        self._append_tool_result(
                            call["id"],
                            "Invalid submit_answer: 'answer' is required and must be a non-empty string. "
                            "Your arguments must be complete JSON - a truncated tool call is rejected. "
                            "Call submit_answer again with valid arguments (keep the answer concise).",
                        )
                        self.trajectory.append(
                            {"step": self.steps, "action": "submit_answer", "tool": "submit_answer",
                             "args": _bounded(args), "status": "error",
                             "summary": "rejected: empty/invalid submit_answer arguments"}
                        )
                        self.tool_errors.append(f"step {self.steps} submit_answer: invalid arguments")
                        continue
                    self.draft = self._build_draft(args)
                    self.trajectory.append(
                        {"step": self.steps, "action": "submit_answer", "tool": "submit_answer",
                         "args": _bounded(args), "status": "accepted",
                         "summary": f"submitted answer with {len(self.draft.sources)} valid citation(s)"}
                    )
                    self._append_tool_result(call["id"], "Answer received. Pending verification.")
                    outcome = "submitted"
                else:  # ask_user
                    question = str(args.get("question") or "").strip()
                    if not question:
                        self._append_tool_result(
                            call["id"],
                            "Invalid ask_user: 'question' must be a non-empty string. Call ask_user again.",
                        )
                        self.trajectory.append(
                            {"step": self.steps, "action": "ask_user", "tool": "ask_user",
                             "args": _bounded(args), "status": "error",
                             "summary": "rejected: empty ask_user question"}
                        )
                        continue
                    self.clarification = question
                    self.trajectory.append(
                        {"step": self.steps, "action": "ask_user", "tool": "ask_user",
                         "args": _bounded(args), "status": "accepted",
                         "summary": "asked the user for clarification"}
                    )
                    self._append_tool_result(call["id"], "Clarification requested; waiting for the user.")
                    outcome = "clarification"
                continue

            ok, payload = rt.execute(name, args, self.injection)
            result_text = json.dumps(payload, ensure_ascii=False)
            self._append_tool_result(call["id"], result_text)
            self._digests[call["id"]] = rt.digest(name, args, ok, payload)

            self.notes.record_tool(name, args, ok, payload)
            self.tools_used.append(name)
            status = "ok" if ok else "error"
            if not ok:
                self.tool_errors.append(f"step {self.steps} {name}: {payload.get('error')}")
            self.trajectory.append(
                {"step": self.steps, "action": "tool", "tool": name, "args": _bounded(args),
                 "status": status, "summary": rt.digest(name, args, ok, payload)}
            )

        self._clear_old_tool_results()
        self._refresh_notes()
        return outcome

    def inject_feedback(self, text: str) -> None:
        """Append verifier feedback so the researcher gets another turn."""
        self.messages.append({"role": "user", "content": text})

    async def force_final(self) -> None:
        """Bounded stop: the step budget is exhausted, so synthesize from the notes."""
        self.messages.append({"role": "user", "content": FORCED_ANSWER_PROMPT})
        response = await _chat_completion(
            messages=self.messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        self.usage.add(response)
        text = (response.choices[0].message.content or "").strip()
        data = parse_json_block(text)
        if data and data.get("answer"):
            if data.get("citations"):
                self.draft = self._build_draft({
                    "answer": data["answer"],
                    "confidence": data.get("confidence"),
                    "citations": data["citations"],
                })
                self.draft.via = "forced"
            else:
                self.draft = Draft(
                    answer=str(data["answer"]).strip(),
                    confidence=_clamp_confidence(data.get("confidence"), 0.3),
                    via="forced",
                )
        else:
            self.draft = Draft(answer=text or "I could not produce an answer within the step budget.",
                               confidence=0.3, via="forced")
        self.trajectory.append(
            {"step": self.steps + 1, "action": "forced_final", "tool": None, "args": {},
             "status": "accepted", "summary": "step budget exhausted; synthesized from notes"}
        )

    async def self_check(self) -> None:
        """Single-agent baseline: critique + revise inside the same context."""
        draft = self.draft
        payload = json.dumps(
            {"answer": draft.answer, "confidence": draft.confidence,
             "citations": draft.sources, "evidence_notes": self.notes.render()},
            ensure_ascii=False,
        )
        self.messages.append({"role": "user", "content": SELF_CHECK_PROMPT.format(draft=payload)})
        response = await _chat_completion(
            messages=self.messages, temperature=0.0, max_tokens=600,
        )
        self.usage.add(response)
        data = parse_json_block(response.choices[0].message.content or "")
        if data and data.get("answer"):
            issues = data.get("issues") or []
            self.draft = Draft(
                answer=str(data["answer"]).strip(),
                confidence=_clamp_confidence(data.get("confidence"), draft.confidence),
                sources=draft.sources,
                fabricated=draft.fabricated,
                evidence_sufficient=bool(data.get("evidence_sufficient", draft.evidence_sufficient)),
                via="self_check",
            )
            self.trajectory.append(
                {"step": self.steps + 1, "action": "self_check", "tool": None, "args": {},
                 "status": "revised" if data.get("revised") else "ok",
                 "summary": f"in-context self-check: {len(issues)} issue(s) noted"}
            )
        else:
            self.trajectory.append(
                {"step": self.steps + 1, "action": "self_check", "tool": None, "args": {},
                 "status": "unparseable", "summary": "self-check returned unparseable output; kept draft"}
            )

    # ------------------------------------------------------------------ helpers
    def _append_assistant(self, content: str | None, tool_calls: list[dict]) -> None:
        message: dict = {"role": "assistant", "content": content or ""}
        if tool_calls:
            message["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["raw"]}}
                for c in tool_calls
            ]
        self.messages.append(message)

    def _append_tool_result(self, tool_call_id: str, payload_text: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": payload_text})

    def _clear_old_tool_results(self) -> None:
        """Context engineering: keep only the newest tool output at full size."""
        tool_messages = [m for m in self.messages if m.get("role") == "tool"]
        for message in tool_messages[:-1]:
            digest = self._digests.get(message.get("tool_call_id", ""))
            if digest and not str(message.get("content", "")).startswith("[cleared]"):
                message["content"] = f"[cleared] {digest}"

    def _refresh_notes(self) -> None:
        self.messages[1]["content"] = self.notes.render()

    def _build_draft(self, args: dict) -> Draft:
        answer = str(args.get("answer") or "").strip() or "(empty answer)"
        citations = args.get("citations") or []
        sources: list[dict] = []
        fabricated: list[dict] = []

        for item in citations:
            if isinstance(item, str):
                chunk_id, document = item.strip(), ""
            elif isinstance(item, dict):
                chunk_id = str(item.get("chunk_id") or "").strip()
                document = str(item.get("document") or "")
            else:
                continue
            if not chunk_id:
                continue
            match = self._match_known_chunk(chunk_id)
            if match:
                sources.append({"chunk_id": match, "document": self.notes.known_chunks[match]})
            else:
                fabricated.append({"chunk_id": chunk_id, "document": document})

        if fabricated:
            self.trajectory.append(
                {"step": self.steps, "action": "citation_check", "tool": None, "args": {},
                 "status": "error",
                 "summary": f"rejected {len(fabricated)} citation(s) never seen in tool results"}
            )
            self.tool_errors.append(
                f"step {self.steps} citation_check: {len(fabricated)} fabricated citation(s) dropped"
            )

        return Draft(
            answer=answer,
            confidence=_clamp_confidence(args.get("confidence")),
            sources=sources,
            fabricated=fabricated,
            evidence_sufficient=bool(args.get("evidence_sufficient", True)),
        )

    def _match_known_chunk(self, chunk_id: str) -> str | None:
        """Exact id, or a unique >=8-char prefix of a chunk actually retrieved."""
        if chunk_id in self.notes.known_chunks:
            return chunk_id
        if len(chunk_id) < 8:
            return None
        matches = [cid for cid in self.notes.known_chunks if cid.startswith(chunk_id)]
        return matches[0] if len(matches) == 1 else None


def _bounded(value: Any, depth: int = 0) -> Any:
    """Trajectory args: cap long strings, keep structures readable for reports."""
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict) and depth < 3:
        return {str(k)[:60]: _bounded(v, depth + 1) for k, v in list(value.items())[:12]}
    if isinstance(value, list) and depth < 3:
        return [_bounded(v, depth + 1) for v in value[:10]]
    return json.dumps(value, ensure_ascii=False)[:240]


async def drive(session: ResearchSession, on_submit) -> str:
    """Run the loop until a terminal condition; ``on_submit`` may reject a draft."""
    while session.stop_reason is None:
        if session.steps >= session.max_steps:
            await session.force_final()
            session.stop_reason = "max_steps"
            break

        outcome = await session.iteration()

        if outcome == "clarification":
            session.stop_reason = "clarification"
            break
        if outcome == "submitted":
            accepted = await on_submit(session)
            if accepted:
                session.stop_reason = "submitted"
                break
            continue  # rejected (e.g. verifier failed): loop gives the model another turn
    return session.stop_reason or "submitted"
