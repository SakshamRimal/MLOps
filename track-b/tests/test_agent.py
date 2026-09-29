"""Unit tests for the agentic research loop.

No network: the LLM seam (`_chat_completion`) and the retriever seam (`_retrieve`)
are replaced with scripted fakes, so every behaviour below is deterministic.
"""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agent import orchestrator as orch
from app.agent import session as session_mod
from app.agent import tool_runtime as rt
from app.agent import verifier as verifier_mod
from app.agent.notes import EvidenceNotes

FAKE_CHUNKS = [
    {
        "chunk_id": "aaa111bbb222ccc333ddd444eee555",
        "text": "9.5 List comprehensions: a list comprehension creates a new list using a pattern.",
        "metadata": {"source": "python.pdf"},
        "distance": 0.31,
    },
    {
        "chunk_id": "fff666ggg777hhh888iii999jjj000",
        "text": "Tuples are immutable sequences created with round brackets.",
        "metadata": {"source": "python.pdf"},
        "distance": 0.42,
    },
]


# --------------------------------------------------------------------------- fakes
def llm_response(text=None, tool_calls=None, prompt=100, completion=50):
    calls = []
    for i, (name, args) in enumerate(tool_calls or [], start=1):
        calls.append(
            SimpleNamespace(
                id=f"call_{i}",
                function=SimpleNamespace(name=name, arguments=json.dumps(args)),
            )
        )
    message = SimpleNamespace(content=text, tool_calls=calls or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
    )


class ScriptedLLM:
    """Pops prepared responses in order; records every call's kwargs."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("scripted LLM exhausted - loop ran more iterations than expected")
        return self.responses.pop(0)


def submit(answer, confidence=0.9, citations=None, sufficient=True):
    return ("submit_answer", {
        "answer": answer,
        "confidence": confidence,
        "evidence_sufficient": sufficient,
        "citations": citations or [],
    })


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- tests
class TestResearchLoop(unittest.TestCase):
    def test_multiple_iterations_before_submit(self):
        """The model decides the tool sequence: two different searches, then submit."""
        fake = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "what is a list comprehension"})]),
            llm_response(tool_calls=[("search_kb", {"query": "list comprehension syntax python"})]),
            llm_response(tool_calls=[submit(
                "A list comprehension builds a list from a pattern.",
                citations=[{"document": "python.pdf", "chunk_id": FAKE_CHUNKS[0]["chunk_id"]}],
            )]),
            llm_response(text='{"answer": "A list comprehension builds a list from a pattern.", "confidence": 0.9, "revised": false, "issues": []}'),
        ])
        with patch.object(session_mod, "_chat_completion", fake), \
             patch.object(rt, "_retrieve", return_value=FAKE_CHUNKS):
            result = run(orch.run_research("what is a list comprehension?", mode="single"))

        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["steps"], 3)
        searches = [t for t in result["trajectory"] if t["tool"] == "search_kb"]
        self.assertEqual(len(searches), 2)
        self.assertNotEqual(searches[0]["args"]["query"], searches[1]["args"]["query"])
        self.assertEqual(result["sources"][0]["chunk_id"], FAKE_CHUNKS[0]["chunk_id"])
        self.assertEqual(result["tokens"]["calls"], 4)  # 3 research + 1 self-check
        self.assertTrue(result["tokens"]["total"] > 0)

    def test_stop_condition_max_steps(self):
        """The loop must terminate at max_steps and still produce an answer."""
        fake = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "one"})]),
            llm_response(tool_calls=[("search_kb", {"query": "two"})]),
            llm_response(text='{"answer": "Best effort answer", "confidence": 0.3}'),
        ])
        with patch.object(session_mod, "_chat_completion", fake), \
             patch.object(rt, "_retrieve", return_value=FAKE_CHUNKS):
            result = run(orch.run_research("never satisfied", mode="single", max_steps=2))

        self.assertEqual(result["stop_reason"], "max_steps")
        self.assertEqual(result["steps"], 2)  # research iterations only
        self.assertEqual(result["answer"], "Best effort answer")
        self.assertEqual(fake.responses, [])

    def test_tool_results_are_cleared_but_notes_survive(self):
        """Context engineering: older tool outputs become digests, notes keep the facts."""
        session = session_mod.ResearchSession("q", max_steps=6)
        fake = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "first"})]),
            llm_response(tool_calls=[("search_kb", {"query": "second"})]),
        ])
        with patch.object(session_mod, "_chat_completion", fake), \
             patch.object(rt, "_retrieve", return_value=FAKE_CHUNKS):
            asyncio.run(session.iteration())
            asyncio.run(session.iteration())

        tool_msgs = [m for m in session.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        self.assertTrue(tool_msgs[0]["content"].startswith("[cleared]"))
        self.assertIn("results", tool_msgs[1]["content"])  # newest stays at full size
        notes_msg = session.messages[1]["content"]
        self.assertIn("EVIDENCE NOTES", notes_msg)
        self.assertIn(FAKE_CHUNKS[0]["chunk_id"], notes_msg)  # facts survive the clearing

    def test_failure_injection_kb_down_is_surfaced_not_hidden(self):
        fake = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "anything"})]),
            llm_response(tool_calls=[submit("I could not verify this.", confidence=0.0)]),
            llm_response(text='{"answer": "I could not verify this.", "confidence": 0.0, "revised": false, "issues": []}'),
        ])
        with patch.object(session_mod, "_chat_completion", fake):
            result = run(orch.run_research(
                "kb fact", mode="single", inject_failure="kb_down"
            ))

        self.assertTrue(result["tool_errors"])
        self.assertIn("unavailable", result["tool_errors"][0])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["fabricated_citations"], [])
        tool_msgs = [m for m in _replay_tool_messages(fake) if m["role"] == "tool"]
        self.assertIn("unavailable", tool_msgs[0]["content"])  # model sees the failure

    def test_malformed_retrieval_is_a_tool_error_not_a_crash(self):
        fake = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "anything"})]),
            llm_response(tool_calls=[submit("Cannot answer.", confidence=0.0)]),
            llm_response(text='{"answer": "Cannot answer.", "confidence": 0.0, "revised": false, "issues": []}'),
        ])
        with patch.object(session_mod, "_chat_completion", fake):
            result = run(orch.run_research(
                "kb fact", mode="single", inject_failure="malformed_retrieval"
            ))
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertTrue(any("malformed" in e for e in result["tool_errors"]))

    def test_fabricated_citation_is_rejected(self):
        fake = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "q"})]),
            llm_response(tool_calls=[submit(
                "Answer citing a chunk nobody retrieved.",
                citations=[{"document": "invented.pdf", "chunk_id": "deadbeef" * 4}],
            )]),
            llm_response(text='{"answer": "Answer citing a chunk nobody retrieved.", "confidence": 0.5, "revised": false, "issues": []}'),
        ])
        with patch.object(session_mod, "_chat_completion", fake), \
             patch.object(rt, "_retrieve", return_value=FAKE_CHUNKS):
            result = run(orch.run_research("q", mode="single"))

        self.assertEqual(result["sources"], [])
        self.assertEqual(len(result["fabricated_citations"]), 1)

    def test_malformed_submit_arguments_are_rejected_and_retried(self):
        """An empty submit_answer must not terminate the loop with an empty answer."""
        fake = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "q"})]),
            llm_response(tool_calls=[("submit_answer", {})]),          # malformed terminal call
            llm_response(tool_calls=[submit("Recovered answer.", confidence=0.7)]),
            llm_response(text='{"answer": "Recovered answer.", "confidence": 0.7, "revised": false, "issues": []}'),
        ])
        with patch.object(session_mod, "_chat_completion", fake), \
             patch.object(rt, "_retrieve", return_value=FAKE_CHUNKS):
            result = run(orch.run_research("q", mode="single"))

        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["answer"], "Recovered answer.")
        rejected = [t for t in result["trajectory"] if t["status"] == "error" and t["tool"] == "submit_answer"]
        self.assertEqual(len(rejected), 1)
        self.assertTrue(any("invalid arguments" in e for e in result["tool_errors"]))

    def test_ask_user_stops_with_clarification(self):
        fake = ScriptedLLM([
            llm_response(tool_calls=[("ask_user", {"question": "Which document do you mean?"})]),
        ])
        with patch.object(session_mod, "_chat_completion", fake):
            result = run(orch.run_research("what about them?", mode="single"))

        self.assertEqual(result["stop_reason"], "clarification")
        self.assertEqual(result["steps"], 1)
        self.assertIn("Which document", result["clarification"])

    def test_empty_search_results_cause_another_search(self):
        """Empty evidence should make the agent search again, not answer from memory."""
        fake = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "obscure topic"})]),
            llm_response(tool_calls=[("search_kb", {"query": "obscure topic rephrased"})]),
            llm_response(tool_calls=[submit("The knowledge base has no information on this.", confidence=0.1)]),
            llm_response(text='{"answer": "The knowledge base has no information on this.", "confidence": 0.1, "revised": false, "issues": []}'),
        ])
        with patch.object(session_mod, "_chat_completion", fake), \
             patch.object(rt, "_retrieve", return_value=[]):
            result = run(orch.run_research("obscure topic?", mode="single"))

        self.assertEqual(result["steps"], 3)
        self.assertEqual(len([t for t in result["trajectory"] if t["tool"] == "search_kb"]), 2)
        self.assertEqual(result["sources"], [])


class TestMultiAgentVerification(unittest.TestCase):
    def test_verifier_rejection_triggers_another_research_round(self):
        researcher = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "q"})]),
            llm_response(tool_calls=[submit("Draft answer without support.", confidence=0.8)]),
            llm_response(tool_calls=[("search_kb", {"query": "q reformulated"})]),
            llm_response(tool_calls=[submit(
                "Revised answer with support.",
                citations=[{"document": "python.pdf", "chunk_id": FAKE_CHUNKS[0]["chunk_id"]}],
            )]),
        ])
        verify_llm = ScriptedLLM([
            llm_response(text='{"verdict": "FAIL", "unsupported_claims": ["claim A"], "comment": "no evidence"}'),
            llm_response(text='{"verdict": "PASS", "unsupported_claims": [], "comment": "supported"}'),
        ])
        with patch.object(session_mod, "_chat_completion", researcher), \
             patch.object(verifier_mod, "_chat_completion", verify_llm), \
             patch.object(rt, "_retrieve", return_value=FAKE_CHUNKS):
            result = run(orch.run_research("q", mode="multi"))

        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["steps"], 4)  # research continued after the failed verdict
        self.assertEqual(result["verification"]["round"], 2)
        self.assertEqual(result["verification"]["verdict"], "PASS")
        self.assertTrue(result["verified"])
        self.assertEqual(len([t for t in result["trajectory"] if t["tool"] == "search_kb"]), 2)

    def test_verification_rounds_are_bounded(self):
        """Two consecutive FAIL verdicts must not loop forever."""
        researcher = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "q"})]),
            llm_response(tool_calls=[submit("Draft 1", confidence=0.8)]),
            llm_response(tool_calls=[submit("Draft 2", confidence=0.8)]),
        ])
        verify_llm = ScriptedLLM([
            llm_response(text='{"verdict": "FAIL", "unsupported_claims": ["x"], "comment": "weak"}'),
            llm_response(text='{"verdict": "FAIL", "unsupported_claims": ["x"], "comment": "weak"}'),
        ])
        with patch.object(session_mod, "_chat_completion", researcher), \
             patch.object(verifier_mod, "_chat_completion", verify_llm), \
             patch.object(rt, "_retrieve", return_value=FAKE_CHUNKS):
            result = run(orch.run_research("q", mode="multi"))

        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["verification"]["round"], 2)
        self.assertFalse(result["verified"])  # accepted only because the budget ran out
        self.assertEqual(result["steps"], 3)

    def test_single_mode_uses_in_context_self_check(self):
        researcher = ScriptedLLM([
            llm_response(tool_calls=[("search_kb", {"query": "q"})]),
            llm_response(tool_calls=[submit("Draft answer", confidence=0.6)]),
            llm_response(text='{"answer": "Checked answer", "confidence": 0.7, "revised": true, "issues": ["tightened claim"]}'),
        ])
        with patch.object(session_mod, "_chat_completion", researcher), \
             patch.object(rt, "_retrieve", return_value=FAKE_CHUNKS):
            result = run(orch.run_research("q", mode="single"))

        self.assertEqual(result["answer"], "Checked answer")
        self.assertEqual(result["verified"], True)
        self.assertIsNone(result["verification"])  # no separate verifier agent
        self.assertEqual(result["tokens"]["calls"], 3)


class TestNotes(unittest.TestCase):
    def test_notes_are_capped(self):
        notes = EvidenceNotes()
        for i in range(8):
            notes.record_tool("search_kb", {"query": f"q{i}"}, True, {
                "count": 1,
                "results": [{"chunk_id": f"id{i}", "document": "python.pdf", "snippet": "x"}],
            })
        rendered = notes.render()
        self.assertIn("earlier note(s) omitted", rendered)
        self.assertLess(len(rendered), 4000)

    def test_known_chunks_whitelist(self):
        notes = EvidenceNotes()
        notes.record_tool("search_kb", {"query": "q"}, True, {
            "count": 1,
            "results": [{"chunk_id": "abc", "document": "python.pdf", "snippet": "x"}],
        })
        self.assertEqual(notes.known_chunks["abc"], "python.pdf")


def _replay_tool_messages(scripted):
    """Pull tool messages back out of the recorded message lists."""
    messages = []
    for call in scripted.calls:
        for message in call.get("messages", []):
            if message.get("role") == "tool":
                messages.append(message)
    return messages


if __name__ == "__main__":
    unittest.main()
