"""Failure-injection support used by the evaluation harness.

One failure is intentionally built into the system so we can observe whether the
agent *recognises* an invalid tool result instead of answering confidently from
missing evidence.
"""
import os

INJECTION_KB_DOWN = "kb_down"
INJECTION_MALFORMED = "malformed_retrieval"
INJECTION_TIMEOUT = "timeout"

VALID_INJECTIONS = (INJECTION_KB_DOWN, INJECTION_MALFORMED, INJECTION_TIMEOUT)


class ToolFailure(Exception):
    """A tool could not complete. The message is surfaced to the model verbatim."""


def resolve_injection(requested: str | None = None) -> str:
    """Resolve the active failure injection: request field wins over env var."""
    value = (requested or os.getenv("FAILURE_INJECTION", "") or "").strip().lower()
    return value if value in VALID_INJECTIONS else ""
