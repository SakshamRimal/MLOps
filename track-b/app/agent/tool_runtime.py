"""Tool runtime for the research agent.

Every call is bounded (single request + timeout), returns a small JSON payload,
and can be failed on purpose via failure injection so the evaluation harness can
observe how the agent reacts to broken tooling.
"""
from __future__ import annotations

import json
from typing import Any

from app.agent.failures import (
    INJECTION_KB_DOWN,
    INJECTION_MALFORMED,
    INJECTION_TIMEOUT,
    ToolFailure,
)
from app.tools import calculator, get_current_time, weather

MAX_CHUNKS = 4
MAX_CHUNK_CHARS = 700

# Action tools (the model decides when to call these)
RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the indexed knowledge base documents. Returns the most relevant chunks with chunk_id and document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, reworded for a different angle."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Current weather conditions for a city (Open-Meteo).",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City or location name."}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression, e.g. '2**10' or '17.7 * 0.15'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Arithmetic expression."}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "get_current_time", "description": "Current UTC date and time.", "parameters": {"type": "object", "properties": {}}},
    },
    # Terminal actions (the model decides when to stop)
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Stop researching and submit the final answer with confidence and citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Final answer to the user."},
                    "confidence": {"type": "number", "description": "Self-rated confidence 0.0-1.0."},
                    "evidence_sufficient": {"type": "boolean", "description": "True if retrieved evidence supports the answer."},
                    "citations": {
                        "type": "array",
                        "description": "REQUIRED for knowledge-base answers: the exact chunk_id values you received from search_kb. Use an empty array for weather/calculator/time answers. Never invent ids.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "document": {"type": "string"},
                                "chunk_id": {"type": "string"},
                            },
                            "required": ["chunk_id"],
                        },
                    },
                },
                "required": ["answer", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Stop and ask the user a clarifying question when the request is ambiguous or information is missing.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Clarifying question for the user."}},
                "required": ["question"],
            },
        },
    },
]

TERMINAL_ACTIONS = {"submit_answer", "ask_user"}


def _retrieve(query: str, top_k: int):
    """Thin seam so tests can stub retrieval without loading the embedding model."""
    from app.rag.retriever import retrieve

    return retrieve(query, top_k=top_k)


def search_kb(query: str, injection: str = "") -> dict:
    if not query.strip():
        raise ToolFailure("invalid arguments: 'query' must be a non-empty string")

    if injection == INJECTION_KB_DOWN:
        raise ToolFailure("knowledge base unavailable (vector store returned HTTP 503)")
    if injection == INJECTION_TIMEOUT:
        raise ToolFailure("knowledge base search timed out after 8s")

    if injection == INJECTION_MALFORMED:
        # Simulate a broken retriever: wrong container type and truncated records.
        chunks: Any = {"rows": "corrupted", "status": 200}
    else:
        chunks = _retrieve(query, top_k=MAX_CHUNKS)

    if not isinstance(chunks, list):
        raise ToolFailure(f"malformed retrieval output: expected list, got {type(chunks).__name__}")

    results = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or "chunk_id" not in chunk or "text" not in chunk:
            raise ToolFailure("malformed retrieval output: chunk missing chunk_id/text field")
        text = str(chunk.get("text") or "")
        metadata = chunk.get("metadata") or {}
        results.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "document": str(metadata.get("source", "unknown")),
                "distance": round(float(chunk.get("distance", 0.0)), 4),
                "snippet": text[:MAX_CHUNK_CHARS],
            }
        )

    payload = {"query": query, "count": len(results), "results": results}
    if not results:
        payload["note"] = "no relevant chunks found (empty index or all results too distant)"
    return payload


def execute(name: str, args: dict, injection: str = "") -> tuple[bool, dict]:
    """Run one research tool. Never raises: errors are returned as payloads."""
    try:
        if name == "search_kb":
            payload = search_kb(str(args.get("query") or ""), injection)
        elif name == "weather":
            city = str(args.get("city") or "")
            if not city.strip():
                raise ToolFailure("invalid arguments: 'city' must be a non-empty string")
            payload = weather(city)
        elif name == "calculator":
            expression = str(args.get("expression") or "")
            if not expression.strip():
                raise ToolFailure("invalid arguments: 'expression' must be a non-empty string")
            payload = calculator(expression)
        elif name == "get_current_time":
            payload = get_current_time()
        else:
            return False, {"error": f"unknown tool '{name}'"}
    except ToolFailure as e:
        return False, {"error": str(e), "tool": name}
    except Exception as e:  # defensive: a tool must never crash the loop
        return False, {"error": f"{type(e).__name__}: {e}", "tool": name}

    if isinstance(payload, dict) and payload.get("error"):
        return False, {"error": str(payload["error"]), "tool": name}
    return True, payload


def digest(name: str, args: dict, ok: bool, payload: dict) -> str:
    """One-line replacement used when an older tool result is cleared."""
    short_args = json.dumps(args, ensure_ascii=False)[:80]
    if not ok:
        return f"{name}({short_args}) -> ERROR: {str(payload.get('error', ''))[:120]}"
    if name == "search_kb":
        ids = ", ".join(
            f"{r.get('document')}#{r.get('chunk_id', '')}"
            for r in (payload.get("results") or [])[:4]
        )
        return f"search_kb({short_args}) -> {payload.get('count', 0)} chunks: {ids}"
    return f"{name}({short_args}) -> {json.dumps(payload, ensure_ascii=False)[:120]}"
