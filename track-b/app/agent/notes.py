"""Structured external notes - the agent's persistent scratchpad.

Raw tool output is verbose (retrieved chunk text, weather JSON). Left in the
conversation it saturates the context after a few iterations. Everything the
agent learns is distilled here instead, so the loop can clear old tool results
while still remembering what it found.
"""
from __future__ import annotations

import json

MAX_NOTE_ITEMS = 6
SNIPPET_CHARS = 140


class EvidenceNotes:
    def __init__(self):
        self.items: list[str] = []
        self.errors: list[str] = []
        self.tools_used: list[str] = []
        self.known_chunks: dict[str, str] = {}  # chunk_id -> document (citation whitelist)
        self.search_count = 0

    # ------------------------------------------------------------------ record
    def record_tool(self, name: str, args: dict, ok: bool, payload: dict) -> None:
        self.tools_used.append(name)

        if name == "search_kb":
            self.search_count += 1

        if not ok:
            error = str(payload.get("error", "unknown error"))
            self.errors.append(f"{name}: {error}")
            self.items.append(f"{name}({_short_args(args)}) -> ERROR: {error[:160]}")
            return

        self.items.append(self._describe(name, args, payload))

    def _describe(self, name: str, args: dict, payload: dict) -> str:
        if name == "search_kb":
            results = payload.get("results") or []
            for r in results:
                if isinstance(r, dict) and r.get("chunk_id"):
                    self.known_chunks[str(r["chunk_id"])] = str(r.get("document", "unknown"))
            if not results:
                return f'search_kb("{args.get("query", "")[:80]}") -> 0 usable chunks (nothing relevant)'
            parts = []
            for r in results[:4]:
                cid = str(r.get("chunk_id", ""))
                doc = str(r.get("document", "?"))
                snippet = str(r.get("snippet", "")).replace("\n", " ")[:SNIPPET_CHARS]
                parts.append(f'{doc}#{cid}: "{snippet}"')
            return f'search_kb("{args.get("query", "")[:80]}") -> ' + " | ".join(parts)

        if name == "weather":
            return (
                f'weather("{args.get("city", "")}") -> {payload.get("condition", "?")}, '
                f'{payload.get("temperature_c", "?")}C, humidity {payload.get("humidity_percent", "?")}% '
                f'({payload.get("location", "")})'
            )

        if name == "calculator":
            return f'calculator("{args.get("expression", "")}") -> {json.dumps(payload, ensure_ascii=False)[:120]}'

        if name == "get_current_time":
            return f"get_current_time() -> {payload.get('utc_time', payload.get('iso', '?'))}"

        return f"{name}({json.dumps(args, ensure_ascii=False)[:120]}) -> {json.dumps(payload, ensure_ascii=False)[:160]}"

    # ------------------------------------------------------------------ render
    def render(self) -> str:
        header = "EVIDENCE NOTES (external scratchpad - persists after tool outputs are cleared):"
        if not self.items and not self.errors:
            return f"{header}\n(empty - no tool has been called yet)"

        lines = [header]
        for item in self.items[-MAX_NOTE_ITEMS:]:
            lines.append(f"- {item}")
        omitted = len(self.items) - MAX_NOTE_ITEMS
        if omitted > 0:
            lines.append(f"({omitted} earlier note(s) omitted)")
        if self.errors:
            lines.append("TOOL ERRORS SO FAR: " + " ; ".join(self.errors[-3:]))
        lines.append(
            f"tools used: {', '.join(sorted(set(self.tools_used))) or 'none'} | "
            f"searches: {self.search_count} | distinct chunks seen: {len(self.known_chunks)}"
        )
        return "\n".join(lines)


def _short_args(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False)[:100]
