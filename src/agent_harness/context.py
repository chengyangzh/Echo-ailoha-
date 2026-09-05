from __future__ import annotations

import json
from typing import Any
from .models import Message, SessionState, ToolResult


class ContextManager:
    """Summary + recent raw turns + structured board with a provider-safe request budget."""

    def __init__(self, max_chars: int = 16_000, keep_recent: int = 8, request_chars: int = 7_000) -> None:
        self.max_chars = max_chars
        self.keep_recent = keep_recent
        self.request_chars = request_chars

    def render(self, state: SessionState, workspace: dict[str, Any] | None = None) -> list[dict]:
        items: list[dict] = []
        if workspace and (workspace.get("situation") or workspace.get("candidates")):
            items.append({"role": "developer", "content": "ACTIVE ANALOGY BOARD:\n" + json.dumps(workspace, ensure_ascii=False)})
        if state.summary:
            items.append({"role": "developer", "content": "SESSION SUMMARY:\n" + state.summary})

        recent = state.messages[-self.keep_recent:]
        last_user_content: str | None = None
        for m in recent:
            # Manual retry after a provider failure used to persist the same user turn again.
            # Collapse adjacent identical user messages at render time so a retry cannot bloat
            # the next provider request.
            if m.role == "user":
                if m.content == last_user_content:
                    continue
                last_user_content = m.content
                items.append({"role": "user", "content": m.content})
            elif m.role == "tool":
                items.append({"role": "developer", "content": f"TOOL OBSERVATION [{m.name}]: {m.content}"})
            elif m.role == "assistant":
                items.append({"role": "assistant", "content": m.content})
                last_user_content = None
        return items

    def with_immediate_results(self, items: list[dict], results: list[ToolResult], feedback: str | None = None) -> list[dict]:
        out = list(items)
        for r in results:
            out.append({"role": "developer", "content": f"LATEST TOOL RESULT [{r.name}]: " + json.dumps(r.payload(), ensure_ascii=False, default=str)})
        if feedback:
            out.append({"role": "developer", "content": feedback})
        return self._fit(out)

    def _fit(self, items: list[dict]) -> list[dict]:
        """Preserve newest user turn and newest evidence within a strict request budget."""
        def cost(x: dict) -> int:
            return len(json.dumps(x, ensure_ascii=False, default=str))

        if sum(cost(x) for x in items) <= self.request_chars:
            return items

        newest_user = next((i for i in range(len(items) - 1, -1, -1) if items[i].get("role") == "user"), None)
        keep: set[int] = {newest_user} if newest_user is not None else set()
        used = sum(cost(items[i]) for i in keep)

        for i in range(len(items) - 1, -1, -1):
            if i in keep:
                continue
            c = cost(items[i])
            if used + c <= self.request_chars:
                keep.add(i)
                used += c
        return [items[i] for i in sorted(keep)]

    def needs_compression(self, state: SessionState) -> bool:
        return self.current_chars(state) > self.max_chars and len(state.messages) > self.keep_recent

    def split_for_compression(self, state: SessionState) -> tuple[list[Message], list[Message]]:
        cut = max(0, len(state.messages) - self.keep_recent)
        return state.messages[:cut], state.messages[cut:]

    def current_chars(self, state: SessionState) -> int:
        return len(state.summary) + sum(len(m.content) for m in state.messages)

    def inspect(self, state: SessionState, workspace: dict[str, Any] | None = None) -> dict:
        return {
            "session_key": state.key,
            "stored_chars": self.current_chars(state),
            "summary_chars": len(state.summary),
            "stored_messages": len(state.messages),
            "recent_window_size": self.keep_recent,
            "request_char_budget": self.request_chars,
            "needs_compression": self.needs_compression(state),
            "analogy_board": workspace or {"situation": "", "frame": {}, "candidates": []},
        }
