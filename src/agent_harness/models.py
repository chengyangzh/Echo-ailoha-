from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import time
import uuid

Role = Literal["user", "assistant", "tool", "summary"]


@dataclass
class Message:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMDecision:
    final_text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_summary: str | None = None


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    ok: bool
    output: Any = None
    error: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class SessionState:
    user_id: str
    session_id: str
    summary: str = ""
    messages: list[Message] = field(default_factory=list)
    iteration_count: int = 0
    updated_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.user_id}:{self.session_id}"

    @classmethod
    def new(cls, user_id: str, session_id: str | None = None) -> "SessionState":
        return cls(user_id=user_id, session_id=session_id or str(uuid.uuid4()))
