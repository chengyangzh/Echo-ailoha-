from __future__ import annotations

from typing import Any

from ..store import SQLiteStore
from ..tooling import Tool, ToolSpec

_FRAME_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "roles": {"type": "array", "items": {"type": "string"}},
        "goal": {"type": "string"},
        "strategy": {"type": "string"},
        "mechanisms": {
            "type": "array",
            "items": {"type": "string"},
        },
        "turning_point": {"type": "string"},
        "outcome": {"type": "string"},
    },
    "required": [],
    "additionalProperties": False,
}


class AnalogyBoardTool(Tool):
    spec = ToolSpec(
        name="analogy_board",
        description=(
            "Persist Echo's session-scoped analogy workspace. Store the current situation/frame "
            "or record why a candidate is selected, rejected, or still being considered. Use get "
            "for tool-assisted follow-ups about earlier candidate decisions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set_frame", "record_candidate", "get"]},
                "situation": {"type": ["string", "null"]},
                "frame": _FRAME_SCHEMA,
                "case_id": {"type": ["string", "null"]},
                "status": {
                    "type": ["string", "null"],
                    "enum": ["selected", "rejected", "considering", None],
                },
                "reason": {"type": ["string", "null"]},
                "mapping": {"type": ["string", "null"]},
                "analogy_break": {"type": ["string", "null"]},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    )

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    async def execute(
        self,
        *,
        user_id: str,
        session_id: str,
        action: str,
        situation: str | None = None,
        frame: dict | None = None,
        case_id: str | None = None,
        status: str | None = None,
        reason: str | None = None,
        mapping: str | None = None,
        analogy_break: str | None = None,
        **_: Any,
    ) -> Any:
        if action == "set_frame":
            if not situation or frame is None:
                raise ValueError("situation and frame are required when action='set_frame'")
            self.store.set_analogy_frame(user_id, session_id, situation, frame)
        elif action == "record_candidate":
            if not case_id or not status:
                raise ValueError("case_id and status are required when action='record_candidate'")
            self.store.record_analogy_candidate(
                user_id,
                session_id,
                case_id=case_id,
                status=status,
                reason=reason or "",
                mapping=mapping or "",
                analogy_break=analogy_break or "",
            )
        return self.store.get_analogy_board(user_id, session_id)
