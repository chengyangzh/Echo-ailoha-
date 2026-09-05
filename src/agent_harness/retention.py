from __future__ import annotations

from typing import Any
from .models import ToolResult


class ObservationRetentionPolicy:
    """Keep current tool evidence rich, but store only compact external evidence long-term."""

    def durable_payload(self, result: ToolResult) -> dict[str, Any]:
        if not result.ok:
            return result.payload()
        out = result.output
        if result.name == "read_case" and isinstance(out, dict) and out.get("provider") == "wikipedia":
            evidence = " ".join(str(out.get("evidence", "")).split())
            return {"ok": True, "output": {
                "provider": "wikipedia", "case_id": out.get("case_id"), "title": out.get("title"),
                "url": out.get("url"), "evidence_preview": evidence[:220],
            }, "error": None}
        if result.name == "search" and isinstance(out, dict) and out.get("scope") == "wikipedia":
            rows = [{k: row.get(k) for k in ("provider", "case_id", "title", "url")}
                    for row in out.get("candidates", []) if isinstance(row, dict)]
            return {"ok": True, "output": {"scope": "wikipedia", "query": out.get("query", ""), "candidates": rows}, "error": None}
        if result.name == "analogy_board":
            return {"ok": True, "output": {"workspace_persisted": True}, "error": None}
        return result.payload()
