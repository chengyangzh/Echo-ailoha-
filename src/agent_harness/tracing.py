from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceLogger:
    """Minimal JSONL execution log: tool calls plus rare runtime guards/errors."""

    def __init__(self, path: str = "trace.jsonl") -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **data: Any) -> None:
        row = {"ts": time.time(), "event": event, **data}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def tail(self, n: int = 12) -> list[dict[str, Any]]:
        path = Path(self.path)
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()[-n:] if x.strip()]
