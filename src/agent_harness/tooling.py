from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import jsonschema


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


class Tool(ABC):
    spec: ToolSpec

    def validate(self, arguments: dict[str, Any]) -> None:
        jsonschema.validate(arguments, self.spec.parameters)

    @abstractmethod
    async def execute(self, *, user_id: str, session_id: str, **arguments: Any) -> Any:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def specs(self) -> list[dict[str, Any]]:
        return [{
            "type": "function",
            "name": t.spec.name,
            "description": t.spec.description,
            "parameters": t.spec.parameters,
        } for t in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)
