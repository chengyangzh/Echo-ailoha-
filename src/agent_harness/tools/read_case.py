from __future__ import annotations

from typing import Any

from ..cases import CaseAtlas
from ..external import WikipediaAdapter
from ..tooling import Tool, ToolSpec


class ReadCaseTool(Tool):
    spec = ToolSpec(
        name="read_case",
        description=(
            "Inspect one candidate returned by search. For Core Atlas candidates, returns the full "
            "structured CaseCard. For Wikipedia candidates, returns a bounded source extract with "
            "provenance; that extract is evidence, not a validated analogy or normalized CaseCard."
        ),
        parameters={
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["core_atlas", "wikipedia"]},
                "case_id": {"type": "string"},
            },
            "required": ["provider", "case_id"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        atlas: CaseAtlas | None = None,
        wikipedia: WikipediaAdapter | None = None,
    ) -> None:
        self.atlas = atlas or CaseAtlas()
        self.wikipedia = wikipedia or WikipediaAdapter()

    async def execute(
        self,
        *,
        user_id: str,
        session_id: str,
        provider: str,
        case_id: str,
        **_: Any,
    ) -> Any:
        if provider == "core_atlas":
            return self.atlas.get(case_id).full()
        if provider == "wikipedia":
            return await self.wikipedia.read(case_id)
        raise ValueError(f"unsupported_case_provider: {provider}")
