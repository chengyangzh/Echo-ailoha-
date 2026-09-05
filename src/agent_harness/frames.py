from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from jsonschema import validate



ANALOGY_FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        },
        "goal": {"type": "string", "minLength": 1},
        "strategy": {"type": "string", "minLength": 1},
        "mechanisms": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True,
            "description": (
                "Open semantic labels are allowed at the provider boundary. Prefer Echo's "
                "controlled mechanism vocabulary when possible; the search tool normalizes "
                "reasonable synonyms instead of rejecting an otherwise valid tool call."
            ),
        },
        "turning_point": {"type": "string", "minLength": 1},
        "outcome": {"type": "string", "minLength": 1},
    },
    "required": ["roles", "goal", "strategy", "mechanisms", "turning_point", "outcome"],
    "additionalProperties": False,
}




def analogy_search_properties() -> dict[str, Any]:
    """Relaxed provider-facing fields for search.

    The strict AnalogyFrame schema is still used for controlled evaluation. Search, however,
    accepts a partial frame because function-calling models often identify the core mechanism
    before every descriptive slot. Runtime/tool code fills missing fields with neutral defaults
    rather than letting the provider reject an otherwise useful action.
    """
    return {
        "roles": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
            "description": "Optional actors/roles. Leave [] if not yet identified.",
        },
        "goal": {"type": "string", "description": "Optional goal. Empty is allowed."},
        "strategy": {"type": "string", "description": "Optional strategy. Empty is allowed."},
        "mechanisms": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "maxItems": 5,
            "uniqueItems": True,
            "description": (
                "Optional open semantic mechanism labels. Prefer the causal mechanism when known; "
                "the tool normalizes common synonyms."
            ),
        },
        "turning_point": {"type": "string", "description": "Optional turning point. Empty is allowed."},
        "outcome": {"type": "string", "description": "Optional outcome. Empty is allowed."},
    }

def analogy_frame_properties() -> dict[str, Any]:
    """Return a defensive copy for embedding the frame contract in larger tool schemas."""
    return deepcopy(ANALOGY_FRAME_SCHEMA["properties"])


@dataclass(frozen=True)
class AnalogyFrame:
    roles: tuple[str, ...]
    goal: str
    strategy: str
    mechanisms: tuple[str, ...]
    turning_point: str
    outcome: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnalogyFrame":
        validate(instance=payload, schema=ANALOGY_FRAME_SCHEMA)
        return cls(
            roles=tuple(payload["roles"]),
            goal=payload["goal"],
            strategy=payload["strategy"],
            mechanisms=tuple(payload["mechanisms"]),
            turning_point=payload["turning_point"],
            outcome=payload["outcome"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "goal": self.goal,
            "strategy": self.strategy,
            "mechanisms": list(self.mechanisms),
            "turning_point": self.turning_point,
            "outcome": self.outcome,
        }
