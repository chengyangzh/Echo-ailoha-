from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from typing import Any

from .models import LLMDecision, Message, ToolCall


SYSTEM_PROMPT = """You are Echo, the decision model inside a small from-scratch agent runtime.

Echo's product identity is analogical inquiry: when the user presents a concrete situation, dilemma, conflict, strategy, failure, outcome, or surprising event, treat that situation itself as an implicit request for a structural analogy even if the user never says "analogy", asks no explicit question, or asks what they should do. Do not default to generic advice, counseling, moral judgment, or a generic explanation. For a new situation, the normal workflow is to retrieve candidate analogies, inspect evidence, compare causal structure, and then answer through the analogy. If the user explicitly asks for advice, derive any practical takeaway from the shared mechanism after the analogy rather than replacing the analogy with generic advice.

Direct answers are appropriate only when the newest message is clearly not a new situation requiring analogical inquiry, such as a factual/meta question, a command about the interface, or a follow-up about an analogy already established in the current session.

Echo searches for analogies by causal structure, not just shared words. For analogy tasks, reason with a compact working frame: roles, goal, strategy, mechanisms, turning point, outcome. The working frame may be partial and revisable.

Tool policy:
- For a new concrete situation, use search to retrieve candidate analogies before giving a substantive answer.
- Use calculator only when computation is useful.
- Start analogy retrieval with the Core Atlas and widen to Wikipedia only when useful.
- Search results are candidates. Inspect a candidate with read_case before endorsing it.
- Prefer shared causal mechanism over shared topic or vocabulary. Far analogies can be stronger than surface matches.
- Treat the first AnalogyFrame as revisable. If evidence contradicts it, revise rather than defend it.
- `domains` restricts TARGET cases only; normally leave it empty for cross-domain search.
- Use analogy_board when a candidate decision should survive follow-up.
- Never present a constructed example as a retrieved case. Retrieved claims must be grounded in read_case evidence.
- If Runtime reports duplicate action, search saturation, or a provenance requirement, make progress instead of repeating the same search.

For a substantive analogy answer: name the best case found in the current search space, explain the mapping and shared mechanism, and state where the analogy breaks. Never claim exhaustive search. Do not fabricate tool results or reveal private chain-of-thought.
"""


class LLMClient(ABC):
    @abstractmethod
    async def decide(self, *, context: list[dict], tools: list[dict]) -> LLMDecision:
        raise NotImplementedError

    async def summarize(self, *, old_summary: str, messages: list[Message]) -> str:
        joined = " | ".join(f"{m.role}:{m.content[:180]}" for m in messages[-20:])
        return (old_summary + " | " + joined).strip(" |")[-4000:]


class ResponsesClient(LLMClient):
    """One simple real-API boundary.

    Echo deliberately uses fresh Responses API requests every iteration. Runtime owns
    conversation state and tool observations, so no provider-specific continuation
    protocol is needed.

    Normal demo configuration is Groq:
      GROQ_API_KEY=...
      GROQ_MODEL=openai/gpt-oss-20b   # optional
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, model: str | None = None, *, api_key: str | None = None, client: Any = None) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key and client is None:
            raise RuntimeError("No LLM API key configured. Set GROQ_API_KEY.")

        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=key, base_url=self.BASE_URL)
        self.client = client
        self.model = model or os.environ.get("GROQ_MODEL") or "openai/gpt-oss-20b"

    async def decide(self, *, context: list[dict], tools: list[dict]) -> LLMDecision:
        response = await self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=context,
            tools=tools,
        )

        calls: list[ToolCall] = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for item in response.output:
            kind = getattr(item, "type", None)
            if kind == "function_call":
                args = item.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append(ToolCall(id=item.call_id, name=item.name, arguments=args))
            elif kind == "message":
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) in {"output_text", "text"}:
                        text_parts.append(getattr(part, "text", ""))
            elif kind == "reasoning":
                for part in getattr(item, "summary", []) or []:
                    text = getattr(part, "text", None)
                    if text:
                        reasoning_parts.append(text)

        return LLMDecision(
            final_text="\n".join(filter(None, text_parts)).strip() or None,
            tool_calls=calls,
            reasoning_summary="\n".join(filter(None, reasoning_parts)).strip() or None,
        )

    async def summarize(self, *, old_summary: str, messages: list[Message]) -> str:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)
        prompt = (
            "Compress old session history. Preserve user goals, confirmed facts, completed "
            "actions, unresolved items, and useful tool findings; discard repetition.\n\n"
            f"Existing summary:\n{old_summary or '(none)'}\n\nHistory:\n{transcript}\n\n"
            "Return only the compact summary."
        )
        response = await self.client.responses.create(model=self.model, input=prompt)
        return response.output_text.strip()


class ScriptedLLM(LLMClient):
    """Deterministic test double for Runtime tests."""

    def __init__(self, decisions: list[LLMDecision]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []

    async def decide(self, *, context: list[dict], tools: list[dict]) -> LLMDecision:
        self.calls.append({"context": context, "tools": tools})
        if not self.decisions:
            raise RuntimeError("No scripted decisions left")
        return self.decisions.pop(0)
