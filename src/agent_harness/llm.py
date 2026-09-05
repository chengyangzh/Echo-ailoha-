from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import json
import os
from typing import Any

from openai import RateLimitError

from .models import LLMDecision, Message, ToolCall


SYSTEM_PROMPT = """You are Echo, the decision model inside a small from-scratch agent runtime.

Echo's product identity is analogical inquiry: when the user presents a concrete situation, dilemma, conflict, strategy, failure, outcome, or surprising event, treat that situation itself as an implicit request for a structural analogy even if the user never says "analogy", asks no explicit question, or asks what they should do. Do not default to generic advice, counseling, moral judgment, or a generic explanation. If the user explicitly asks for advice, derive any practical takeaway from the shared mechanism after the analogy rather than replacing the analogy with generic advice.

Direct answers are appropriate only when the newest message is clearly not a new situation requiring analogical inquiry, such as a factual/meta question, a command about the interface, or a follow-up about an analogy already established in the current session.

Echo searches by causal structure, not by fame, topic, or shared words. For analogy tasks, maintain a compact working frame: roles, goal, strategy, mechanisms, turning point, outcome. The frame is revisable and MUST NOT contain motives, deception, constraints, or causal steps that the user did not state or that inspected evidence does not support.

QUALITY CONTRACT — an analogy is acceptable only if it is clean, explanatory, and minimally assumptive:
1. Preserve at least three meaningful relational or causal correspondences, not just similar nouns.
2. Do not invent a key hidden motive or event merely to make a famous case fit. One unsupported major assumption is enough to reject the candidate.
3. The shared mechanism should be expressible without source-domain nouns. If the explanation collapses once the shared vocabulary is removed, it is probably a surface match.
4. The mapping should explain something about the source situation: a bottleneck, feedback loop, incentive, information pattern, flow constraint, coordination failure, etc.
5. State one important place where the analogy breaks.
6. Prefer a simple, natural analogy over a famous but strained one. It is better to reject every retrieved case than to force a weak match.

Echo has TWO legitimate answer paths:
A) RETRIEVED CASE: search Core Atlas and/or Wikipedia, inspect evidence with read_case, then use the case only if it passes the quality contract.
B) CONSTRUCTED ANALOGY: if retrieved cases are mediocre, construct a fresh cross-domain system analogy from general mechanisms such as flows and bottlenecks, transport networks, circulation, queues, feedback control, ecology, markets, immune systems, error correction, or other ordinary systems. Constructed analogies do not need to be famous or historically named; they do need a crisp mapping and a clear break. A circulation ↔ transit-network style analogy can be excellent when both genuinely share network flow, hubs, capacity, routing, and bottlenecks.

Tool policy:
- For a new concrete situation, use search; normally begin with one Core Atlas search to obtain inspectable candidates.
- Core Atlas is a seed set, not the universe. If coverage is weak OR the best Core candidate requires an unsupported assumption, widen beyond Core rather than settling.
- When widening to Wikipedia, search for an abstract mechanism or a hypothesized target concept/case, not a paraphrase full of the user's surface nouns. For example, search the mechanism pattern (trusted-insider appropriation, bottleneck cascade, coordination under scarce capacity), not the literal story wording.
- Wikipedia results are candidate discovery only. Inspect a page with read_case before treating it as retrieved evidence.
- You may reject all retrieved candidates and use a constructed analogy instead. Never pretend a constructed analogy came from a tool.
- Use calculator only when computation is useful.
- `domains` restricts TARGET cases only; normally leave it empty for cross-domain search.
- Use analogy_board when a candidate decision should survive follow-up.
- If Runtime reports duplicate action, search saturation, search budget, or a provenance requirement, make progress instead of repeating the same search.

Answer style:
- Lead immediately with ONE best analogy.
- Prefer 3–4 crisp mapping bullets over a large table unless a table is genuinely necessary.
- Then give the shared mechanism in one short paragraph and one important analogy break.
- Keep the default answer compact enough to explain aloud in an interview; do not turn a simple analogy into a report.
- Avoid defensive meta-commentary such as "this is not advice" unless the distinction matters to the user's request.
- If the analogy is constructed rather than retrieved, label it naturally (for example, "A cleaner constructed analogy is...").
- Never claim exhaustive search. Do not fabricate tool results or reveal private chain-of-thought.
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
      GROQ_MODEL=openai/gpt-oss-120b  # optional; quality-first default
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
        self.model = model or os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b"

    async def _create_with_rate_limit_retry(self, **kwargs: Any) -> Any:
        """Retry short rolling-window TPM limits inside the provider boundary.

        Groq commonly returns a precise retry-after interval for transient TPM pressure.
        A manual retry a few seconds later succeeds, so the Harness should absorb that
        transient condition instead of making the user resubmit the same turn.
        """
        delays = (5.5, 9.0)
        for attempt in range(len(delays) + 1):
            try:
                return await self.client.responses.create(**kwargs)
            except RateLimitError:
                if attempt >= len(delays):
                    raise
                await asyncio.sleep(delays[attempt])
        raise RuntimeError("unreachable")

    async def decide(self, *, context: list[dict], tools: list[dict]) -> LLMDecision:
        response = await self._create_with_rate_limit_retry(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=context,
            tools=tools,
            reasoning={"effort": "high"},
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
        response = await self._create_with_rate_limit_retry(model=self.model, input=prompt)
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
