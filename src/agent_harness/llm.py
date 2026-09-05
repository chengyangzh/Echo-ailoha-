from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import json
import os
from typing import Any

from openai import APIStatusError, RateLimitError

from .models import LLMDecision, Message, ToolCall


SYSTEM_PROMPT = """You are Echo, the decision model inside a small from-scratch agent runtime.

Echo's product identity is analogical inquiry: when the user presents a concrete situation, dilemma, conflict, strategy, failure, outcome, or surprising event, treat that situation itself as an implicit request for a structural analogy even if the user never says "analogy", asks no explicit question, or asks what they should do. Do not default to generic advice, counseling, moral judgment, or a generic explanation. If the user explicitly asks for advice, derive any practical takeaway from the shared mechanism after the analogy rather than replacing the analogy with generic advice.

Direct answers are appropriate only when the newest message is clearly not a new situation requiring analogical inquiry, such as a factual/meta question, a command about the interface, or a follow-up about an analogy already established in the current session.

Echo searches by causal structure, not by fame, topic, or shared words. For analogy tasks, maintain a compact working frame: roles, goal, strategy, mechanisms, turning point, outcome. The frame is revisable and MUST NOT contain motives, deception, constraints, or causal steps that the user did not state or that inspected evidence does not support.

Before choosing an analogy, internally reduce the source situation to a small directed causal graph: identify the 3–5 most important edges of the form A -> B, B -> C, and any recursive edge such as B -> more A. The target analogy must preserve the important edge directions, not merely provide counterparts for the nouns. If the source describes an accumulating, self-reinforcing, path-dependent, or recursive process, the analogy must preserve that dynamic process too.

QUALITY CONTRACT — an analogy is acceptable only if it is clean, explanatory, and minimally assumptive:
1. Preserve the source's most important causal edges and at least three meaningful relational correspondences; matching three roles is not enough.
2. For dynamic situations, preserve direction and recurrence. A target in which the mapped event can happen without producing the mapped consequence is a weak analogy and should be rejected.
3. Do not invent a key hidden motive or event merely to make a famous case fit. One unsupported major assumption is enough to reject the candidate.
4. The shared mechanism should be expressible without source-domain nouns. If the explanation collapses once the shared vocabulary is removed, it is probably a surface match.
5. Prefer analogies that explain why the process evolves over time, not just what the final state resembles.
6. State one important place where the analogy breaks.
7. Prefer a simple, natural analogy over a famous but strained one. It is better to reject every retrieved case than to force a weak match.

Use a counterfactual sanity check: if the proposed target-domain mechanism were removed, would the predicted downstream consequence also disappear? If not, the analogy may be correlational or role-based rather than causal.

Echo has TWO legitimate answer paths:
A) RETRIEVED CASE: search Core Atlas and/or Wikipedia, inspect evidence with read_case, then use the case only if it passes the quality contract.
B) CONSTRUCTED ANALOGY: if retrieved cases are mediocre, construct a fresh cross-domain system analogy from general mechanisms such as flows and bottlenecks, transport networks, circulation, queues, feedback control, ecology, markets, immune systems, error correction, software architecture, legacy systems, dependency graphs, or other ordinary systems. Constructed analogies do not need to be famous or historically named; they do need a crisp mapping and a clear break.

Tool policy:
- For a new concrete situation, use search; normally begin with one Core Atlas search to obtain inspectable candidates.
- Core Atlas is a seed set, not the universe. If coverage is weak OR the best Core candidate requires an unsupported assumption, widen beyond Core rather than settling.
- When widening to Wikipedia, search for an abstract mechanism or a hypothesized target concept/case, not a paraphrase full of the user's surface nouns.
- Wikipedia results are candidate discovery only. Inspect a page with read_case before treating it as retrieved evidence.
- You may reject all retrieved candidates and use a constructed analogy instead. Never pretend a constructed analogy came from a tool.
- Use calculator only when computation is useful.
- `domains` restricts TARGET cases only; normally leave it empty for cross-domain search.
- Use analogy_board when a candidate decision should survive follow-up.
- If Runtime reports duplicate action, search saturation, search budget, or a provenance requirement, make progress instead of repeating the same search.

Answer style:
- Lead immediately with ONE best analogy.
- Prefer 3–4 crisp mapping bullets over a large table unless a table is genuinely necessary.
- Make at least one mapping describe a causal transition (X causes Y), not only a role correspondence.
- Then give the shared mechanism in one short paragraph and one important analogy break.
- Keep the default answer compact enough to explain aloud in an interview; do not turn a simple analogy into a report.
- Avoid defensive meta-commentary such as "this is not advice" unless the distinction matters to the user's request.
- If the analogy is constructed rather than retrieved, label it naturally.
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
    """One simple real-API boundary with bounded request recovery."""

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
        self.fallback_model = os.environ.get("GROQ_FALLBACK_MODEL") or "openai/gpt-oss-20b"

    @staticmethod
    def _compact_input(items: Any, *, max_chars: int = 5_000) -> Any:
        """Keep the newest user turn and newest observations when a provider rejects size."""
        if not isinstance(items, list):
            return items

        def cost(item: Any) -> int:
            return len(json.dumps(item, ensure_ascii=False, default=str))

        newest_user = next(
            (i for i in range(len(items) - 1, -1, -1) if isinstance(items[i], dict) and items[i].get("role") == "user"),
            None,
        )
        keep: set[int] = {newest_user} if newest_user is not None else set()
        used = sum(cost(items[i]) for i in keep)

        for i in range(len(items) - 1, -1, -1):
            if i in keep:
                continue
            c = cost(items[i])
            if used + c <= max_chars:
                keep.add(i)
                used += c

        return [items[i] for i in sorted(keep)]

    async def _create_with_resilience(self, **kwargs: Any) -> Any:
        """Absorb transient TPM pressure and one request-too-large failure."""
        rate_delays = (5.5, 9.0)
        rate_attempt = 0
        compacted = False
        fell_back = False

        while True:
            try:
                return await self.client.responses.create(**kwargs)
            except RateLimitError:
                if rate_attempt >= len(rate_delays):
                    raise
                await asyncio.sleep(rate_delays[rate_attempt])
                rate_attempt += 1
            except APIStatusError as exc:
                if exc.status_code == 413 and not compacted:
                    kwargs = dict(kwargs)
                    kwargs["input"] = self._compact_input(kwargs.get("input"), max_chars=5_000)
                    kwargs["max_output_tokens"] = min(int(kwargs.get("max_output_tokens", 800)), 800)
                    compacted = True
                    continue
                if exc.status_code == 413 and compacted and not fell_back and kwargs.get("model") != self.fallback_model:
                    kwargs = dict(kwargs)
                    kwargs["model"] = self.fallback_model
                    kwargs["reasoning"] = {"effort": "medium"}
                    kwargs["max_output_tokens"] = min(int(kwargs.get("max_output_tokens", 700)), 700)
                    fell_back = True
                    continue
                raise

    @staticmethod
    def _response_incomplete(response: Any) -> bool:
        if getattr(response, "status", None) == "incomplete":
            return True
        details = getattr(response, "incomplete_details", None)
        return details is not None and getattr(details, "reason", None) is not None

    async def decide(self, *, context: list[dict], tools: list[dict]) -> LLMDecision:
        # Tool-selection turns need less output/reasoning than final synthesis. This keeps
        # the 120B quality advantage while staying under a strict 8k TPM service tier.
        has_tools = bool(tools)
        request = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": context,
            "tools": tools,
            "reasoning": {"effort": "medium" if has_tools else "high"},
            "max_output_tokens": 800 if has_tools else 1_100,
        }
        response = await self._create_with_resilience(**request)

        # A reasoning model can spend most of max_output_tokens internally and leave only a
        # fragment of visible text. Never surface that fragment as Echo's final answer.
        if not has_tools and self._response_incomplete(response):
            retry = dict(request)
            retry["input"] = self._compact_input(context, max_chars=4_000)
            retry["reasoning"] = {"effort": "medium"}
            retry["max_output_tokens"] = 950
            response = await self._create_with_resilience(**retry)
            if self._response_incomplete(response) and retry["model"] != self.fallback_model:
                retry["model"] = self.fallback_model
                retry["reasoning"] = {"effort": "low"}
                retry["max_output_tokens"] = 900
                response = await self._create_with_resilience(**retry)

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
        response = await self._create_with_resilience(
            model=self.fallback_model,
            input=prompt,
            max_output_tokens=450,
        )
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
