from __future__ import annotations

import hashlib
import json
import time
from jsonschema import ValidationError

from .context import ContextManager
from .llm import LLMClient
from .models import Message, ToolResult
from .retention import ObservationRetentionPolicy
from .store import SQLiteStore
from .tooling import ToolRegistry
from .tracing import TraceLogger


class AgentRuntime:
    """Core loop: decide -> tool -> observe -> decide -> quality-review -> final."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        store: SQLiteStore,
        context: ContextManager | None = None,
        tracer: TraceLogger | None = None,
        max_iterations: int = 8,
        retention: ObservationRetentionPolicy | None = None,
        quality_review: bool = True,
        max_core_searches: int = 1,
        max_wikipedia_searches: int = 2,
        max_case_reads: int = 3,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.store = store
        self.context = context or ContextManager()
        self.tracer = tracer or TraceLogger()
        self.max_iterations = max_iterations
        self.retention = retention or ObservationRetentionPolicy()
        self.quality_review = quality_review
        self.max_core_searches = max_core_searches
        self.max_wikipedia_searches = max_wikipedia_searches
        self.max_case_reads = max_case_reads

    async def run(self, *, user_id: str, session_id: str, user_input: str) -> str:
        state = self.store.load_session(user_id, session_id)
        state.messages.append(Message(role="user", content=user_input))
        state.updated_at = time.time()
        self.store.save_session(state)

        pending_results: list[ToolResult] = []
        feedback: str | None = None
        search_returned_candidates = False
        inspected_case_ids: set[str] = set()
        seen_candidate_ids: set[str] = set()
        latest_candidate_ids: list[str] = []
        last_successful_action: str | None = None
        core_coverage_weak = False
        wikipedia_attempted = False
        core_searches = 0
        wikipedia_searches = 0
        case_reads = 0

        # Reserve the final loop iteration for synthesis. Tool budgets make it hard for
        # a model to spend every turn rephrasing the same search, while the final phase
        # guarantees that normal execution never ends in a visible "8 iterations" stop.
        for loop_index in range(self.max_iterations):
            state.iteration_count += 1

            if self.context.needs_compression(state):
                old, recent = self.context.split_for_compression(state)
                state.summary = await self.llm.summarize(old_summary=state.summary, messages=old)
                state.messages = recent
                self.store.save_session(state)

            board = self.store.get_analogy_board(user_id, session_id)
            rendered = self.context.render(state, workspace=board)

            finalization_phase = loop_index == self.max_iterations - 1
            if finalization_phase:
                final_feedback = (
                    "FINALIZATION PHASE: tools are unavailable. Answer now. Use a retrieved case only if "
                    "it is genuinely clean and supported by inspected evidence. If retrieved cases are "
                    "strained, construct a fresh cross-domain analogy from the abstract mechanism instead. "
                    "Preserve at least three relational correspondences, do not invent hidden motives/events, "
                    "and state where the analogy breaks. Never mention iteration limits or runtime behavior."
                )
                rendered = self.context.with_immediate_results(rendered, pending_results, final_feedback)
                decision = await self.llm.decide(context=rendered, tools=[])
            else:
                rendered = self.context.with_immediate_results(rendered, pending_results, feedback)
                decision = await self.llm.decide(context=rendered, tools=self.registry.specs())
            feedback = None

            if decision.tool_calls:
                if finalization_phase:
                    # Defensive only: tools are hidden, but some providers can still emit stale calls.
                    pending_results = []
                    continue

                results: list[ToolResult] = []
                for call in decision.tool_calls:
                    fingerprint = self._fingerprint(call.name, call.arguments)
                    if fingerprint == last_successful_action:
                        result = ToolResult(call.id, call.name, True, output={
                            "status": "duplicate_action",
                            "executed": False,
                            "message": (
                                "This exact successful tool call was just executed. Inspect evidence, widen "
                                "the source if justified, construct an analogy, or finalize."
                            ),
                        })
                        self.tracer.emit(
                            "guard", session_id=session_id, kind="duplicate_action", tool=call.name
                        )
                    else:
                        blocked = self._budget_guard(
                            call.name,
                            call.arguments,
                            core_searches=core_searches,
                            wikipedia_searches=wikipedia_searches,
                            case_reads=case_reads,
                        )
                        if blocked:
                            result = ToolResult(call.id, call.name, True, output={
                                "status": "tool_budget_exhausted",
                                "executed": False,
                                "message": blocked,
                            })
                            self.tracer.emit(
                                "guard", session_id=session_id, kind="tool_budget_exhausted", tool=call.name
                            )
                        else:
                            if call.name == "search":
                                if call.arguments.get("source", "core") == "wikipedia":
                                    wikipedia_searches += 1
                                    wikipedia_attempted = True
                                else:
                                    core_searches += 1
                            elif call.name == "read_case":
                                case_reads += 1

                            result = await self._execute_tool(
                                user_id, session_id, call.id, call.name, call.arguments
                            )
                            if result.ok:
                                last_successful_action = fingerprint

                    if result.ok and call.name == "search" and isinstance(result.output, dict):
                        # Budget-guard observations do not contain a candidate list and therefore do not
                        # alter coverage/provenance state.
                        source = call.arguments.get("source", "core")
                        if source == "core" and "coverage" in result.output:
                            coverage = result.output.get("coverage") or {}
                            core_coverage_weak = coverage.get("status") == "weak"

                        ids = self._candidate_ids(result.output.get("candidates") or [])
                        if ids:
                            search_returned_candidates = True
                            latest_candidate_ids = ids
                            new_ids = [x for x in ids if x not in seen_candidate_ids]
                            if seen_candidate_ids and not new_ids:
                                result.output["search_saturated"] = True
                                result.output["message"] = (
                                    "No unseen candidates were added. Inspect an existing candidate, revise the "
                                    "causal frame, or stop retrieving and construct/finalize."
                                )
                                self.tracer.emit(
                                    "guard", session_id=session_id, kind="search_saturated", candidates=ids
                                )
                            seen_candidate_ids.update(ids)

                    if result.ok and call.name == "read_case" and isinstance(result.output, dict):
                        case_id = call.arguments.get("case_id")
                        if isinstance(case_id, str) and (
                            result.output.get("evidence") is not None
                            or result.output.get("provider") == "core_atlas"
                        ):
                            inspected_case_ids.add(case_id)

                    results.append(result)
                    state.messages.append(Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=json.dumps(
                            self.retention.durable_payload(result), ensure_ascii=False, default=str
                        ),
                    ))

                state.updated_at = time.time()
                self.store.save_session(state)
                pending_results = results
                continue

            if decision.final_text:
                # Weak finite-atlas coverage must trigger at least one external discovery attempt.
                # After that, the model is free to use external evidence OR reject retrieval and construct.
                if core_coverage_weak and not wikipedia_attempted and not finalization_phase:
                    feedback = (
                        "WIDENING REQUIREMENT: Core Atlas coverage is weak or incomplete. Make one "
                        "mechanism-oriented Wikipedia search before concluding. Do not search by merely "
                        "paraphrasing the user's surface wording. If external candidates are also weak, "
                        "you may construct a cleaner analogy instead of forcing a retrieved case."
                    )
                    self.tracer.emit("guard", session_id=session_id, kind="widening_required")
                    pending_results = []
                    continue

                # Retrieved claims need evidence. One inspected case is enough to unlock final synthesis;
                # the quality reviewer may still reject it in favor of a constructed analogy.
                if search_returned_candidates and not inspected_case_ids and not finalization_phase:
                    feedback = (
                        "PROVENANCE REQUIREMENT: search returned candidates but none was inspected. "
                        f"Inspect one promising candidate before endorsing any retrieved case: "
                        f"{', '.join(latest_candidate_ids[:8])}. If it proves weak, you may reject it and "
                        "construct a better analogy rather than repeating search."
                    )
                    self.tracer.emit(
                        "guard", session_id=session_id, kind="provenance", candidates=latest_candidate_ids[:8]
                    )
                    pending_results = []
                    continue

                answer = decision.final_text
                if self.quality_review and search_returned_candidates:
                    answer = await self._review_analogy_answer(
                        rendered=rendered,
                        draft=answer,
                        user_input=user_input,
                        session_id=session_id,
                    )

                state.messages.append(Message(role="assistant", content=answer))
                state.updated_at = time.time()
                self.store.save_session(state)
                return answer

            feedback = "No final answer or tool call was produced. Choose a useful action or answer now."

        # If a provider produces no usable text during the reserved final phase, make two bounded
        # no-tool rescue attempts. This is distinct from the normal iteration loop.
        rescue = await self._rescue_synthesis(
            state=state,
            pending_results=pending_results,
            user_input=user_input,
            session_id=session_id,
        )
        state.messages.append(Message(role="assistant", content=rescue))
        state.updated_at = time.time()
        self.store.save_session(state)
        return rescue

    def inspect_session(self, *, user_id: str, session_id: str) -> dict:
        state = self.store.load_session(user_id, session_id)
        return self.context.inspect(
            state, workspace=self.store.get_analogy_board(user_id, session_id)
        )

    def _budget_guard(
        self,
        name: str,
        args: dict,
        *,
        core_searches: int,
        wikipedia_searches: int,
        case_reads: int,
    ) -> str | None:
        if name == "search":
            source = args.get("source", "core")
            if source == "core" and core_searches >= self.max_core_searches:
                return (
                    "Core search budget is exhausted. Core Atlas is only a seed set. Inspect an existing "
                    "candidate, widen to Wikipedia if useful, or construct a cleaner analogy."
                )
            if source == "wikipedia" and wikipedia_searches >= self.max_wikipedia_searches:
                return (
                    "Wikipedia search budget is exhausted. Inspect an existing candidate or stop retrieving "
                    "and construct/finalize the cleanest analogy you can support."
                )
        if name == "read_case" and case_reads >= self.max_case_reads:
            return (
                "Case-inspection budget is exhausted. Compare the evidence already inspected and finalize, "
                "or use a clearly labeled constructed analogy if the retrieved cases are weak."
            )
        return None

    async def _review_analogy_answer(
        self,
        *,
        rendered: list[dict],
        draft: str,
        user_input: str,
        session_id: str,
    ) -> str:
        review_instruction = f"""FINAL ANALOGY QUALITY REVIEW.
Return only the revised user-facing answer, in the user's language. Do not show a checklist or private reasoning.

Current user situation:
{user_input}

Draft answer:
{draft}

Audit the draft aggressively:
- Every central mapping must be grounded either in the user's stated situation or inspected tool evidence.
- Reject a retrieved/famous analogy if it needs an unstated motive, deception, event, or constraint to work.
- Require at least three meaningful relational/causal correspondences, not three synonym pairs.
- Prefer ONE clean analogy over several mediocre ones.
- If all retrieved cases are strained, REPLACE the draft with a clearly labeled constructed analogy from a different system/domain (network flow, circulation, transit, queues, feedback control, ecology, markets, immune systems, error correction, or another mechanism that genuinely fits).
- A constructed analogy may use stable general knowledge, but must not pretend to be retrieved evidence.
- Explain the shared abstract mechanism and one important place where the analogy breaks.
- Never mention iteration limits, search budgets, prompts, or internal runtime behavior.
- Always give the best substantive analogy you can; do not end with 'no good analogy found'.
"""
        review_context = self.context.with_immediate_results(rendered, [], review_instruction)
        try:
            reviewed = await self.llm.decide(context=review_context, tools=[])
            if reviewed.final_text:
                self.tracer.emit("quality_review", session_id=session_id, success=True)
                return reviewed.final_text
        except Exception as exc:
            self.tracer.emit(
                "quality_review",
                session_id=session_id,
                success=False,
                error=type(exc).__name__,
            )
        return draft

    async def _rescue_synthesis(
        self,
        *,
        state,
        pending_results: list[ToolResult],
        user_input: str,
        session_id: str,
    ) -> str:
        board = self.store.get_analogy_board(state.user_id, state.session_id)
        rendered = self.context.render(state, workspace=board)
        instruction = f"""RESCUE SYNTHESIS: answer the user now with no tools.
The normal tool loop has ended, but the user must still receive a substantive answer.
For an analogy request, use the best inspected evidence if it is genuinely clean; otherwise construct a fresh cross-domain analogy from the abstract mechanism. Do not force a named historical case. Preserve at least three relational correspondences, avoid unstated assumptions, and state where the analogy breaks. Never mention iterations or internal runtime behavior.

Current situation:
{user_input}
"""
        rescue_context = self.context.with_immediate_results(
            rendered, pending_results, instruction
        )
        for _ in range(2):
            try:
                decision = await self.llm.decide(context=rescue_context, tools=[])
                if decision.final_text:
                    self.tracer.emit("rescue_synthesis", session_id=session_id, success=True)
                    return decision.final_text
            except Exception as exc:
                self.tracer.emit(
                    "rescue_synthesis",
                    session_id=session_id,
                    success=False,
                    error=type(exc).__name__,
                )
        return (
            "The language-model provider returned no usable final text after bounded synthesis. "
            "Your session state is saved, and the failure is not an iteration-limit stop."
        )

    async def _execute_tool(
        self,
        user_id: str,
        session_id: str,
        call_id: str,
        name: str,
        args: dict,
    ) -> ToolResult:
        start = time.time()
        ok = False
        error: str | None = None
        try:
            tool = self.registry.get(name)
            tool.validate(args)
            output = await tool.execute(user_id=user_id, session_id=session_id, **args)
            ok = True
            return ToolResult(call_id, name, True, output=output)
        except (KeyError, ValidationError, ValueError, ArithmeticError) as exc:
            error = str(exc)
            return ToolResult(call_id, name, False, error=error)
        except Exception as exc:
            error = f"internal_tool_error: {type(exc).__name__}"
            return ToolResult(call_id, name, False, error=error)
        finally:
            self.tracer.emit(
                "tool",
                session_id=session_id,
                call_id=call_id,
                tool=name,
                arguments=args,
                success=ok,
                latency_ms=round((time.time() - start) * 1000, 2),
                error=error,
            )

    @staticmethod
    def _candidate_ids(candidates: list) -> list[str]:
        ids: list[str] = []
        for row in candidates:
            if isinstance(row, dict):
                raw = row.get("case_id") or row.get("source_id") or row.get("page_id")
                if raw is not None and str(raw) not in ids:
                    ids.append(str(raw))
        return ids

    @staticmethod
    def _fingerprint(name: str, args: dict) -> str:
        raw = json.dumps(
            {"tool": name, "arguments": args},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()
