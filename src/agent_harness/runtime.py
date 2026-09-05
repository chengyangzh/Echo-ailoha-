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
    """Core loop: decide -> tool -> observe -> decide -> final."""

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
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.store = store
        self.context = context or ContextManager()
        self.tracer = tracer or TraceLogger()
        self.max_iterations = max_iterations
        self.retention = retention or ObservationRetentionPolicy()

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

        # Reserve the final iteration for answer synthesis. This keeps the agent
        # bounded without ever turning a normal user request into a bare
        # "iteration limit" response.
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
                    "FINALIZATION PHASE: do not call tools. Answer the user now using the best "
                    "analogy supported by the evidence already available in context. Prefer a "
                    "natural, useful analogy over a needlessly distant one. If evidence is weak, "
                    "state the uncertainty and where the analogy breaks, but still provide the best "
                    "answer you can. Do not mention iteration limits or internal runtime behavior."
                )
                rendered = self.context.with_immediate_results(rendered, pending_results, final_feedback)
                decision = await self.llm.decide(context=rendered, tools=[])
            else:
                rendered = self.context.with_immediate_results(rendered, pending_results, feedback)
                decision = await self.llm.decide(context=rendered, tools=self.registry.specs())
            feedback = None

            if decision.tool_calls:
                if finalization_phase:
                    feedback = "Finalize with the evidence already available; no more tools are allowed."
                    pending_results = []
                    continue
                results: list[ToolResult] = []
                for call in decision.tool_calls:
                    fingerprint = self._fingerprint(call.name, call.arguments)
                    if fingerprint == last_successful_action:
                        result = ToolResult(call.id, call.name, True, output={
                            "status": "duplicate_action",
                            "executed": False,
                            "message": "This exact successful tool call was just executed. Choose a different action or finalize.",
                        })
                        self.tracer.emit("guard", session_id=session_id, kind="duplicate_action", tool=call.name)
                    else:
                        result = await self._execute_tool(user_id, session_id, call.id, call.name, call.arguments)
                        if result.ok:
                            last_successful_action = fingerprint

                    if call.name == "search" and call.arguments.get("source", "core") == "wikipedia":
                        wikipedia_attempted = True

                    if result.ok and call.name == "search" and isinstance(result.output, dict):
                        source = call.arguments.get("source", "core")
                        if source == "core":
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
                                    "No unseen candidates were added. Inspect an existing candidate with read_case, "
                                    "revise the causal frame, or widen the source only if needed."
                                )
                                self.tracer.emit("guard", session_id=session_id, kind="search_saturated", candidates=ids)
                            seen_candidate_ids.update(ids)

                    if result.ok and call.name == "read_case":
                        case_id = call.arguments.get("case_id")
                        if isinstance(case_id, str):
                            inspected_case_ids.add(case_id)

                    results.append(result)
                    state.messages.append(Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=json.dumps(self.retention.durable_payload(result), ensure_ascii=False, default=str),
                    ))

                state.updated_at = time.time()
                self.store.save_session(state)
                pending_results = results
                continue

            if decision.final_text:
                if core_coverage_weak and not wikipedia_attempted:
                    feedback = (
                        "WIDENING REQUIREMENT: the latest Core Atlas search had weak structural coverage. "
                        "Before concluding, call search with source='wikipedia' to widen candidate discovery. "
                        "You still decide the Wikipedia query and which candidate, if any, is worth inspecting."
                    )
                    self.tracer.emit("guard", session_id=session_id, kind="widening_required")
                    pending_results = []
                    continue

                if search_returned_candidates and not inspected_case_ids:
                    feedback = (
                        "PROVENANCE REQUIREMENT: search returned candidates but none was inspected. "
                        f"Call read_case on one of these candidate IDs before endorsing a retrieved analogy: "
                        f"{', '.join(latest_candidate_ids[:8])}. Do not repeat the same search just to bypass this."
                    )
                    self.tracer.emit("guard", session_id=session_id, kind="provenance", candidates=latest_candidate_ids[:8])
                    pending_results = []
                    continue

                state.messages.append(Message(role="assistant", content=decision.final_text))
                state.updated_at = time.time()
                self.store.save_session(state)
                return decision.final_text

            feedback = "No final answer or tool call was produced. Choose a tool or answer the user."

        msg = "I couldn't complete a reliable analogy from the available evidence. Please try the situation once more."
        state.messages.append(Message(role="assistant", content=msg))
        state.updated_at = time.time()
        self.store.save_session(state)
        return msg

    def inspect_session(self, *, user_id: str, session_id: str) -> dict:
        state = self.store.load_session(user_id, session_id)
        return self.context.inspect(state, workspace=self.store.get_analogy_board(user_id, session_id))

    async def _execute_tool(self, user_id: str, session_id: str, call_id: str, name: str, args: dict) -> ToolResult:
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
        raw = json.dumps({"tool": name, "arguments": args}, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
