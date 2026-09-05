import json
from pathlib import Path

import pytest

from agent_harness.cases import CaseAtlas
from agent_harness.context import ContextManager
from agent_harness.llm import ScriptedLLM
from agent_harness.models import LLMDecision, ToolCall
from agent_harness.runtime import AgentRuntime
from agent_harness.store import SQLiteStore
from agent_harness.tooling import ToolRegistry
from agent_harness.tools.analogy_board import AnalogyBoardTool
from agent_harness.tools.calculator import CalculatorTool
from agent_harness.tools.read_case import ReadCaseTool
from agent_harness.tools.search import SearchTool
from agent_harness.tracing import TraceLogger


def make_runtime(tmp_path: Path, llm, *, max_iterations=8, context=None):
    store = SQLiteStore(str(tmp_path / "agent.db"))
    atlas = CaseAtlas()
    reg = ToolRegistry()
    for tool in (CalculatorTool(), SearchTool(atlas), ReadCaseTool(atlas), AnalogyBoardTool(store)):
        reg.register(tool)
    return AgentRuntime(
        llm=llm, registry=reg, store=store,
        context=context or ContextManager(), tracer=TraceLogger(str(tmp_path / "trace.jsonl")),
        max_iterations=max_iterations,
    )


@pytest.mark.asyncio
async def test_direct_answer_without_tool(tmp_path):
    rt = make_runtime(tmp_path, ScriptedLLM([LLMDecision(final_text="hello")]))
    assert await rt.run(user_id="u", session_id="s", user_input="hi") == "hello"
    assert rt.tracer.tail() == []


@pytest.mark.asyncio
async def test_tool_loop_observation_then_final(tmp_path):
    llm = ScriptedLLM([
        LLMDecision(tool_calls=[ToolCall("c1", "calculator", {"expression": "6*7"})]),
        LLMDecision(final_text="42"),
    ])
    rt = make_runtime(tmp_path, llm)
    assert await rt.run(user_id="u", session_id="s", user_input="compute 6*7") == "42"
    assert "42" in json.dumps(llm.calls[1]["context"])
    events = rt.tracer.tail()
    assert len(events) == 1 and events[0]["event"] == "tool"
    assert events[0]["tool"] == "calculator" and events[0]["success"] is True


@pytest.mark.asyncio
async def test_tool_error_becomes_observation_not_crash(tmp_path):
    llm = ScriptedLLM([
        LLMDecision(tool_calls=[ToolCall("c1", "calculator", {"expression": "__import__('os')"})]),
        LLMDecision(final_text="I could not compute that safely."),
    ])
    rt = make_runtime(tmp_path, llm)
    out = await rt.run(user_id="u", session_id="s", user_input="unsafe")
    assert "safely" in out
    assert "unsupported expression" in json.dumps(llm.calls[1]["context"]).lower()
    assert rt.tracer.tail()[0]["success"] is False


@pytest.mark.asyncio
async def test_session_resume_and_isolation(tmp_path):
    llm = ScriptedLLM([
        LLMDecision(final_text="saved"),
        LLMDecision(final_text="resumed"),
        LLMDecision(final_text="separate"),
    ])
    rt = make_runtime(tmp_path, llm)
    await rt.run(user_id="alice", session_id="window-1", user_input="remember red")
    await rt.run(user_id="alice", session_id="window-1", user_input="what did I say?")
    assert "remember red" in json.dumps(llm.calls[1]["context"])
    await rt.run(user_id="alice", session_id="window-2", user_input="new window")
    assert "remember red" not in json.dumps(llm.calls[2]["context"])


@pytest.mark.asyncio
async def test_context_compresses_old_history(tmp_path):
    llm = ScriptedLLM([LLMDecision(final_text="a"), LLMDecision(final_text="b")])
    rt = make_runtime(tmp_path, llm, context=ContextManager(max_chars=120, keep_recent=1))
    await rt.run(user_id="u", session_id="s", user_input="x" * 100)
    await rt.run(user_id="u", session_id="s", user_input="y" * 100)
    state = rt.store.load_session("u", "s")
    assert state.summary
    assert len(state.messages) <= 2


@pytest.mark.asyncio
async def test_provenance_guard_forces_read_before_retrieved_final(tmp_path):
    search_args = {
        "source": "core", "query": "", "roles": ["signal sender", "receivers", "threat"],
        "goal": "get a true warning believed", "strategy": "repeated warnings",
        "mechanisms": ["signaling", "credibility_decay"],
        "turning_point": "receivers stop believing the warning", "outcome": "real danger is ignored",
        "domains": [], "top_k": 5, "diversify_domains": False,
    }
    llm = ScriptedLLM([
        LLMDecision(tool_calls=[ToolCall("s1", "search", search_args)]),
        LLMDecision(final_text="Boy Who Cried Wolf"),
        LLMDecision(tool_calls=[ToolCall("r1", "read_case", {"provider": "core_atlas", "case_id": "boy_who_cried_wolf"})]),
        LLMDecision(final_text="Boy Who Cried Wolf, with a credibility-decay mapping."),
    ])
    rt = make_runtime(tmp_path, llm)
    out = await rt.run(user_id="u", session_id="s", user_input="find an analogy")
    assert "credibility" in out
    third_context = json.dumps(llm.calls[2]["context"])
    assert "PROVENANCE REQUIREMENT" in third_context
    assert "boy_who_cried_wolf" in third_context
    events = rt.tracer.tail(20)
    assert [e["tool"] for e in events if e["event"] == "tool"] == ["search", "read_case"]


@pytest.mark.asyncio
async def test_search_saturation_is_reported_to_model(tmp_path):
    base = {
        "source": "core", "roles": ["signal sender", "receivers", "threat"],
        "goal": "warning believed", "strategy": "repeated warnings",
        "mechanisms": ["signaling", "credibility_decay"],
        "turning_point": "trust collapses", "outcome": "real warning ignored",
        "domains": [], "top_k": 5, "diversify_domains": False,
    }
    llm = ScriptedLLM([
        LLMDecision(tool_calls=[ToolCall("s1", "search", {**base, "query": "first"})]),
        LLMDecision(tool_calls=[ToolCall("s2", "search", {**base, "query": "second"})]),
        LLMDecision(tool_calls=[ToolCall("r1", "read_case", {"provider": "core_atlas", "case_id": "boy_who_cried_wolf"})]),
        LLMDecision(final_text="done"),
    ])
    rt = make_runtime(tmp_path, llm)
    assert await rt.run(user_id="u", session_id="s", user_input="analogy") == "done"
    assert "search_saturated" in json.dumps(llm.calls[2]["context"])
    assert any(e.get("kind") == "search_saturated" for e in rt.tracer.tail(20))


@pytest.mark.asyncio
async def test_duplicate_successful_action_is_not_executed_twice(tmp_path):
    llm = ScriptedLLM([
        LLMDecision(tool_calls=[ToolCall("c1", "calculator", {"expression": "2+2"})]),
        LLMDecision(tool_calls=[ToolCall("c2", "calculator", {"expression": "2+2"})]),
        LLMDecision(final_text="4"),
    ])
    rt = make_runtime(tmp_path, llm)
    assert await rt.run(user_id="u", session_id="s", user_input="2+2") == "4"
    assert len([e for e in rt.tracer.tail(20) if e["event"] == "tool"]) == 1
    assert any(e.get("kind") == "duplicate_action" for e in rt.tracer.tail(20))


@pytest.mark.asyncio
async def test_max_iteration_limit_reserves_final_answer(tmp_path):
    llm = ScriptedLLM([
        LLMDecision(tool_calls=[ToolCall("c0", "calculator", {"expression": "0"})]),
        LLMDecision(tool_calls=[ToolCall("c1", "calculator", {"expression": "1"})]),
        LLMDecision(final_text="Best answer from the evidence already available."),
    ])
    rt = make_runtime(tmp_path, llm, max_iterations=3)
    out = await rt.run(user_id="u", session_id="s", user_input="loop")
    assert out == "Best answer from the evidence already available."
    assert llm.calls[-1]["tools"] == []


class FakeWikipediaForWidening:
    async def search(self, query: str, *, limit: int = 5):
        return [{"provider":"wikipedia","case_id":"9001","title":"External analogue","snippet":"A lightweight candidate discovered outside the Core Atlas.","url":"https://example.invalid/?curid=9001","retrieved_at":1.0,"candidate_only":True}]
    async def read(self, source_id: str, *, max_chars: int = 1200):
        return {"provider":"wikipedia","case_id":source_id,"title":"External analogue","url":"https://example.invalid/?curid=9001","retrieved_at":2.0,"evidence":"Bounded external evidence for the candidate.","evidence_scope":"bounded_source_extract","normalization_status":"not_yet_casecard","notice":"Evidence only."}


def make_runtime_with_wikipedia(tmp_path: Path, llm):
    store=SQLiteStore(str(tmp_path/"agent.db")); atlas=CaseAtlas(); wiki=FakeWikipediaForWidening(); reg=ToolRegistry()
    for tool in (CalculatorTool(), SearchTool(atlas,wiki), ReadCaseTool(atlas,wiki), AnalogyBoardTool(store)): reg.register(tool)
    return AgentRuntime(llm=llm,registry=reg,store=store,context=ContextManager(),tracer=TraceLogger(str(tmp_path/"trace.jsonl")),max_iterations=8)


@pytest.mark.asyncio
async def test_weak_core_coverage_requires_wikipedia_before_final(tmp_path):
    weak_core={"source":"core","query":"","roles":["unrelated actor"],"goal":"do something idiosyncratic","strategy":"unique process with no atlas analogue","mechanisms":["totally_unseen_mechanism"],"turning_point":"unmatched event","outcome":"unmatched outcome","domains":[],"top_k":5,"diversify_domains":False}
    llm=ScriptedLLM([LLMDecision(tool_calls=[ToolCall("s1","search",weak_core)]),LLMDecision(final_text="I will stop at the Core Atlas."),LLMDecision(tool_calls=[ToolCall("s2","search",{"source":"wikipedia","query":"external analogue unusual mechanism","top_k":3})]),LLMDecision(tool_calls=[ToolCall("r1","read_case",{"provider":"wikipedia","case_id":"9001"})]),LLMDecision(final_text="External analogue after widening.")])
    rt=make_runtime_with_wikipedia(tmp_path,llm); out=await rt.run(user_id="u",session_id="s",user_input="find a case")
    assert out=="External analogue after widening."
    assert "WIDENING REQUIREMENT" in json.dumps(llm.calls[2]["context"])
    events=rt.tracer.tail(20); assert any(e.get("kind")=="widening_required" for e in events); tools=[e for e in events if e.get("event")=="tool"]; assert [e["tool"] for e in tools]==["search","search","read_case"]; assert tools[1]["arguments"]["source"]=="wikipedia"


@pytest.mark.asyncio
async def test_strong_core_coverage_does_not_force_wikipedia(tmp_path):
    strong_core={"source":"core","query":"","roles":["signal sender","signal receivers","real threat"],"goal":"preserve response to a real warning","strategy":"repeatedly use a warning channel for low-value alerts","mechanisms":["signaling","credibility_decay","feedback_loop"],"turning_point":"receivers learn the channel is unreliable","outcome":"a genuine warning is ignored","domains":[],"top_k":5,"diversify_domains":False}
    llm=ScriptedLLM([LLMDecision(tool_calls=[ToolCall("s1","search",strong_core)]),LLMDecision(tool_calls=[ToolCall("r1","read_case",{"provider":"core_atlas","case_id":"boy_who_cried_wolf"})]),LLMDecision(final_text="Boy Who Cried Wolf.")])
    rt=make_runtime_with_wikipedia(tmp_path,llm); out=await rt.run(user_id="u",session_id="s",user_input="find a case")
    assert out=="Boy Who Cried Wolf."
    assert not any(e.get("kind")=="widening_required" for e in rt.tracer.tail(20))


@pytest.mark.asyncio
async def test_final_iteration_hides_tools_and_synthesizes(tmp_path):
    llm=ScriptedLLM([LLMDecision(tool_calls=[ToolCall("c1","calculator",{"expression":"2+2"})]),LLMDecision(tool_calls=[ToolCall("c2","calculator",{"expression":"3+3"})]),LLMDecision(final_text="A bounded final answer.")])
    rt=make_runtime(tmp_path,llm,max_iterations=3); answer=await rt.run(user_id="u",session_id="s",user_input="A difficult situation")
    assert answer=="A bounded final answer."
    assert llm.calls[-1]["tools"]==[]
    assert "Stopped after" not in answer
