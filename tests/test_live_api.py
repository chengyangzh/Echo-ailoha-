"""Opt-in real Groq smoke test. Normal unit tests never spend API credits."""
import os
from pathlib import Path
import pytest
from agent_harness.app import build_runtime

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("GROQ_API_KEY"),
    reason="Set RUN_LIVE_TESTS=1 and GROQ_API_KEY.",
)
@pytest.mark.asyncio
async def test_real_llm_can_choose_calculator_and_finish(tmp_path: Path):
    runtime = build_runtime(str(tmp_path / "live.db"), str(tmp_path / "trace.jsonl"))
    output = await runtime.run(
        user_id="live-user", session_id="live-session",
        user_input="Use the calculator tool to compute 17 * 19. Return only the number.",
    )
    assert "323" in output
    assert any(e["event"] == "tool" and e.get("tool") == "calculator" for e in runtime.tracer.tail(20))


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("GROQ_API_KEY"),
    reason="Set RUN_LIVE_TESTS=1 and GROQ_API_KEY.",
)
@pytest.mark.asyncio
async def test_real_llm_routes_bare_situation_into_echo_workflow(tmp_path: Path):
    runtime = build_runtime(str(tmp_path / "routing.db"), str(tmp_path / "routing-trace.jsonl"))
    output = await runtime.run(
        user_id="routing-user", session_id="routing-session",
        user_input="我和多个女生暧昧，结果她们发现了彼此的存在，现在都拒绝和我产生恋爱关系。",
    )
    events = runtime.tracer.tail(30)
    assert any(e.get("event") == "tool" and e.get("tool") == "search" and e.get("success") for e in events)
    assert any(e.get("event") == "tool" and e.get("tool") == "read_case" and e.get("success") for e in events)
    assert output.strip()
