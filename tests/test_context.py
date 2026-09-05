from agent_harness.context import ContextManager
from agent_harness.models import Message, SessionState


def test_render_keeps_board_summary_and_recent_window():
    state = SessionState(user_id="u", session_id="s", summary="important old fact")
    state.messages = [Message(role="user", content=f"m{i}") for i in range(5)]
    board = {"situation": "current situation", "frame": {}, "candidates": [{"case_id": "x"}]}
    context = ContextManager(keep_recent=2).render(state, workspace=board)
    assert "ACTIVE ANALOGY BOARD" in context[0]["content"]
    assert "SESSION SUMMARY" in context[1]["content"]
    assert [item["content"] for item in context[-2:]] == ["m3", "m4"]


def test_tool_observation_is_rendered_as_context_not_user_text():
    state = SessionState(user_id="u", session_id="s")
    state.messages = [Message(role="tool", name="search", content='{"ok": true}')]
    rendered = ContextManager().render(state)
    assert rendered[0]["role"] == "developer"
    assert "TOOL OBSERVATION [search]" in rendered[0]["content"]
