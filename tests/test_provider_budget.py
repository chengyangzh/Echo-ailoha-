from agent_harness.context import ContextManager
from agent_harness.llm import ResponsesClient
from agent_harness.models import Message, SessionState


def test_emergency_compaction_keeps_latest_user_turn():
    items = [
        {"role": "developer", "content": "x" * 5000},
        {"role": "user", "content": "old question"},
        {"role": "developer", "content": "y" * 5000},
        {"role": "user", "content": "latest question"},
        {"role": "developer", "content": "latest evidence"},
    ]
    compact = ResponsesClient._compact_input(items, max_chars=500)
    assert any(row.get("role") == "user" and row.get("content") == "latest question" for row in compact)
    assert sum(len(str(row)) for row in compact) < 2000


def test_render_collapses_adjacent_duplicate_user_retries():
    state = SessionState(user_id="u", session_id="s")
    state.messages = [
        Message(role="user", content="same failed request"),
        Message(role="user", content="same failed request"),
        Message(role="user", content="same failed request"),
    ]
    rendered = ContextManager().render(state)
    users = [row for row in rendered if row.get("role") == "user"]
    assert users == [{"role": "user", "content": "same failed request"}]


def test_context_default_request_budget_is_provider_safe():
    manager = ContextManager()
    assert manager.request_chars <= 7000
    assert manager.keep_recent <= 8
