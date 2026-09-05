from types import SimpleNamespace

import pytest

from agent_harness.llm import ResponsesClient, SYSTEM_PROMPT


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(output=self.output, output_text="summary")


class FakeClient:
    def __init__(self, output):
        self.responses = FakeResponses(output)


@pytest.mark.asyncio
async def test_parses_tool_call_final_text_and_reasoning_summary():
    output = [
        SimpleNamespace(type="function_call", call_id="c1", name="calculator", arguments='{"expression":"2+2"}'),
        SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="final")]),
        SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="brief reasoning summary")]),
    ]
    client = FakeClient(output)
    llm = ResponsesClient(model="m", api_key="x", client=client)
    decision = await llm.decide(context=[{"role": "user", "content": "hi"}], tools=[])
    assert decision.tool_calls[0].arguments == {"expression": "2+2"}
    assert decision.final_text == "final"
    assert decision.reasoning_summary == "brief reasoning summary"
    request = client.responses.requests[0]
    assert "previous_response_id" not in request
    assert request["input"] == [{"role": "user", "content": "hi"}]
    assert request["reasoning"] == {"effort": "high"}


def test_missing_groq_key_is_clean_configuration_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        ResponsesClient()


def test_system_prompt_routes_bare_situations_to_analogy_mode():
    prompt = SYSTEM_PROMPT.lower()
    assert "implicit request for a structural analogy" in prompt
    assert "do not default to generic advice" in prompt
    assert "for a new concrete situation, use search" in prompt
    assert "follow-up about an analogy already established" in prompt
