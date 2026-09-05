import pytest

from agent_harness.cases import CaseAtlas
from agent_harness.tools.search import SearchTool


class FakeWikipedia:
    def __init__(self):
        self.calls = []

    async def search(self, query: str, *, limit: int = 5):
        self.calls.append((query, limit))
        return []


@pytest.mark.asyncio
async def test_provider_can_request_top_k_10_but_execution_is_clamped_to_8():
    wiki = FakeWikipedia()
    tool = SearchTool(CaseAtlas(), wiki)
    args = {
        "source": "wikipedia",
        "query": "trusted insider appropriation scarce opportunity",
        "top_k": 10,
    }

    # Regression for Groq provider-level tool validation: this must not raise.
    tool.validate(args)

    out = await tool.execute(user_id="u", session_id="s", **args)
    assert wiki.calls == [("trusted insider appropriation scarce opportunity", 8)]
    assert "clamped to 8" in out["notice"]
