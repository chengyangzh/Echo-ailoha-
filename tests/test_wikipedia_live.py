import os

import pytest

from agent_harness.external import WikipediaAdapter

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EXTERNAL_TESTS") != "1",
    reason="Set RUN_EXTERNAL_TESTS=1 to run the live Wikipedia smoke test",
)


@pytest.mark.external
@pytest.mark.asyncio
async def test_live_wikipedia_search_then_bounded_read():
    adapter = WikipediaAdapter()
    candidates = await adapter.search("Cuban Missile Crisis brinkmanship", limit=3)
    assert candidates
    first = candidates[0]
    assert first["provider"] == "wikipedia"
    assert first["candidate_only"] is True

    evidence = await adapter.read(first["case_id"])
    assert evidence["provider"] == "wikipedia"
    assert evidence["evidence_scope"] == "bounded_source_extract"
    assert evidence["normalization_status"] == "not_yet_casecard"
    assert 1 <= len(evidence["evidence"]) <= 1400
