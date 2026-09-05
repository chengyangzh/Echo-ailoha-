from pathlib import Path

import pytest

from agent_harness.cases import CaseAtlas
from agent_harness.store import SQLiteStore
from agent_harness.tools.analogy_board import AnalogyBoardTool
from agent_harness.tools.calculator import CalculatorTool
from agent_harness.tools.read_case import ReadCaseTool
from agent_harness.tools.search import SearchTool


@pytest.mark.asyncio
async def test_calculator_safe_arithmetic():
    t = CalculatorTool()
    assert await t.execute(user_id="u", session_id="s", expression="(2+3)*4") == 20


@pytest.mark.asyncio
async def test_calculator_rejects_code_execution():
    t = CalculatorTool()
    with pytest.raises(ValueError):
        await t.execute(user_id="u", session_id="s", expression="__import__('os').system('echo hacked')")


@pytest.mark.asyncio
async def test_structural_far_match_outranks_surface_distractor():
    """Inspired by story-analogy experiments: structure should beat shared surface objects.

    Source structure: a hostile actor hides inside an accepted form to cross a defense.
    Trojan Horse is a far-domain structural match. Boy Who Cried Wolf shares salient
    wolf/shepherd story surface with another fable family but has the wrong mechanism.
    """
    t = SearchTool(CaseAtlas())
    out = await t.execute(
        user_id="u",
        session_id="s",
        source="core",
        query="",
        roles=["hidden attacker", "defender", "trusted carrier"],
        goal="cross a protected boundary",
        strategy="hide hostile identity inside something the target accepts",
        mechanisms=["strategic_deception", "information_asymmetry"],
        turning_point="the defender admits the disguised threat",
        outcome="the attacker gains access without a direct breach",
        domains=[],
        top_k=8,
        diversify_domains=False,
    )
    ids = [c["case_id"] for c in out["candidates"]]
    assert "trojan_horse" in ids
    assert "boy_who_cried_wolf" not in ids


@pytest.mark.asyncio
async def test_cross_domain_diversification_when_scores_are_competitive():
    t = SearchTool(CaseAtlas())
    out = await t.execute(
        user_id="u",
        session_id="s",
        source="core",
        query="",
        roles=["strategic actor", "opponent", "unequal resources"],
        goal="improve the overall result rather than win every local contest",
        strategy="accept a local sacrifice to improve global positioning",
        mechanisms=["local_global_tradeoff", "resource_matching"],
        turning_point="resources are evaluated by their system-level effect",
        outcome="a local loss supports a better aggregate outcome",
        domains=[],
        top_k=3,
        diversify_domains=True,
    )
    domains = [c["domain"] for c in out["candidates"]]
    assert len(domains) == len(set(domains))
    assert "tian_ji_horse_racing" in [c["case_id"] for c in out["candidates"]]


@pytest.mark.asyncio
async def test_search_is_explicitly_non_exhaustive():
    t = SearchTool(CaseAtlas())
    out = await t.execute(
        user_id="u", session_id="s", source="core", query="",
        roles=[], goal="coordinate", strategy="act together",
        mechanisms=["coordination"], turning_point="members coordinate", outcome="group resists pressure",
        domains=[], top_k=2, diversify_domains=False,
    )
    assert out["exhaustive"] is False
    assert "not proof" in out["notice"]


@pytest.mark.asyncio
async def test_read_case_requires_real_case_id():
    t = ReadCaseTool(CaseAtlas())
    card = await t.execute(user_id="u", session_id="s", provider="core_atlas", case_id="wells_fargo_cross_sell")
    assert "proxy_failure" in card["mechanisms"]
    with pytest.raises(ValueError, match="case_not_found"):
        await t.execute(user_id="u", session_id="s", provider="core_atlas", case_id="does-not-exist")


@pytest.mark.asyncio
async def test_analogy_board_is_session_scoped(tmp_path: Path):
    store = SQLiteStore(str(tmp_path / "agent.db"))
    board = AnalogyBoardTool(store)
    frame = {
        "roles": ["actor", "evaluator"],
        "goal": "gain approval",
        "strategy": "suppress disagreement",
        "mechanisms": ["signaling", "self_defeating_strategy"],
        "turning_point": "agreement erases evidence of independent value",
        "outcome": "lower perceived competence",
    }
    await board.execute(
        user_id="same-user", session_id="s1", action="set_frame",
        situation="A person agrees with everything to gain approval.", frame=frame,
        case_id=None, status=None, reason=None, mapping=None, analogy_break=None,
    )
    await board.execute(
        user_id="same-user", session_id="s1", action="record_candidate",
        situation=None, frame=None, case_id="boy_who_cried_wolf", status="rejected",
        reason="same feedback shape but wrong social mechanism", mapping="", analogy_break="",
    )
    assert store.get_analogy_board("same-user", "s1")["candidates"][0]["status"] == "rejected"
    assert store.get_analogy_board("same-user", "s2")["candidates"] == []


class FakeWikipedia:
    def __init__(self):
        self.search_queries = []
        self.read_ids = []

    async def search(self, query: str, *, limit: int = 5):
        self.search_queries.append((query, limit))
        return [
            {
                "provider": "wikipedia",
                "case_id": "123",
                "title": "Example political crisis",
                "snippet": "A short lexical search snippet.",
                "url": "https://example.invalid/?curid=123",
                "retrieved_at": 1.0,
                "candidate_only": True,
            }
        ]

    async def read(self, source_id: str, *, max_chars: int = 1200):
        self.read_ids.append(source_id)
        return {
            "provider": "wikipedia",
            "case_id": source_id,
            "title": "Example political crisis",
            "url": "https://example.invalid/?curid=123",
            "retrieved_at": 2.0,
            "evidence": "Bounded source evidence about the event.",
            "evidence_scope": "bounded_source_extract",
            "normalization_status": "not_yet_casecard",
            "notice": "Evidence only.",
        }


@pytest.mark.asyncio
async def test_wikipedia_search_returns_lightweight_candidate_not_casecard():
    wiki = FakeWikipedia()
    t = SearchTool(CaseAtlas(), wiki)
    out = await t.execute(
        user_id="u", session_id="s", source="wikipedia",
        query="political concession increases future demands",
        roles=["coalition leader", "faction"], goal="preserve support",
        strategy="make repeated concessions", mechanisms=["feedback_loop"],
        turning_point="concession changes bargaining expectations",
        outcome="future demands rise", domains=[], top_k=3, diversify_domains=False,
    )
    assert wiki.search_queries == [("political concession increases future demands", 3)]
    candidate = out["candidates"][0]
    assert candidate["candidate_only"] is True
    assert candidate["provider"] == "wikipedia"
    assert "mechanisms" not in candidate
    assert "outcome" not in candidate


@pytest.mark.asyncio
async def test_wikipedia_read_returns_evidence_not_validated_casecard():
    wiki = FakeWikipedia()
    t = ReadCaseTool(CaseAtlas(), wiki)
    packet = await t.execute(
        user_id="u", session_id="s", provider="wikipedia", case_id="123"
    )
    assert wiki.read_ids == ["123"]
    assert packet["evidence_scope"] == "bounded_source_extract"
    assert packet["normalization_status"] == "not_yet_casecard"
    assert "mechanisms" not in packet
    assert "principle" not in packet


def test_registered_tools_have_name_description_and_json_schema(tmp_path: Path):
    from agent_harness.tooling import ToolRegistry
    store = SQLiteStore(str(tmp_path / "registry.db"))
    reg = ToolRegistry()
    for tool in (CalculatorTool(), SearchTool(CaseAtlas()), ReadCaseTool(CaseAtlas()), AnalogyBoardTool(store)):
        reg.register(tool)
    specs = reg.specs()
    assert len(specs) >= 3
    for spec in specs:
        assert spec["name"] and spec["description"]
        assert spec["parameters"]["type"] == "object"


def test_search_provider_schema_accepts_partial_frame_from_live_groq_failure():
    """Regression: provider should not reject a useful partial semantic frame."""
    t = SearchTool(CaseAtlas())
    args = {
        "source": "core",
        "query": "fire alarm false",
        "diversify_domains": True,
        "domains": [],
        "goal": "find analogy",
        "strategy": "structural",
        "turning_point": "",
        "mechanisms": ["credibility_decay"],
        "roles": [],
        "top_k": 5,
    }
    t.validate(args)


@pytest.mark.asyncio
async def test_search_executes_partial_frame_without_inventing_missing_slots():
    t = SearchTool(CaseAtlas())
    out = await t.execute(
        user_id="u",
        session_id="s",
        source="core",
        query="fire alarm false",
        diversify_domains=True,
        domains=[],
        goal="find analogy",
        strategy="structural",
        turning_point="",
        mechanisms=["credibility_decay"],
        roles=[],
        top_k=5,
    )
    assert out["candidates"]
    assert "partial AnalogyFrame" in out["notice"]
    assert "outcome" in out["notice"]


def test_analogy_board_provider_schema_allows_action_specific_arguments(tmp_path: Path):
    store = SQLiteStore(str(tmp_path / "board-schema.db"))
    board = AnalogyBoardTool(store)
    board.validate({"action": "get"})
    board.validate({
        "action": "set_frame",
        "situation": "example",
        "frame": {"mechanisms": ["trust_erosion"]},
    })


@pytest.mark.asyncio
async def test_core_search_reports_strong_coverage_for_clear_structural_match():
    t = SearchTool(CaseAtlas())
    out = await t.execute(
        user_id="u", session_id="s", source="core", query="",
        roles=["signal sender", "signal receivers", "real threat"],
        goal="preserve response to a real warning",
        strategy="repeated low-value warnings",
        mechanisms=["signaling", "credibility_decay", "feedback_loop"],
        turning_point="receivers stop trusting the warning channel",
        outcome="real warning ignored",
        domains=[], top_k=5, diversify_domains=False,
    )
    assert out["coverage"]["status"] == "strong"
    assert out["coverage"]["top_structural_score"] >= out["coverage"]["threshold"]


@pytest.mark.asyncio
async def test_core_search_reports_weak_coverage_when_structure_is_not_represented():
    t = SearchTool(CaseAtlas())
    out = await t.execute(
        user_id="u", session_id="s", source="core", query="",
        roles=["unrelated actor"], goal="unmatched goal", strategy="unmatched strategy",
        mechanisms=["totally_unseen_mechanism"], turning_point="unmatched turn",
        outcome="unmatched outcome", domains=[], top_k=5, diversify_domains=False,
    )
    assert out["coverage"]["status"] == "weak"
    assert "low_structural_score" in out["coverage"]["reasons"]
    assert "widen candidate discovery to Wikipedia" in out["notice"]

@pytest.mark.asyncio
async def test_unmapped_causal_mechanism_marks_core_coverage_weak():
    t = SearchTool(CaseAtlas())
    out = await t.execute(
        user_id="u",
        session_id="s",
        source="core",
        query="",
        roles=["trusted peer", "victim", "scarce opportunity"],
        goal="obtain a scarce opportunity",
        strategy="use privileged trust or access",
        mechanisms=["trusted_insider_appropriation"],
        turning_point="the peer takes the opportunity",
        outcome="the victim loses both opportunity and trust",
        domains=[],
        top_k=5,
        diversify_domains=True,
    )
    assert out["coverage"]["status"] == "weak"
    assert "unmapped_causal_mechanism" in out["coverage"]["reasons"]
    assert out["coverage"]["unmapped_mechanisms"] == ["trusted_insider_appropriation"]
