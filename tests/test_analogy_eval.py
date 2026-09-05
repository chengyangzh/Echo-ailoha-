import json
from pathlib import Path

from agent_harness.cases import CaseAtlas


def test_controlled_analogy_retrieval_eval():
    """Small deterministic capability eval inspired by relational-vs-surface analogy designs."""
    cases = json.loads((Path(__file__).parents[1] / "eval" / "analogy_retrieval_cases.json").read_text())
    atlas = CaseAtlas()
    for item in cases:
        frame = item["frame"]
        all_results = atlas.search(
            **frame,
            domains=[],
            top_k=len(atlas.all()),
            diversify_domains=False,
        )
        ids = [r["case_id"] for r in all_results]
        for expected in item["expected_top_k"]:
            assert expected in ids[:5], f"{item['id']}: expected {expected} in top 5, got {ids[:5]}"
        for better, worse in item["must_outrank"]:
            assert ids.index(better) < ids.index(worse), f"{item['id']}: {better} should outrank {worse}"
