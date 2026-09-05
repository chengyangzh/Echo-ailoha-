import json
from pathlib import Path

from agent_harness.evaluation import (
    evaluate_frame_predictions,
    frame_extraction_prompt,
    parse_frame_json,
)
from agent_harness.frames import ANALOGY_FRAME_SCHEMA, AnalogyFrame
from agent_harness.tools.search import SearchTool

ROOT = Path(__file__).parents[1]
DATA = ROOT / "eval" / "frame_extraction_controlled.json"


def _items():
    return json.loads(DATA.read_text(encoding="utf-8"))["items"]


def test_frame_eval_has_near_far_and_surface_pressure_conditions():
    items = _items()
    assert len(items) == 20
    assert {item["condition"] for item in items} == {"near", "far", "surface_pressure"}
    assert all(item["critical_mechanisms"] for item in items)
    assert all(item["gold_case_id"] and item["surface_trap_case_id"] for item in items)


def test_gold_frames_are_schema_valid_and_saturate_evaluator():
    predictions = {item["id"]: item["expected_frame"] for item in _items()}
    metrics = evaluate_frame_predictions(DATA, predictions)
    assert metrics.valid_frame_rate == 1.0
    assert metrics.critical_mechanism_hit_rate == 1.0
    assert metrics.mechanism_macro_f1 == 1.0
    assert metrics.forbidden_surface_mechanism_rate == 0.0
    assert metrics.downstream_structural_over_surface_accuracy == 1.0


def test_evaluator_detects_surface_mechanism_substitution():
    cards = {
        card["id"]: card
        for card in json.loads((ROOT / "src/agent_harness/data/cases.json").read_text())
    }
    predictions = {}
    for item in _items():
        frame = dict(item["expected_frame"])
        # Deliberately replace relational mechanisms with the surface trap's mechanisms.
        frame["mechanisms"] = cards[item["surface_trap_case_id"]]["mechanisms"][:5]
        predictions[item["id"]] = frame

    metrics = evaluate_frame_predictions(DATA, predictions)
    assert metrics.valid_frame_rate == 1.0  # valid JSON is not the same as valid reasoning
    assert metrics.critical_mechanism_hit_rate < 0.25
    assert metrics.forbidden_surface_mechanism_rate >= 0.90
    assert metrics.downstream_structural_over_surface_accuracy <= 0.25


def test_frame_prompt_is_strict_while_search_schema_is_partial_compatible():
    prompt = frame_extraction_prompt("A repeated low-value warning makes a later real warning ignored.")
    assert "controlled vocabulary" in prompt.lower()
    assert "surface nouns" in prompt.lower()
    parsed = parse_frame_json('''```json\n{
      "roles": ["sender", "receiver"],
      "goal": "preserve response",
      "strategy": "send warnings",
      "mechanisms": ["signaling", "credibility_decay"],
      "turning_point": "receiver discounts sender",
      "outcome": "real warning ignored"
    }\n```''')
    frame = AnalogyFrame.from_dict(parsed)
    assert frame.mechanisms == ("signaling", "credibility_decay")
    # Controlled evaluation uses a complete AnalogyFrame; the live search tool deliberately
    # accepts a partial frame so provider-side function validation cannot reject a useful
    # intermediate hypothesis before Runtime sees it. The field names and base JSON types stay
    # aligned, while search relaxes completeness constraints such as minItems/minLength.
    search_props = SearchTool.spec.parameters["properties"]
    for name, schema in ANALOGY_FRAME_SCHEMA["properties"].items():
        assert name in search_props
        assert search_props[name]["type"] == schema["type"]
    assert set(ANALOGY_FRAME_SCHEMA["required"]) == {
        "roles", "goal", "strategy", "mechanisms", "turning_point", "outcome"
    }
    assert SearchTool.spec.parameters["required"] == []
