import json
from pathlib import Path

from agent_harness.evaluation import evaluate_retrieval

ROOT = Path(__file__).parents[1]


def test_controlled_eval_schema_has_near_far_surface_and_unrelated():
    payload = json.loads((ROOT / "eval" / "retrieval_controlled.json").read_text())
    conditions = {item["condition"] for item in payload["items"]}
    assert {"near", "far", "unrelated"} <= conditions
    scored = [item for item in payload["items"] if item["gold_case_id"]]
    assert scored
    assert all(item["surface_trap_case_id"] for item in scored)


def test_selection_stimuli_have_four_controlled_candidate_types():
    payload = json.loads((ROOT / "eval" / "selection_controlled.json").read_text())
    assert len(payload["items"]) == 8
    for item in payload["items"]:
        assert set(item["candidates"]) == {"near", "far", "surface_trap", "unrelated"}


def test_current_echo_structural_scorer_beats_far_surface_traps():
    metrics = evaluate_retrieval(ROOT / "eval" / "retrieval_controlled.json")
    # Critical dissociation: far structural matches should beat surface-similar traps.
    assert metrics.far_structural_over_surface_accuracy >= 0.90, metrics.failures
    # The surface-only control should be materially easier to fool on the same items.
    assert metrics.far_structural_over_surface_accuracy - metrics.far_lexical_over_surface_accuracy >= 0.40


def test_current_echo_has_reasonable_far_recall():
    metrics = evaluate_retrieval(ROOT / "eval" / "retrieval_controlled.json")
    assert metrics.far_recall_at_5 >= 0.75, metrics.failures


def test_unrelated_controls_do_not_cross_pre_registered_match_threshold():
    metrics = evaluate_retrieval(ROOT / "eval" / "retrieval_controlled.json")
    assert metrics.unrelated_false_match_rate <= 0.25, metrics.failures


def test_selection_trial_expansion_has_critical_and_no_match_trials():
    from agent_harness.evaluation import build_selection_trials, selection_prompt

    trials = build_selection_trials(ROOT / "eval" / "selection_controlled.json")
    kinds = [trial["kind"] for trial in trials]
    assert kinds.count("far_vs_surface") == 8
    assert kinds.count("near_vs_unrelated") == 8
    assert kinds.count("no_match") == 2
    critical = next(t for t in trials if t["kind"] == "far_vs_surface")
    prompt = selection_prompt(critical)
    assert "higher-order relational/causal structure" in prompt
    assert "Answer only A or B" in prompt


def test_adversarial_selection_has_role_direction_and_causal_order_controls():
    from agent_harness.evaluation import build_adversarial_selection_trials, selection_prompt

    trials = build_adversarial_selection_trials(ROOT / "eval" / "selection_adversarial.json")
    kinds = [trial["kind"] for trial in trials]
    assert kinds.count("role_direction") == 3
    assert kinds.count("causal_order") == 3
    assert all(trial["expected"] == "A" for trial in trials)
    assert all(trial["control"] for trial in trials)
    prompt = selection_prompt(trials[0])
    assert "higher-order relational/causal structure" in prompt
