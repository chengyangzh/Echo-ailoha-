# Echo evaluation design

Runtime tests answer **“does the agent loop work?”** This benchmark answers a different question: **“is Echo using relational structure rather than surface resemblance?”**

## Core distinction

Echo varies two dimensions independently:

| condition | surface similarity | structural similarity |
|---|---:|---:|
| near | high | high |
| far | low | high |
| surface trap | high | low |
| unrelated | low | low |

The critical comparison is:

```text
far structural match  >  surface-similar causal trap
```

If Echo only matches nouns/topics, it should fail this contrast.

## Case representation

Core Atlas cases share the same fields used by production search:

```text
roles
goal
strategy
mechanisms
turning_point
outcome
```

Gold cases are built by freezing the causal mechanism first, then constructing surface traps that preserve topic/outcome cues while breaking a critical causal relation.

## Two evaluation layers

### 1. Retrieval

Given a controlled `AnalogyFrame`, rank Core Atlas cases.

Metrics:

- Recall@5
- MRR
- structural-over-surface accuracy
- far structural-over-surface accuracy
- lexical baseline on the same far-vs-surface trials
- unrelated false-match rate

A lexical baseline is included because a high structural score is uninformative if simple word overlap succeeds equally well.

### 2. Frame extraction control

Retrieval can look perfect when the correct frame is handed to it. Production starts from raw user language, so `frame_extraction_controlled.json` also tests whether a predicted frame contains the critical mechanisms and whether that frame would retrieve the intended structural case.

A negative control deliberately substitutes the surface trap's mechanisms while keeping valid JSON. This checks that schema compliance alone is not counted as reasoning success.

## Adversarial controls

Additional controlled stimuli target:

- role-direction reversal;
- causal-order permutation;
- outcome-matched but mechanism-mismatched cases.

These are diagnostics, not claims of human-level analogical reasoning.

## Run

```bash
echo-eval
```

The command is deterministic and spends no API credits. The real API requirement is tested separately by the opt-in runtime smoke test in `tests/test_live_api.py`.
