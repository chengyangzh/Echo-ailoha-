# Echo

> **Search by structure, not by words.**

Echo is a minimal analogical reasoning agent. Give it a situation, dilemma, failure, strategy, conflict, or surprising outcome and it searches for a **clean structural analogy**, not merely the most famous or lexically similar case. A new concrete situation is treated as an implicit analogy request. Echo first probes a curated Core Atlas, widens to Wikipedia when the finite atlas is structurally incomplete, and can reject retrieval entirely in favor of a freshly **constructed cross-domain analogy** when that produces a cleaner mapping. Every substantive analogy is checked for unsupported assumptions, relational coverage, and a clear break before it is returned.

```text
many false alerts
→ warning channel loses credibility
→ real alert is ignored

            ↕

The Boy Who Cried Wolf
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

export GROQ_API_KEY='YOUR_KEY'
export GROQ_MODEL='openai/gpt-oss-120b'  # optional; quality-first default

agent-harness --user user-a --session window-1
```

`echo-agent` is an equivalent CLI alias. Open another terminal with the same user but a different session:

```bash
agent-harness --user user-a --session window-2
```

The two windows have independent conversation and analogy-workspace state.

### Useful CLI commands

```text
/multi     enter one multi-line message; finish with /end
/context   inspect what is occupying the current session context
/trace     show recent tool and runtime-guard events
/help
exit
```

`/context` is a small project-specific feature called **Context Lens**. It does not ask another model to explain the context; it exposes the harness state directly, so summary size, recent-window usage, compression state, session isolation, and the active Analogy Board can be inspected rather than inferred.

## Architecture

```text
User
  ↓
AgentRuntime
  ├── ContextManager ── summary + recent window + active board
  ├── ResponsesClient ─ GPT-OSS 120B, high reasoning, typed function calls
  ├── ToolRegistry ──── calculator / search / read_case / analogy_board
  │       └── search/read_case ── Core Atlas OR Wikipedia adapter
  ├── SQLiteStore ───── (user_id, session_id)
  └── TraceLogger ───── compact JSONL execution events
           ↑
       tool result
           └──────────── loop

After an analogy draft is produced, a no-tool **quality review** can replace it entirely if the retrieved case is strained. The runtime then persists only the reviewed answer.
```

The LLM chooses the next action; the runtime owns validation, execution, state, context, persistence, and termination.

## Agent loop

Conceptually:

```python
for iteration in range(MAX_ITERATIONS):
    maybe_compress()
    decision = llm.decide(context, available_tool_schemas)

    if decision.tool_calls:
        results = validate_budget_execute(decision.tool_calls)
        add_tool_observations_to_context(results)
        continue

    if decision.final_text:
        if weak_core_requires_widening():
            continue
        draft = decision.final_text
        reviewed = quality_review(draft, tools=[])
        persist_and_return(reviewed)

# reserved no-tool synthesis / bounded rescue if needed
```

Echo uses the Responses-style function-calling interface through the OpenAI Python SDK pointed at Groq's Responses-compatible endpoint. The quality-first default is `openai/gpt-oss-120b` with `reasoning={"effort": "high"}` for decision turns; `GROQ_MODEL` can still override the model. The model returns typed `function_call` objects; the runtime validates and executes them, then places the resulting observation into the next context it constructs. Each iteration is a fresh provider request: Echo deliberately keeps conversation continuity in SQLite and does not rely on provider-hosted conversation state or `previous_response_id`.

The protocol reference is the OpenAI Responses/function-calling API:

- https://developers.openai.com/api/reference/responses
- https://developers.openai.com/api/docs/guides/function-calling

## Tool registration

Each tool has a machine-readable contract:

```python
ToolSpec(
    name="calculator",
    description="Evaluate a basic arithmetic expression safely.",
    parameters={...JSON Schema...},
)
```

and an executable implementation:

```python
async def execute(self, *, user_id, session_id, **arguments): ...
```

`ToolRegistry` exposes these schemas to the model, so the runtime does not hard-code a fixed `search → read → final` pipeline. Runtime-side validation still happens before execution because model-generated arguments are untrusted input. Provider-facing schemas are intentionally strict about shape and types but permissive about open semantic hypotheses; Echo's stricter complete `AnalogyFrame` is reserved for controlled evaluation.

## Echo-specific reasoning layer

### CaseCard

Every curated case in the Core Atlas is normalized into the same relational representation:

```text
roles / goal / strategy / mechanisms / turning_point / outcome / principle
```

The atlas mixes fables, Chinese and world history, business, institutions, science/technology, and strategy patterns. A live search frame may be partial: Echo can identify a mechanism first, retrieve candidates, and revise the abstraction after inspecting evidence instead of inventing missing fields just to satisfy a schema.

A typical analogy run is:

```text
situation
  ↓
working AnalogyFrame
  ↓
Core search
  ↓
inspect / compare
  ├── clean retrieved case ───────────────┐
  └── weak / assumption-heavy candidate   │
              ↓                            │
      Wikipedia widening                   │
              ↓                            │
      inspect / reject / revise            │
              ├── clean external case ─────┤
              └── still weak               │
                     ↓                     │
           construct fresh analogy         │
                     └──────────────┬───────┘
                                    ↓
                             quality review
                                    ↓
                      mapping + mechanism + break
```

Search and inspection are intentionally separate. A retrieved candidate is not evidence that the analogy is valid; a candidate must be inspected with `read_case` before Echo can endorse it. Retrieval is also **not mandatory as the final answer source**: if all named cases are strained, Echo may construct a system analogy directly from the abstract causal pattern, while clearly distinguishing that construction from retrieved evidence.

### External search boundary

Echo does **not** expose a generic web-search or RAG stack. `search` has two explicit sources:

```text
source="core"       -> structural retrieval over curated CaseCards
source="wikipedia"  -> open-domain candidate discovery through MediaWiki Action API
```

The Wikipedia boundary is deliberately asymmetric:

```text
Wikipedia search result
        ↓
ExternalCandidate
(title / snippet / page id / URL)
        ↓
read_case
        ↓
bounded source evidence
        ↓
LLM decides whether the causal structure actually matches
```

A Wikipedia hit is **not** silently normalized into a CaseCard and is never treated as a validated analogy. The adapter owns network I/O and provenance; the LLM owns relational judgment. Echo treats the Core Atlas as a **seed set, not a closed world**. The deterministic coverage gate now marks Core coverage weak when the best structural score is below `0.48`, when less than roughly two-thirds of the supplied canonical mechanisms are preserved, or when an important open mechanism cannot be mapped into the finite Atlas vocabulary. Weak coverage requires at least one mechanism-oriented Wikipedia discovery attempt before a normal final answer. The model is instructed to search the abstract mechanism or a hypothesized target concept rather than paraphrasing the user's surface nouns. A failed Wikipedia request still counts as an attempt so network problems cannot trap the loop. Retrieval failure is not treated as evidence that no useful analogy exists: Echo can switch to a clearly labeled constructed analogy.

Wikisource, live-news search, embeddings, and a vector database are intentionally outside the current search boundary. They could expand coverage, but they are not necessary to preserve Echo's central distinction between candidate retrieval and structural judgment.

### Analogy quality gate

Echo intentionally separates **candidate generation** from **answer acceptance**. A famous or high-scoring case can still be rejected if the mapping needs facts that were never present in the user's situation. The final quality review applies four practical checks:

1. at least three meaningful relational or causal correspondences;
2. no central hidden motive, event, or constraint invented merely to make the analogy fit;
3. a shared mechanism that can be stated without relying on the same surface nouns;
4. one explicit place where the analogy stops transferring.

This matters for cases like a betrayal scenario that superficially resembles the Trojan Horse: if the source never says there was long-term disguise or planned infiltration, Echo should not invent those premises just to use a famous story. In that situation the reviewer can discard the retrieved draft and construct a cleaner cross-domain analogy instead.

Production execution also has a small search budget: one Core search, at most two Wikipedia searches, and at most three case inspections per user turn. This prevents repeated query-rewording from consuming the whole loop. The last normal iteration exposes **no tools** and forces synthesis from existing evidence; if a provider still returns no usable text, the runtime makes two additional bounded no-tool rescue attempts. Therefore a normal search loop no longer ends with a user-visible `ran out of 8 iterations` message.


### Analogy evaluation

The controlled evaluation is inspired by **Taylor Webb, Keith J. Holyoak, and Hongjing Lu (2023), _Emergent analogical reasoning in large language models_, Nature Human Behaviour 7, 1526–1541**, published 31 July 2023. The paper directly compared large language models and humans across several analogy tasks, including story analogies, and is especially relevant to Echo because it treats analogy as a problem of relational structure rather than simple surface resemblance. Paper: https://doi.org/10.1038/s41562-023-01659-w

Echo does not reuse that study as evidence of human-equivalent reasoning. It borrows the **experimental logic**: surface similarity and higher-order relational similarity should be separable. That leads to four controlled conditions:

| Condition | Surface similarity | Structural similarity |
|---|---:|---:|
| Near | high | high |
| Far | low | high |
| Surface trap | high | low |
| Unrelated | low | low |

The main stress test is **Far vs. Surface Trap**: a candidate from a different domain that preserves the critical causal relations should outrank a topically similar case whose mechanism is wrong. The benchmark also includes cross-domain retrieval and pairwise ranking checks, plus a simple lexical baseline so a good Echo score is not automatically interpreted as evidence of structural reasoning. Runtime correctness and analogy quality are evaluated separately; a tool loop can work perfectly while the abstraction itself is wrong.

See [`docs/EVALUATION_DESIGN.md`](docs/EVALUATION_DESIGN.md) for the controlled stimuli and metrics.

### Structured state before lossy summary

The active **Analogy Board** is stored separately from transcript text. It can retain the current situation, working frame, selected/rejected candidates, mapping notes, and where an analogy breaks. On a fresh decision it is injected before the session summary, so transcript compression can remove verbose old dialogue without erasing a decision that should survive follow-up.

Several lightweight guards keep this reasoning loop productive. An exact successful tool call cannot be immediately repeated; searches that return no unseen candidates are marked saturated; weak Core coverage requires external widening; retrieved claims require an inspected case; and per-turn tool budgets prevent cosmetic query rewrites. These guards constrain execution without hard-coding which candidate the model must choose.

## Session model

The state key is:

```text
(user_id, session_id)
```

not only `user_id`.

```text
user-a
  ├── window-1 -> messages, summary, analogy board
  └── window-2 -> independent state
```

SQLite persistence means closing and reopening the process does not erase the session. Reusing the same `(user_id, session_id)` restores the conversation and its structured analogy state; changing the session ID creates an independent window.

## Context management

Before each model decision, Echo reconstructs a bounded working context from:

```text
active analogy board
session summary
recent raw user / assistant / compact tool messages
latest rich tool result
```

When the stored conversation exceeds the basic character budget, older messages are summarized while a recent raw window is retained. The summary prompt preserves active goals, confirmed facts, completed actions, important tool findings, unresolved work, and decisions while discarding repeated wording and obsolete tentative discussion. Private chain-of-thought is never persisted.

Large external evidence follows a similar rule: the immediate next decision can see a richer bounded source extract, while durable history keeps a smaller provenance/evidence preview. The current implementation uses transparent character budgets rather than tokenizer-specific accounting; the policy is intentionally simple and testable.

## Memory: recall timing and placement

Echo's implemented memory is session-local: SQLite history, a compact session summary, and the structured Analogy Board. Recall happens **before each model decision** when the runtime reloads the session and reconstructs the active context. The transcript remembers what was said; the board remembers what was decided.

A larger long-term memory system should not inject every stored memory on every turn. A natural extension would add a recall gate:

```text
request
  ↓
does this answer need episodic / personal history?
  ↓ yes
retrieve candidate memories
  ↓
rerank for relevance, confidence, freshness, expected answer gain
  ↓
small MEMORY section in active context
```

This keeps irrelevant history from consuming the context window or biasing a current answer. The fuller design is discussed in `ARCHITECTURE_ANSWERS.md`.

## Reasoning / output parsing

The Responses boundary parses three kinds of model output:

- typed function calls;
- final assistant text;
- an optional provider-exposed reasoning summary, if present.

Echo does not request, expose, or persist hidden chain-of-thought. Durable state is built from user statements, actions, observations, structured board decisions, and final answers rather than private internal reasoning.

## Error handling

Handled failure modes include unknown tools, JSON Schema validation failures, tool-specific validation errors, arithmetic errors, unexpected internal tool exceptions, empty model decisions, repeated/no-progress actions, exhausted search/read budgets, provider/network failures at the CLI boundary, and bounded termination through `max_iterations=8`. The final normal iteration is reserved for no-tool synthesis, followed by bounded rescue synthesis only if the provider returned no usable text. A recoverable tool failure becomes an observation that the model can reason over instead of crashing the session; provider failures do not erase already persisted session state.

## Trace / observability

`trace.jsonl` is intentionally compact. It records actual tool executions and the few runtime guards that materially change control flow rather than logging every internal model request. A normal tool event looks like:

```json
{
  "event": "tool",
  "session_id": "window-1",
  "call_id": "fc_...",
  "tool": "search",
  "arguments": {"source": "core"},
  "success": true,
  "latency_ms": 5.13,
  "error": null
}
```

Guard events cover duplicate actions, search saturation, weak-Core widening, provenance enforcement, and tool-budget exhaustion. Separate `quality_review` and `rescue_synthesis` events make the final acceptance path visible without logging private reasoning. `/trace` surfaces the recent events for debugging and demonstration without turning observability into a second subsystem.

## Tests

Normal deterministic tests make no API calls:

```bash
python -m pytest -q -m 'not live'
```

Current local result:

```text
49 passed, 1 skipped, 2 deselected
```

The one skipped test in this command is the opt-in external Wikipedia smoke test. Two live Groq tests are deselected: one verifies real function calling and tool-result continuation, and one verifies Echo's product routing by giving the model a bare concrete situation and requiring successful `search` and `read_case` activity.

Run the real API tests with:

```bash
RUN_LIVE_TESTS=1 GROQ_API_KEY='...' python -m pytest -q tests/test_live_api.py -m live -s
```

Run the read-only Wikipedia smoke test separately with:

```bash
RUN_EXTERNAL_TESTS=1 python -m pytest -q tests/test_wikipedia_live.py -m external -s
```

Both paths are opt-in so ordinary testing stays deterministic and does not depend on API credits or external network availability.

## Repository

```text
src/agent_harness/
  runtime.py       agent loop and progress guards
  llm.py           Responses-compatible model boundary
  context.py       summary + recent-window context
  store.py         SQLite sessions and Analogy Board
  tooling.py       tool interface and registry
  cases.py         Core Atlas
  external.py      Wikipedia adapter
  tools/           calculator / search / read_case / analogy_board

tests/             runtime, behavior, live, and external smoke tests
eval/              controlled analogy stimuli
docs/              evaluation design
```

Echo is intentionally modest in its claims: it does not claim exhaustive historical search or human-level analogical reasoning. Its stronger product goal is narrower and testable: retrieve broadly enough to escape a finite case library, reject assumption-heavy matches, construct a fresh system analogy when retrieval is worse than synthesis, and always terminate a normal turn with a substantive analogy rather than a loop-limit message.
