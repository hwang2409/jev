# Jev Tool Router — Phase 3 Design (live agent harness)

Date: 2026-09-17
Status: approved (Henry: "Cool, let's do it. Go!", 2026-09-17)
Depends on: phase 2 (`2026-09-17-jev-router-phase2-design.md`, complete at `3a63c7c`)

## Question phase 3 answers

Does the one-tool harness work end-to-end with a real agent, and what does it
cost/save vs a conventional harness? Two harnesses, same tasks, same model:

- **jev harness**: the agent sees ONE persistent tool, `route`. It describes
  its next step; the harness asks Jev (phase-2 `router.route`, catalog =
  `CATALOG_120`) and injects the chosen tool's schema into the NEXT API
  request's tools list (top-3 schemas when choice confidence < 0.8 — the rule
  phase 2 validated). After that tool is called, injected schemas are removed.
- **baseline harness**: all 120 tool schemas in the tools list every turn,
  native selection, no route tool.

Both run simulated tool execution — no real side effects. Measurement:
completion, correct tool sequence, token cost (headline), route interventions.

## Anthropic API contract (verified against the claude-api skill, 2026-09-17)

Workers MUST follow these shapes exactly; do not improvise from memory:

- SDK: `pip install anthropic` (1.x). Client: `anthropic.Anthropic()` (reads
  `ANTHROPIC_API_KEY` from env). All tests mock the client; zero network in
  tests.
- Model: `"claude-opus-5"`. Thinking is ON by default (omit the `thinking`
  parameter entirely; do NOT send `budget_tokens` — it 400s). Set
  `output_config={"effort": "medium"}` on every request (identical for both
  harnesses; consistency matters more than the value).
- Requests go through `client.beta.messages.create(...)` with
  `betas=["server-side-fallback-2026-07-01"]` and `fallbacks="default"`
  (refusal fallback, recommended default for opus-5). `max_tokens=8000`.
- Manual agent loop (the Tool Runner cannot mutate tools per turn):
  - while `response.stop_reason == "tool_use"`: collect ALL `tool_use` blocks
    from `response.content`; execute each; append
    `{"role": "assistant", "content": response.content}` (full block list —
    preserves thinking blocks for same-model replay) then ONE
    `{"role": "user", "content": [<tool_result block per tool_use>]}`.
  - `tool_result` block: `{"type": "tool_result", "tool_use_id": <id>,
    "content": <string>}`; on execution error add `"is_error": True`.
  - Stop on `end_turn`. Guard `stop_reason == "refusal"` (read
    `response.stop_details` only then) — record and end the scenario as
    failed-refusal. Hard cap 14 API turns per scenario; exceeding = failure.
- Tool definitions: `{"name", "description", "input_schema"}` with
  `input_schema` = `{"type": "object", "properties": {"details": {"type":
  "string", "description": "All arguments for this action, in plain words"}},
  "required": ["details"], "additionalProperties": False}` and `"strict": True`
  (top-level field on the tool). One uniform schema for every catalog tool —
  execution is simulated. NOTE for interpretation: uniform 1-field schemas make
  the baseline's context artificially SMALL; real catalogs have larger schemas,
  so measured savings are a lower bound.
- Track `response.usage.input_tokens` and `output_tokens` per turn (sum
  cache-read tokens separately if present: `usage.cache_read_input_tokens`).

## Components

1. `sim_exec.py` — `execute(tool_name: str, details: str, overrides: dict) ->
   str`. Returns the scenario's canned result when `tool_name` is in
   `overrides`, else a generic deterministic acknowledgement string built from
   the tool name and details. Never raises.
2. `harness.py` — `run_scenario(scenario: dict, mode: "jev" | "baseline",
   client=None, route_fn=None) -> dict` implementing both loops per the API
   contract. jev mode: tools = [`route`] + currently injected; `route` input
   schema = `{"step": string}`; executing `route` calls
   `route_fn(scenario_task, step, catalog=CATALOG_120)` (default
   `router.route`), returns a tool_result string naming the chosen tool
   (and top-3 with probabilities when confidence < 0.8) and injects the
   corresponding schemas for the next request. Injected schemas clear after
   any non-route tool call. Baseline mode: all 120 schemas, no route.
   System prompts: shared task-agent preamble; jev adds the one-tool protocol
   ("describe your next step to route, then call the tool it returns").
   Returns per-scenario record: mode, turns, tool_calls (ordered names),
   route_calls, expansions (top-3 injections), input_tokens, output_tokens,
   cache_read_tokens, completed (end_turn with non-empty final text),
   refusal (bool), final_text.
3. `scenarios.jsonl` — 10 scenarios. Schema per line: `id`, `task` (the user
   request), `expected_tools` (ordered list, 3-5 entries from `CATALOG_120`),
   `results` (map tool_name -> canned result string that feeds the next step),
   `answer_keys` (list of 1-3 strings the final answer must contain,
   case-insensitive — drawn from canned results). Scenarios span >= 6 domains;
   at least 3 scenarios cross domains mid-sequence; steps must be inferable
   from the task + prior results (no tool names in task text — leak rule from
   phase 2 applies to tasks vs descriptions).
4. `run_phase3.py` — runs all scenarios in both modes (jev first, then
   baseline, serially), prints a comparison table (per scenario and totals:
   completed, sequence_match, tokens in/out, turns, route stats) and saves
   `results/phase3-<ts>.json` with full per-turn logs. Sequence match =
   `expected_tools` is exactly the ordered list of non-route tool calls
   (extra or reordered calls = mismatch, recorded with the actual list).
   `--mode jev|baseline|both` flag.
5. `tests/test_phase3.py` — stubbed client (scripted responses) + stubbed
   route_fn: jev loop injects/clears schemas correctly, top-3 expansion on
   low confidence, tool_result batching (all results in one user message),
   refusal guard, turn cap, sequence matching, scenario schema invariants
   (counts, tools exist in catalog, no tool names in task text, answer_keys
   nonempty), token accounting.

## Interpretation bar

Measurement, not pass/fail. Extract: (1) completion + sequence-match rates per
harness; (2) input tokens per completed scenario (the context-cost headline);
(3) route overhead (Jev calls + expansions) vs baseline schema overhead;
(4) failure modes (wrong tool, spun out, refusal). If the jev harness
completes <= half of what baseline completes, the harness protocol (not the
router) is the suspect — one protocol-prompt revision round is in scope.

## Out of scope

Real tool execution, streaming, prompt caching tuning (record cache reads,
do not optimize), multi-model comparison, latency benchmarking.
