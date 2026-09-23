# Jev Tool Router — Phase 1 Design

Date: 2026-09-17
Status: approved in chat (Henry), pending spec review
Scope: standalone router tool + eval harness. No agent harness integration in this phase.

## Idea

An agent harness exposes ONE tool: a router. The agent describes its current
step; the router asks Jev (TypeSafe's System One model) to classify the step
and returns which tool the agent should call next. Tool selection moves out of
the agent's sampled text into a fast, calibrated classifier: the catalog can be
large without bloating agent context, and routing decisions come with
probability distributions the harness can act on.

Phase 1 builds and evaluates the router in isolation — no agent attached.
Phase 2 (separate design) scales the catalog to 50–150 synthetic tools and
measures degradation. Phase 3 attaches the router to a real agent loop.

## Routing call (approach B: Choice + gates, one API call)

One `POST https://api.typesafe.ai/v1/systemone` call per hop, three questions
evaluated in parallel against the same state:

| id | type | question |
|---|---|---|
| `tool` | choice | Which tool should the agent call for this step? Criteria = the 15-tool catalog (name -> one-line description). |
| `needs_tool` | noul | Does this step need a tool call at all, or can the agent answer directly from what it already knows? |
| `step_clarity` | noul | Is the step description specific enough to route to a single tool? |

State sent to Jev:

```json
{
  "task": "<the broader task the agent is working on>",
  "current_step": "<the agent's description of what it needs to do now>",
  "recent_steps": ["<up to 5 prior step descriptions, oldest first>"]
}
```

`recent_steps` is optional and empty in most eval cases; a handful of eval
cases exercise it to confirm history helps rather than distracts.

## Router API

```python
route(task: str, step: str, history: list[str] = []) -> RouteResult

@dataclass
class RouteResult:
    tool: str                      # top-1 choice
    probabilities: dict[str, float]  # full distribution over the catalog
    confidence: float              # Jev's choice confidence
    needs_tool: float              # noul 0-1
    step_clarity: float            # noul 0-1
    usage: dict[str, int]          # input/output tokens
```

The full distribution survives so a future harness can do top-k fallback when
confidence is low. The router itself never truncates to top-1 internally.

Auth: a Vercel AI Gateway key. Model: `jev-latest`. Errors 429/529 retry with
exponential backoff (3 attempts); other HTTP errors raise.

## Catalog (phase 1)

~15 real coding-agent tools mirroring a Claude Code-style harness, defined as
`name -> one-line description` in `catalog.py`:

Read, Write, Edit, Bash, Grep, Glob, WebFetch, WebSearch, Agent (subagent
fan-out), TodoWrite, NotebookEdit, AskUserQuestion, KillShell/TaskStop,
ListDir, LSP (go-to-definition/references). Exact set finalized in
implementation; the property that matters is realistic overlap (Read vs Grep
vs Glob; Bash vs dedicated tools) so routing is non-trivial.

## Eval set

`evalset.jsonl`, ~60 hand-labeled cases. Each line:

```json
{"id": "...", "task": "...", "step": "...", "history": [],
 "expected_tool": "Grep", "expected_needs_tool": true, "vague": false}
```

Composition:
- ~46 clear steps across the catalog (every tool covered by >= 2 cases),
  including deliberately confusable pairs (Read vs Grep, Bash vs Glob,
  Edit vs Write, WebFetch vs WebSearch).
- ~8 "no tool" cases (`expected_needs_tool: false`) — the agent should answer
  directly (summarize what it found, explain a concept it knows).
- ~6 vague steps (`vague: true`, e.g. "handle the file stuff") — no expected
  tool; success = low `step_clarity`.

## Metrics (`run_eval.py`)

- Top-1 accuracy on clear steps; top-3 accuracy as secondary.
- Confusion pairs: expected -> chosen for every miss.
- Calibration split: mean choice confidence on correct vs incorrect routes.
- Gate quality: `needs_tool` on no-tool cases vs tool cases;
  `step_clarity` on vague vs clear cases (report AUC-style separation:
  do the distributions overlap?).
- Cost: total tokens and $/1k routes if pricing is documented.

Output: plain-text report to stdout + `results/<timestamp>.json`.

## Success criteria (phase 1 "good enough" bar)

1. >= 90% top-1 accuracy on clear steps.
2. Mean confidence on wrong routes measurably below mean confidence on
   correct routes (the property phase 2/3 depend on).
3. Vague steps separate from clear steps on `step_clarity`.

If (2) fails, the probabilities are decoration and the harness idea needs a
rethink before phase 2.

## Repo layout

```
jev/
  router.py       # route() + Jev HTTP client + retry
  catalog.py      # phase-1 tool catalog
  evalset.jsonl   # hand-labeled cases
  run_eval.py     # runs evalset through route(), prints report
  results/        # timestamped eval outputs (gitignored)
  docs/superpowers/specs/  # this spec
```

Python 3.11+, stdlib + `requests` only. No package scaffolding, no framework —
this is an experiment, but the router module is written to be lifted into a
harness later unchanged.

## Testing

The eval harness IS the test for routing quality. Unit tests are limited to
what does not spend API tokens: request-body construction, response parsing,
retry behavior (mocked), evalset schema validation. `pytest`, single test
file.

## Out of scope (phase 1)

- Agent harness integration, tool execution, top-k fallback logic.
- Hierarchical/category routing (phase 2 lever).
- Baseline comparison vs native LLM tool selection (candidate for phase 2/3).
- GitHub repo / CI — local git only until Henry says otherwise.
