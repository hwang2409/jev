# jev-zeta: router mode design

Date: 2026-09-17
Status: approved (Henry: fork zeta, local-only repo, route tool replaces the standard toolset)
Repo: `~/me/fun/jev-zeta` — local fork of zeta. NO remotes (verified), never pushed, never merged back into `~/me/fun/zeta`. Highly experimental.

## Idea

Zeta's agent normally sees ~10 tool schemas every turn. In router mode the
model sees ONE tool: `route`. It describes its next step; the handler asks
Jev (TypeSafe System One, `jev-latest`) to classify the step over the live
tool catalog, and the loop advertises the routed tool's schema on the next
turn (top-3 when choice confidence < 0.8 — the rule validated in the jev
phase-2 experiment, `~/me/fun/jev/RESULTS.md`). Tool execution, approvals,
TUI, and persistence are untouched.

## Components

1. `src/zeta/providers/jev.py` — async Jev client.
   - `async route_step(step: str, catalog: dict[str, str], history: list[str] = []) -> RouteResult`
   - POST `https://api.typesafe.ai/v1/systemone`, model `"jev-latest"`, auth
     `Authorization: Bearer $JEV_API_KEY`.
   - Body: `state={"current_step": step, "recent_steps": history[-5:]}`,
     `questions`: `tool` (choice, criteria = catalog),
     `needs_tool` (noul), `step_clarity` (noul) — same shapes as
     `~/me/fun/jev/router.py` (read it; port, don't reinvent).
   - `RouteResult`: tool, probabilities, confidence, needs_tool, step_clarity,
     usage. Retry 429/529, 3 attempts, exponential backoff; other HTTP errors
     raise `JevRouterError` (zeta-style exception in the module).
   - Uses `httpx.AsyncClient` (zeta's HTTP stack). No new dependencies.
2. `src/zeta/tools/route.py` — registered tool (follow the registry's
   auto-discovery + registration pattern of an existing simple tool, e.g.
   `todo.py`). Schema: `{"step": string}` required; description tells the
   model to describe its next concrete action.
   - Handler: build catalog from the registry's current schemas (name ->
     description first line, <= 150 chars; exclude `route` itself), call
     `route_step`, return a text result: routed tool name + confidence, or
     top-3 with probabilities when confidence < 0.8, plus a low-clarity nudge
     ("restate the step more concretely") when step_clarity < 0.3.
   - Reports the routed tool set to the loop (mechanism: whatever zeta's
     tool-to-loop contract supports cleanly — `ToolExecutionContext` binding
     or a loop-installed callback; implementer's choice, reviewer judges).
   - If `JEV_API_KEY` is unset or Jev errors after retries: return an error
     result naming the failure; the loop then advertises the FULL toolset for
     the next turn (fail-open — an experiment session must not wedge).
3. `src/zeta/loop.py` — router mode.
   - `AgentLoop` state: `router_mode: bool` (default ON), `_routed_tools:
     list[str]` (default empty).
   - `_active_tool_schemas()`: when router_mode, return the `route` schema
     plus schemas whose names are in `_routed_tools`, then apply the existing
     plan-mode filter on top (composition order: router first, plan filter
     second; `route` itself is always allowed in plan mode).
   - Semantics (mirror the jev phase-3 harness, which is reviewed and
     tested): executing `route` REPLACES `_routed_tools` with the new set;
     executing any non-route tool CLEARS `_routed_tools`; a mixed batch
     (route + tool in one assistant turn) keeps the newly routed set.
   - Unrouted calls: in router mode, a tool_call naming a registered tool
     that was not advertised this turn gets an error result ("not available
     this turn — describe your step to route first") through the registry's
     existing error-governance path; log/count as `unrouted_attempts`.
4. CLI/settings: `--no-router` flag (cli.py) and matching setting to disable
   router mode (restores stock zeta behavior). Default is ON — router mode is
   this fork's identity.

## Testing

Python only. NO cargo/GUI builds or tests, ever (zeta rule). NO network in
tests: mock the Jev HTTP layer (monkeypatch/respx) and any backend.
New tests: `tests/test_jev_provider.py` (request shape, retry, error ->
JevRouterError), `tests/test_router_mode.py` (advertisement in router mode,
replace/clear/mixed-batch semantics, unrouted rejection, top-3 expansion,
fail-open on Jev failure, catalog excludes route, `--no-router` restores the
full set, plan-mode composition). Workers run TARGETED tests only (their new
files + directly touched modules); the orchestrator runs the wider python
suite once at gate.

Setup: `uv sync --frozen` creates the fork venv; run tests via
`uv run --frozen pytest <paths>`.

## Live smoke (orchestrator step, after review)

Headless zeta session on the Claude subscription OAuth (token in
`~/.zeta/anthropic-oauth.json`) with `JEV_API_KEY` set: a small real
multi-tool task; verify every tool selection went through route (transcript
shows route calls; unrouted_attempts == 0) and the task completes.

## Out of scope

GUI changes, MCP tools in the catalog (keep whatever the registry reports,
but no special handling), baseline-vs-router token benchmarking (later),
upstream zeta merges (never).

## Status: live smoke PASSED (2026-09-18)

Headless run at head `b8b9bbb0` (claude provider, subscription OAuth, yolo,
router default-on): task "read notes.txt, count jev mentions, write count".
Tool order `route -> read -> route -> write`; first route hit confidence
0.79 and the top-3 expansion fired (read 0.79 / bash 0.17 / exec 0.04) with
the agent choosing correctly; second route `write` at 1.00; zero unrouted
attempts; output correct. Build history: 4 implement lanes (JEV-15..18),
3 review rounds, findings 6 -> 2 -> 0.
