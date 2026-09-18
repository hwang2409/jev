# jev-zeta: router v2 design — invisible auto-routing, cache-stable surface

Date: 2026-09-18
Status: approved (Henry: "let's do it — spec this next lane once the compaction work lands")
Repo: `~/me/fun/jev-zeta` (local-only fork). Depends on: router v1 (complete), live eval findings (`evals/RESULTS.md`).

## Why v2

The live eval showed v1's three taxes: per-turn tools mutation defeats prompt
caching (stock reused 98.5k tokens at 0.1x; v1 paid fresh input), route
round-trips consume API turns and the turn budget, and routing on the agent's
one-line self-description misroutes when phrasing is off (bash for an edit).
V2 removes all three by construction.

## Design

1. **Harness-side auto-routing (no agent-visible route tool).** Before each
   provider request, the loop calls Jev (`providers/jev.py`, new
   `auto_route(...)` reusing the client): state = `{"task": <latest user
   objective, 500 chars>, "last_assistant": <excerpt 300 chars>,
   "last_results": [<tool, excerpt 200 chars> x up to 2]}`; questions =
   `tool` Choice over the live catalog + `needs_tool` Noul. The result gates
   the NEXT turn only. Zero extra API turns; routing input is real
   conversation state, not agent phrasing.
2. **Static tool surface.** In v2 the tools param is permanently
   `[invoke]` — never mutates, so the prompt-cache prefix behaves exactly like
   stock zeta. `invoke` schema: `{"tool": string, "args": object}` (args
   free-form; strict off). The routed tool's full schema (name, description,
   parameters rendered as text) is delivered per turn as an extra text block
   appended to the SAME user message that carries the tool results (or the
   initial user message on turn 1) — appended content rides the growing
   message tail, so history prefix-caches normally.
3. **Gating semantics.** Advertise the top choice; top-3 schemas when Choice
   confidence < 0.8. `needs_tool` < 0.35 -> advertise none ("answer directly
   this turn"). `invoke` naming an un-routed tool -> governance error result
   with `error_kind: "unrouted_tool"` (reuse v1's marker) telling the agent to
   state what it needs in text — the next auto-route reads that text, so
   misroutes self-correct in one turn. Executing invoke dispatches to the
   named tool through the normal registry path (approvals unchanged).
4. **State lifetime.** Same hardened rules as v1: routed set is per-turn,
   reset at user-turn boundaries; auto-route failure (Jev error/missing key)
   -> fail-open to the full stock toolset for that turn only, with the v1
   observability event. One Jev call per provider turn, no retap on batch.
5. **Config.** Setting + flag `--router-style tool|auto` (v1 = `tool`,
   v2 = `auto`); default `auto` in the fork. `--no-router` still restores
   stock entirely. Child agents inherit style. Telemetry: same
   `service: "jev"` usage surfacing and routing-decision event data as v1.
6. **Eval.** Extend `evals/run_evals.py` with mode `auto` (CLI:
   `--router-style auto`; `router` mode maps to `--router-style tool`).
   Three-way comparison table. Rerun is the orchestrator's live step.

## Testing

Python only, no network, no cargo/GUI, mocked Jev + backend. Cover: auto-route
request shape (state excerpts, truncation); static tools param never changes
across turns (assert identity across a 3-turn scripted run); schema-text
delivery block placement; top-3 and needs_tool gating; invoke dispatch to
registry (approvals still consulted); unrouted invoke -> marker error;
self-correction flow (text request -> next-turn route includes it via
last_assistant); fail-open one-turn semantics; user-turn reset; style flag
plumbing incl. children; v1 behavior unchanged under `tool` style.
Targeted test files only.

## Out of scope

Removing v1 (keep for A/B), MCP special-casing, provider-side strict schemas
for invoke, GUI.
