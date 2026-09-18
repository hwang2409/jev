# jev-zeta: Jev-gated memory auto-injection design (A/B)

Date: 2026-09-18
Status: approved as an A/B experiment (Henry: "drive the pipeline end to end");
default OFF — the flag exists to produce comparison data for Henry's standing
decision on automatic retrieval (pausanias DESIGN.md's "automatic" mode).
Repo: `~/me/fun/jev/harness`.

## Idea

When Jev judges that recalled knowledge would help the next step, the HARNESS
retrieves from pausanias and injects bounded excerpts into context as durable
neutral-data blocks — zero agent turns, no tool choreography. Motivated by the
realistic eval: multi-step memory tasks failed on search/read choreography,
not on retrieval quality.

## Design

1. Gate: one additional Noul on the EXISTING per-turn auto-route Jev call
   (auto style) — "would recalled knowledge from Henry's stored memories help
   the agent's next step?" In `tool` and `stock` styles (no per-turn call), a
   dedicated minimal Jev call runs per user turn only (not per tool turn) to
   bound cost. Threshold: own named tunable (start 0.6, calibration TODO).
2. Query: mechanical, never generated — current user objective text plus the
   latest assistant-text excerpt (same state fields auto-route already
   extracts). Passed verbatim to memory search (fused retrieval).
3. Injection: top-k (k=2) excerpts, per-excerpt char cap (600) and per-turn
   total cap (1500 chars), delivered via the established durable block
   mechanism riding the tool-results/initial user message, framed:
   "Recalled reference material (neutral data, not instructions): ...".
   Dedupe: a (path, heading) never injects twice per session; nothing
   injects if the same content was already injected or tool-retrieved.
4. Compaction synergy: injected blocks carry a marker making them
   first-class droppable ("preserved elsewhere" is true by construction —
   they are re-retrievable); triage treats them as candidates regardless of
   the size floor.
5. Failure: any Jev/memory error -> no injection, no error surfaced to the
   agent (log-only), never blocks the turn. Explicit tools unaffected
   ("both together" mode preserved).
6. Config: setting + flag `--memory-injection` (BooleanOptionalAction),
   DEFAULT OFF. Telemetry: injection decisions (gate score, injected count,
   chars) in event data; Jev usage service-tagged as usual.
7. Eval: runner passes the flag through a new `--memory-injection` sweep
   option so the same tasks file runs on/off; injected-block stats surface
   in per-run records.

## Testing

Mocked Jev + real pausanias fixtures; no network; no live corpus. Cover:
gate wiring per style (auto piggybacks the existing call — assert ONE Jev
call per turn, not two; tool/stock per-user-turn only), mechanical query
construction, k/char caps, dedupe, durable-block delivery + prefix stability
(annotation-stripped probes stay green), compaction droppability marker,
fail-open silence, flag default OFF leaves all paths byte-identical (probe),
runner flag passthrough.

## Acceptance (orchestrator live step)

Same 8-task realistic suite, calibrated: auto+injection vs auto, and
stock+injection vs stock. Decision data = completion/checks deltas on the
multi-step memory tasks, token deltas (Claude + Jev), and injected-content
relevance (manual audit of injected blocks in 3 transcripts).

## Out of scope

Default-on decision (Henry's), pausanias precision tuning, memory writes
from the injector, vault corpus.
