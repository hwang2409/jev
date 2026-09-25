# Offline browser eval baseline — 2026-09-25

The committed runner is deterministic and uses `FakeBrowserAdapter` plus a
local mocked Jev transport. It opens no network connection, provider key, or
Playwright binary.

Corpus: 60 tasks covering 10, 40, 120, 500, and 2,000 elements; static,
moderate, and full churn; clear targets, repeated labels, search triage, and
forms. Both arms use the same fixtures, prompts, timeouts, budgets, and safety
policy.

| Metric | Routed | Stock large-tool |
|---|---:|---:|
| Top-1 accuracy | 0.7500 | 0.7500 |
| Top-3 coverage | 0.8000 | 1.0000 |
| Page-state accuracy | 1.0000 | 1.0000 |
| Task success rate | 0.7333 | 0.7333 |
| Risky false-approval rate | 0.0000 | 0.0000 |
| Jev tokens | 22,623 | 349,710 |
| Jev tokens per successful step | 514.1591 | 7,947.9545 |
| Jev cost | 0.024489 | 0.351720 |
| Model input tokens | 20,757 | 347,700 |
| Cache reads | 0 | 0 |
| Provider turns | 123 | 135 |
| Stale recovery rate | 0.7000 | 0.7000 |
| Time per successful step (seconds) | 0.013916 | 0.013943 |

Failure categories remain separate. Routed: 12 pre-filter misses and 3 Jev
selection misses. Stock: 15 Jev selection misses. Adapter and page failures
are both zero in this clean corpus and have dedicated regression coverage.
Threshold version and confidence calibration are recorded per primitive.

The crossover matrix reports 12 routing wins in 15 cells. A win requires task
success and safety parity plus a cost or time gain. Token reduction alone does
not count as a win.

Command:

`uv run --frozen python -m evals.browser_eval --tasks evals/browser_tasks.jsonl --output /tmp/browser-eval-baseline.json`

# Router vs stock — live eval results (2026-09-18)

Head `ba0564d7`, 6 tasks x 2 modes, claude-sonnet-4-6 via subscription OAuth,
jev-1.13.0 routing. Full records: `results/20260918T142954577340Z.json`.

| Metric | Router | Stock |
|---|---|---|
| Tasks passing all checks | 4/6 | 5/6 |
| Raw tokens (Claude in/out + Jev) | 58.2k/6.4k + 18.2k Jev | 21.9k/2.9k |
| Cache reads | 11.6k | 98.5k |
| Est. cost (sonnet pricing, cache-adjusted) | ~$0.27 + Jev | ~$0.14 |
| Wall clock (total) | 89s | 68s |

## Findings

1. Dynamic per-turn tool injection DEFEATS prompt caching: the tools list
   changes every turn, invalidating the prefix. Stock's static 10-tool
   prompt cached 98.5k tokens at 0.1x; the router paid fresh input on
   almost every turn. At this catalog size the router is ~2x the dollar
   cost despite 38 percent fewer raw combined tokens.
2. Route round-trips consume the turn budget: 3 router runs finished their
   work (checks green) but ran out of turns before a final message.
3. One genuine routing-quality miss: in-place-edit routed to bash (the
   agent phrased its step shell-style) and botched the edit; stock used
   read+edit cleanly.
4. chain-and-verify failed strict checks in BOTH modes (formatting drift) —
   an eval-strictness issue, not a router issue. Manual audit of both event
   streams: no checker gaming; the router version executed the full chain
   routed (read manifest -> reads -> writes) but skipped the final verify
   re-read.
5. Router mechanics were flawless: 0 unrouted attempts, 0 router errors,
   12 runs clean.

## Implications for the one-tool harness thesis

The thesis holds only where catalog schema bloat exceeds cache savings:
big catalogs (phase 2: 120+ tools), cache-hostile contexts, or providers
without prefix caching. At ~10 cached tools, stock wins on cost and turn
efficiency. Design leads if pursued further: cache-stable injection
(append-only tool exposure; or deliver routed schemas via user-message
content instead of the tools param) and a turn-budget-aware route protocol.

# Three-way eval: auto (v2) vs router (v1) vs stock — 2026-09-18

Head `4866bcfe`. Same 6 tasks, same day. Full records: `results/threeway-*.json`.

| Metric | auto (v2) | router (v1) | stock |
|---|---|---|---|
| Completed | 6/6 | 2/6 | 5/6 |
| Checks passed | 5/6 | 5/6 | 5/6 |
| Claude tokens (in+out) | 50.6k | 59.0k | 128.6k |
| Jev tokens | 24.8k | 19.5k | 0 |
| Cache reads | 0 | 8.7k | 102.4k |
| Est. cost (cache-adjusted) | ~$0.21 + Jev | ~$0.27 + Jev | ~$0.15 |

## Findings

1. V2 delivered its behavioral goals: quality parity with stock (6/6
   completed — the turn tax is gone; v1 completed 2/6), zero unrouted
   attempts, zero router errors, invisible routing.
2. V2 did NOT deliver cache parity, for a newly understood reason: auto's
   static `[invoke]` surface shrinks the cacheable static prefix below
   Anthropic's ~1024-token minimum ("shorter prefixes silently won't
   cache") — cache_creation is 0 on every auto call. Stock's first call
   caches 3,940 tokens of system+schemas and re-reads them every call.
   The fat tool schemas v1/v2 remove are exactly what makes stock's
   prefix cacheable at this scale.
3. Zeta's marker policy ("cache only completed history before the active
   user turn") means single-user-turn agentic tasks never cache history
   in ANY mode; stock's wins are purely the static prefix. Headroom for
   all modes: marking completed intra-turn messages.
4. Standing conclusion sharpened: at ~10 tools, prompt caching makes the
   static catalog nearly free (~$0.03 per task-suite of cache reads), so
   routing cannot win on cost. Routing's value cases remain: quality
   parity now proven, huge/churning catalogs (MCP mounts, cache-TTL
   expiry in slow loops), cache-less providers, and policy/telemetry.

# Cache arc: three-way rerun with in-turn history caching — 2026-09-18

Head `78f335d8` (marker advanced into the active user turn + structured
criteria from the docs refactor). Records: `results/cachearc-*.json`.

| Metric | auto (v2) | router (v1) | stock |
|---|---|---|---|
| Completed | 6/6 | 2/6 | 5/6 |
| All-checks-passed tasks | 4/6 | 4/6 | 5/6 |
| Claude fresh in / out / cache | 12.6k / 4.1k / 13.0k | 8.9k / 3.7k / 23.8k | 3.1k / 2.9k / 112.0k |
| Est. Claude cost | $0.103 | $0.090 | $0.087 |
| Jev tokens | 43.7k | 34.1k | 0 |

## Findings

1. ACCEPTANCE PASSED: cache reads now nonzero in every mode. Auto's
   Claude-side cost halved ($0.21 -> $0.103) and the auto-vs-stock gap
   nearly closed ($0.103 vs $0.087; it was $0.21 vs $0.15).
2. In-turn caching improved STOCK dramatically too ($0.149 -> $0.087) —
   the change is a general zeta win, not a router accommodation. Strong
   upstream-to-zeta candidate (Henry's call; fork never merges itself).
3. Jev token usage rose ~75% (structured criteria carry per-call weight:
   what/not_for/examples across 17 tools ≈ +2-4k tokens per routing
   call). Tunable: trim examples, or cache criteria server-side if
   TypeSafe ever supports it. With Jev priced meaningfully below
   sonnet, auto remains cost-competitive; exact parity depends on Jev
   pricing.
4. chain-and-verify failed strict checks in ALL three modes (as in prior
   runs) — task strictness, not a mode regression; other single-check
   misses differ per mode and look like run variance on exact-match
   checks. Completion quality unchanged (auto 6/6).

## Arc verdict

The remaining cost story: at 10 tools, auto is now within ~18 percent of
stock on Claude cost with quality parity, invisible routing, and the
policy/telemetry surface. The structural blockers (turn tax, cache tax)
are both resolved; what remains is Jev's own call cost, which scales
with catalog size and criteria richness.

# Realistic-task eval: memory + calendar surfaces — 2026-09-18

Head `2a48b44`, 8 tasks x 3 modes, isolated corpora + fake calendar.
Records: `results/realistic-*.json`.

## Raw scoreboard

| Mode | Completed | All-checks | Claude tokens | Jev tokens | Cache reads |
|---|---|---|---|---|---|
| auto (v2) | 6/8 | 3/8 | 102.3k | 114.1k | 71.4k |
| router (v1) | 7/8 | 5/8 | 52.0k | 57.3k | 32.8k |
| stock | 6/8 | 5/8 | 209.0k | 0 | 197.7k |

## Task-calibration defects found (mode-independent failures)

1. store-then-recall: check hardcodes `memory/launch-review.md` (isolated
   corpus lives elsewhere) and the sequence demands the literal query term
   "Tuesday". ALL modes actually stored AND recalled correctly (recall.txt
   passed everywhere).
2. free-slot-reasoning: all three modes computed the CORRECT slot; the
   normalized_equals format string failed them (auto wrote
   "2026-09-19 10:00-11:00").
3. create-then-verify (auto only): calendar "Work" vs required "work" —
   case-sensitive equality; event otherwise exact.

## Adjusted (model-real) picture

- Memory single-shot flows (store, recall, seeded recall) work in every
  mode. Fused retrieval answered paraphrased questions.
- Cross-surface (calendar -> memory) passed in ALL modes — the flagship
  realistic flow works everywhere.
- Multi-step memory work is the real weakness: two-note synthesis (auto
  spiraled into bash; router missed one content check; stock passed) and
  append-then-latest (hard for everyone; router did the work but hit the
  turn cap; auto and stock genuinely failed).
- Routing comparison on realistic tasks: v1 router aged surprisingly
  well (best completion, checks parity with stock, HALF stock's raw
  Claude tokens). Auto's per-turn Jev calls ballooned on long tasks
  (114k Jev tokens — structured criteria paid every turn) and it showed
  the weakest multi-step discipline (bash spiral, calendar-name casing).
- Stock remains cache-cheapest per effective token; router v1 is now
  cost-competitive since in-turn caching also covers route round-trips.

## Follow-ups filed

Fix the three task calibrations (corpus-relative check paths, format-
tolerant slot check or format-specified prompt, case-insensitive calendar
equality) and rerun before treating the checks column as capability truth.

# Calibrated baseline (post-fix rerun) — 2026-09-18

Head `afe4b5e`. Single-run samples; treat small deltas cautiously.

| Mode | Completed | All-checks | Claude tokens | Jev tokens |
|---|---|---|---|---|
| auto | 6/8 | 4/8 | 91.6k | 114.0k |
| router (v1) | 5/8 | 4/8 | 60.9k | 63.0k |
| stock | 7/8 | 5/8 | 200.6k | 0 |

This is the reference for the memory auto-injection A/B.

# Memory auto-injection A/B — 2026-09-21

Head `62bbe41` (PR #1). Injection-on arms vs the calibrated baseline
(off-arm valid via the proven byte-identical off-flag). Records:
`results/injection-*.json`.

## The finding: the gate never opens

Across all 16 injection-on runs, ZERO injections occurred. Every gate
score fell in 0.13-0.47 against the 0.6 threshold — including tasks where
recall obviously helps (two-note-synthesis 0.23, seeded-recall 0.19).
All apparent completion deltas (+1/-1 auto, -2 stock) are run variance:
nothing was injected, so the arms were functionally identical.

## Diagnosis

The gate asks Jev an unanswerable question: "would stored memories help?"
judged from task+assistant text alone — Jev cannot see whether relevant
memories EXIST, and (per its documented literal-reading jaggedness) it
correctly hedges low on unverifiable claims. Same conservatism pattern as
compaction triage keep-probabilities. This is a design lesson, not a Jev
defect: speculative gates on unseen evidence do not open.

## Paths forward (Henry's pick)

1. Retrieve-then-judge redesign (recommended): always run the cheap local
   pausanias search (~ms), then Jev judges the CANDIDATES' relevance to
   the next step (the rerank/semantic_find cookbook pattern — a question
   Jev demonstrably answers well). Gate on candidate relevance, not
   speculation.
2. Threshold tuning (weak: scores cluster 0.13-0.47; a 0.3 threshold
   would fire on noise as often as signal).
3. Accept the negative and keep memory tool-mediated only.

Costs held: stock's gate ran per-user-turn at 3.1k Jev tokens total.

# Retrieve-then-judge injection — 2026-09-21

Head `3e32677` (PR #2). Same arms vs the calibrated baseline.
Records: `results/rtj-*.json`.

## The redesign works: injections fire, and discriminate

10 injections across the arms, relevance scores 0.79-0.97, ALL on memory
tasks and ZERO on calendar tasks — the evidence-based question gets the
confident, discriminating answers the speculative gate could not (the
phase-2 calibration pattern, reproduced in production).

| Arm | Completed | All-checks | Claude tokens | vs baseline |
|---|---|---|---|---|
| stock+inject | 8/8 | 6/8 (best ever on this suite) | 129.4k | -35% tokens |
| auto+inject | 8/8 | 3/8 | 36.9k | -60% tokens |
| stock baseline | 7/8 | 5/8 | 200.6k | — |
| auto baseline | 6/8 | 4/8 | 91.6k | — |

## Findings

1. The flagship rescues happened: stock+inject PASSED two-note-synthesis
   (both notes injected at 0.91/0.94) and seeded-recall (0.97) — the exact
   multi-step choreography failures that motivated the feature. Injection
   also collapses tool round-trips: massive token drops in both arms.
2. New hazard, real and specific: STALE-VERSION injection on update
   tasks. stock+inject regressed append-then-latest — the pre-update
   content (0.81) was injected and anchored the "latest" report.
   Refinement candidates: recency weighting, or suppressing injection for
   topics the task is actively modifying (memory_store knows).
3. Auto mode's multi-step flakiness persists independent of injection
   (two-note regressed there despite perfect 0.92/0.94 injections;
   cross-surface regressed with zero injections involved). Injection
   helps most where the base mode is steady: stock.

## Recommendation

stock + retrieve-then-judge injection is the best configuration measured
on this suite: 8/8 completed, best checks, 35% cheaper than plain stock.
Default-on is Henry's call; the stale-version hazard is the one caveat
worth fixing first for daily use.
