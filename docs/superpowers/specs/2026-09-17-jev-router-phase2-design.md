# Jev Tool Router — Phase 2 Design (scale + calibration)

Date: 2026-09-17
Status: approved (Henry: "drive this to completion", 2026-09-17)
Depends on: phase 1 (`2026-09-17-jev-tool-router-design.md`, complete at `b2b34e3`)

## Questions phase 2 answers

1. **Degradation curve:** how does routing accuracy and confidence change as
   the catalog grows from 15 to 120 tools with a fixed question set?
2. **Calibration:** on a catalog hard enough to produce real misses, do wrong
   routes get lower confidence than correct routes? (Phase 1 could not test
   this: zero misses.)

Feasibility probed live 2026-09-17: a flat Choice with 120 criteria works
(correct pick, confidence 1.0, ~3.4k input tokens/call). Flat routing; no
hierarchical fallback in phase 2.

## Synthetic catalog (`catalogs.py`)

- `CATALOG_120: dict[str, str]` — 120 tools, 10 domains x 12 tools each.
  Domains: files, shell, web, calendar, email, crm, deploy, data, chat,
  payments. Names: `<domain>_<verb_object>` snake_case (e.g.
  `calendar_create_event`). Descriptions: one line, 10-120 chars, concrete.
- Each domain MUST contain at least 3 near-neighbor tools that are easy to
  confuse (e.g. `email_send_message` / `email_draft_message` /
  `email_reply_thread`; `deploy_rollback_release` / `deploy_promote_release`).
  Confusability is the point: phase 2 needs misses.
- `SUBSETS: dict[int, dict[str, str]]` — nested subsets keyed 15, 30, 60, 120
  with `SUBSETS[15] ⊂ SUBSETS[30] ⊂ SUBSETS[60] ⊂ SUBSETS[120] == CATALOG_120`.
  Every domain is represented in `SUBSETS[15]` (1-2 tools per domain).
  Subsets are defined by explicit name lists, not slicing dict order.
- Phase-1 `catalog.py` stays untouched; phase 2 uses only the synthetic set.

## Evalsets (same JSONL schema as phase 1)

Schema per line: `id`, `task`, `step`, `history`, `expected_tool`,
`expected_needs_tool` (always true here), `vague` (always false here).
The needs_tool/step_clarity gates were validated in phase 1; phase 2 keeps
the same request shape but evaluates only routing.

- `evalset_curve.jsonl` — 40 cases; every `expected_tool` is in `SUBSETS[15]`;
  at least 2 cases per subset-15 tool. Steps mention concrete objects/URLs/
  names, phase-1 style. This set runs unchanged at all four catalog sizes:
  growth adds only distractors, so the curve isolates distractor count.
- `evalset_full.jsonl` — 140 cases; every tool in `CATALOG_120` covered by
  exactly 1 coverage case, plus 20 hard cases deliberately aimed between
  near-neighbors (label the single best tool; the wording must still make
  one answer defensibly correct).

## Runner (`run_phase2.py`, reusing `router.route` with its `catalog` param)

- Curve experiment: for each size in [15, 30, 60, 120], evaluate all 40 curve
  cases against `SUBSETS[size]`. Per size: top-1/top-3 accuracy, mean
  confidence on correct and incorrect, confusion list, tokens.
- Full experiment: evaluate all 140 full cases against `CATALOG_120`.
  Metrics: top-1/top-3 accuracy overall and split coverage vs hard cases;
  confusion list; calibration — mean confidence correct vs incorrect,
  accuracy within confidence bins [0-0.5, 0.5-0.8, 0.8-0.95, 0.95-1.0],
  Brier score on top-1 (predicted prob of chosen tool vs correct 0/1).
- Errors: same rule as phase 1 — errored cases stay in denominators as
  misses, error count reported.
- Output: plain-text report + `results/phase2-<timestamp>.json` with both
  experiments' full per-case results.
- Reuse phase-1 helpers where they fit (`top_k`, `auc` import from
  `run_eval.py` is fine); do not duplicate logic.

## Success/interpretation bar

Phase 2 is measurement, not pass/fail. Deliverable = the curve + calibration
report. Judgments to extract:

1. Where (if anywhere) top-1 accuracy drops below 0.9 on the curve.
2. Whether mean confidence on wrong routes is meaningfully below correct
   routes, and whether low-confidence bins have low accuracy (usable signal
   for a top-k fallback harness).
3. Token cost per route at 120 tools (harness viability).

If the full experiment still yields fewer than 5 misses, the hard cases were
not hard enough — one revision round of the 20 hard cases is in scope before
concluding "calibration unmeasurable".

## Testing

Machine checks only (no network in tests): catalog size/nesting/domain
invariants, near-neighbor presence (>= 3 tools sharing a domain prefix with
overlapping description keywords is NOT machine-checkable — reviewer audits
that), evalset schema/counts/coverage invariants, metric math on stubbed
route functions (including Brier and bin edges), error-as-miss behavior.

## Out of scope

Hierarchical routing, harness integration (phase 3), gate re-validation,
prompt tuning of question instructions.
