# Battery Redesign Experiments — Design

Status: DRAFT v1 (design review pending)
Context: memory-gate RESULTS.md (2026-09-29) — the standalone memory_relevance
Noul cannot separate answer-bearing memories from topically-adjacent garbage
(28.25% false-injection at tau=0.58; no viable tau). Target use case is now
POINTER INJECTION ("N memories may be relevant: [titles]"), where a false
positive costs a noise line, not a poisoned turn. This is an EXPERIMENT
series: it ranks candidate batteries; it grants no enablement authority and
uses no lock/witness ceremony. A winning battery graduates to its own
authoritative run under the full DESIGN.md protocol before any production
change.

## 1. Candidate batteries

All scored over the SAME inputs (reused artifacts, §3). Battery A is the
current baseline (already scored — free).

- **A (baseline): independent Nouls** — "is this excerpt relevant to the
  agent's next step?" Existing scores from runs/20260929-jev-b.
- **B (sharper question): answer-bearing Nouls** — same independent-Noul
  shape, criteria rewritten to demand answer-bearing content: true = "the
  excerpt CONTAINS the specific information the query asks for"; not_for =
  "topically related but does not contain the requested information".
  Cheapest change; directly attacks the observed failure mode.
- **C (forced choice): Choice over candidates + none** — one Choice question
  per case: options = candidate_0, candidate_1 (when present), and
  none_of_these_help, with per-option probabilities. Mirrors the router
  battery shape whose calibration was excellent (router RESULTS: Brier
  0.0046). Abstention becomes a first-class option instead of an absent
  pressure.
- **D (richer state): B's question + auto-route-shaped state** — adds
  task/last_assistant context fields to the state. DEFERRED unless B and C
  both fail (real sessions have this context; eval cases only have a query,
  so D on eval data would fabricate context — noted as the reason for
  deferral, revisit with real-session data).

## 2. What we measure (pointer framing)

Per battery, both lanes:

- **Separation**: score distributions of calibration positives vs locomo
  safety candidates (all negative by construction). Report AUC treating
  {calibration positive} vs {safety candidate} as the two classes, plus
  overlap plots as percentile tables (P10/50/90 per class). This is the
  headline number: can ANY threshold separate answer-bearing from
  adversarial-adjacent?
- **Pointer operating points**: for tau in the observed-boundary sweep,
  calibration pointer-recall (>=1 positive-labeled candidate surfaced among
  cases with >=1 positive) vs safety pointer-rate (fraction of the 446
  unanswerable cases surfacing >=1 pointer). Report the frontier and three
  named operating points: max-recall@safety<=5%, @<=10%, @<=20% (pointer
  tolerances; explicitly NOT the 2% content-injection bar).
- **Calibration-lane label agreement (L1)**: ROC-AUC/PR-AUC/Brier against the
  existing adjudicated labels, as before. For battery C, the per-candidate
  score is that candidate's Choice probability; abstention probability
  reported separately (C-specific metric: none-probability distribution on
  safety vs calibration-abstainable cases).
- **Cost/latency** per battery (C halves the question count; report it).

## 3. Reuse and new artifacts

REUSED (no regeneration): calibration candidates.jsonl + labels.jsonl
(labels judge excerpt usefulness — battery-independent by construction);
safety-cases.json (446 questions with verified retrieval, adapter v3);
production pipeline semantics (truncate/dedupe/cap-2) via pipeline.py.

NEW: memory-gate/experiments/batteries.py — battery definitions (question
builders + parsers per battery, same evaluate_production seam and jm cache);
experiments/run_experiments.py — score a battery over both lanes, write
experiments/results/<battery>-<stamp>/{scores.jsonl, analysis.md};
experiments/ANALYSIS.md — cross-battery comparison, auto-generated tables
plus a hand-written conclusions section.

Experimental question shapes live in experiments/ ONLY — providers/jev.py is
untouched until a winner graduates (production fidelity machinery, golden
tests, and the full protocol apply at graduation, not before).

## 4. Validity guards (inherited, lightened where honest)

- Coverage rule unchanged: missing/invalid scores invalidate a battery run
  (artifacts.py validators reused where shapes allow; experiment scores carry
  the same provenance fields incl. configured/served model identity).
- No lock/witness: results are labeled NON-AUTHORITATIVE in every artifact
  header. The safety lane here is an experiment input, not a gate.
- The mass-discard and zero-batteries guards apply (reuse run.py machinery).
- Battery C parser: per-option probabilities must sum sanely (tolerance
  documented); malformed Choice responses are request errors, not zeros.
- Seed/model identity recorded; batteries scored against the same served
  model or the comparison is refused.

## 5. Execution plan

1. Implement batteries.py (B, C) + runner + offline tests (fake transports,
   fixture parity with existing patterns; suite additions, ruff clean).
2. Live-score B and C on neenerair (each: 62 calibration + 886 safety
   batteries for B; ~50+446 Choice calls for C — pennies, cached).
3. Generate ANALYSIS.md; if neither battery reaches max-recall>=80% @
   safety<=10% pointer-rate, D gets designed properly (with real-session
   state capture) rather than faked on eval data.
4. Present analysis; graduation decision is the user's.

## 6. Non-goals

- No production changes, no enablement, no threshold selection authority.
- No battery D on fabricated context (deferral rationale in §1).
- No new labeling (existing adjudicated labels only).
- No auto-injection revival — pointer framing only.
