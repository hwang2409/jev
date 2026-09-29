# Memory-Gate Calibration Eval — Design

Status: DRAFT v2 (revised after design review round 1)
Owner: henry / zeta-orchestrated
Depends on: pausanias (frozen eval cases + LoCoMo fixtures), jm (Jev client + cache), harness (production adapter)

## 1. Problem

zeta's memory auto-injection pipeline (pausanias search -> per-excerpt Jev
`memory_relevance` Noul -> `MEMORY_RELEVANCE_GATE` -> packet injection) is
implemented but disabled behind uncalibrated thresholds (`# TODO: calibrate`,
"thresholds do not transfer" — Jev jaggedness). PAUS-15 measured 100%
false-injection on unanswerable queries when gating on cosine floors alone;
the Jev gate is the designed fix, but `0.6` is a guess. The router subproject
earned its thresholds with RESULTS.md-grade evals; the memory gate gets the
same treatment before `memory_injection=True` ships.

## 2. Scope declaration (what exactly is calibrated)

**Runtime mode: non-auto routing only.** Production has two request shapes:
the standalone `build_memory_relevance_request` path (state fields: `query`,
`memory_candidates`) and the auto-route path (`build_auto_route_request`:
memory judged jointly with tool routing over `task`, `last_assistant`,
`last_results`). Thresholds must be assumed not to transfer between batteries.
This eval calibrates the **standalone path**; auto-route calibration is a
follow-up reusing the same machinery with its own battery and its own
threshold. Enabling injection under auto routing is NOT authorized by this
eval's results.

**Conversation state: fresh session.** Production's stateful skips
(known-memory dedupe by content hash, actively-modified, superseded) are
replicated in their fresh-conversation form: no known or actively modified
memories at scoring time. Stated as a limitation in RESULTS.md.

## 3. Pipeline fidelity (primary lane semantics)

The primary lane replicates `routing.py` exactly, in order:

1. retrieval (pausanias search, production configuration and query formation
   for the standalone path)
2. per-candidate excerpt truncation to `MEMORY_INJECTION_EXCERPT_CHARS` (600)
   — content hash computed BEFORE truncation, as production does
3. skip/dedupe pass (fresh-session semantics)
4. **cap at `MEMORY_INJECTION_TOP_K` (2) BEFORE scoring** — production sends
   at most 2 candidates to Jev, so the primary lane scores exactly the
   candidates production would score
5. score via the **complete production adapter** — request construction,
   `runtime_preset`/`State` formation, battery composition, parser, and
   error/coverage behavior of the standalone path; not a hand-rolled call to
   the question builder
6. gate at strict `score > tau`, preserving retrieval order (production never
   reranks by Jev score)
7. render blocks (`MEMORY_INJECTION_PREFIX`, path, heading, newlines, excerpt)
   and apply the `MEMORY_INJECTION_TOTAL_CHARS` (1500) cap to the FULL
   rendered block, breaking on the first over-budget eligible candidate

Implementation reuses or faithfully extracts the production packet-selection
function, with boundary tests for: exact serialization, strict `>`, retrieval
order, first-over-budget break, dedupe, and skip behavior. A **golden test**
asserts byte-equivalent canonical request state/questions and identical parsed
scores between the eval path and the production adapter at a pinned harness
revision.

## 4. Datasets and gold labels

**Calibration lane (development): pausanias frozen cases** (`eval/cases.json`,
59 cases; 49 answerable, 10 abstain).

- Candidates: the exact <=2 presented excerpts per case from the primary-lane
  pipeline -> at most 118 (query, presented-excerpt) pairs.
- **Gold: hand-labeled presented excerpts.** Path membership is provenance,
  not truth: a section from an expected file can be useless, and a useful
  excerpt can come from elsewhere. Henry labels each presented (truncated)
  excerpt against written guidance as positive ("useful and safe to inject
  for this next step") / negative / ambiguous. Ambiguous pairs are excluded
  from primary metrics and reported. A stratified 25% subset is
  second-annotated (fresh-context session) for agreement. At <=118 pairs this
  is roughly an hour.
- Path annotations retained separately: `expected_source_paths` (provenance),
  `forbidden_paths` (harm flag — forbidden-injection reported as its own
  metric, never averaged away).

**Safety lane (held-out, REQUIRED): LoCoMo abstention set**
(`tests/fixtures/locomo/abstention.json`, category-5 adversarial, 446
questions). All candidates are negative by construction (unanswerable), so no
hand-labeling is needed. This is the only set large enough to support a
near-2% false-injection claim, and it directly represents the PAUS-15 failure.
The chosen threshold is **validated** here, never developed here. If its
candidate generation proves infeasible, the eval's conclusion is downgraded to
"exploratory calibration — do not enable injection," explicitly.

**Judge-stress lane (secondary, reported only):** N=8 retrieval candidates per
case, scored in production-shaped batteries, labeled by the same hand-label
process where feasible; used for judge-quality analysis (score distributions,
hard-negative behavior). **Cannot influence the chosen threshold** — it scores
a candidate population production never sees.

## 5. Metrics (denominators defined)

Per swept threshold (sweep = all unique observed score boundaries under strict
`>`, plus the current 0.6 for reference):

- **Abstain/safety strata:** any-injection rate = cases with >=1 injected
  block / all abstain cases. This is the PAUS-15-comparable false-injection
  metric. Reported with binomial 95% CIs on both lanes (the 10-case
  calibration stratum is directional only; the 446-question safety lane is
  the citable number).
- **Answerable stratum (calibration lane):** packet recall = cases where >=1
  positive-labeled block injected / cases with >=1 positive available among
  presented candidates (cases with no positive presented are reported as a
  retrieval-miss rate — the gate cannot fix retrieval and is not penalized
  for it). Packet precision = positive injected blocks / injected blocks,
  over cases that injected anything; the no-injection case count is its own
  reported figure, not folded into precision. Exact-packet rate = cases whose
  injected set equals the labeled-optimal set. Forbidden-injection rate
  reported separately. Absolute values AND retention-vs-tau=0, both with
  bootstrap CIs (case-level resampling).
- **Judge quality (L1, pair-level, calibration + stress lanes):** ROC-AUC,
  PR-AUC with prevalence baseline, Brier, ECE with per-bin counts; reported
  case-macro and pair-micro. Clustered/imbalanced caveats stated inline.

## 6. Jaggedness probe

Production battery shape is preserved: one request = one case's <=2 candidates.
For a stratified 12-case subset (4 abstain / 4 verbatim / 4 paraphrase),
re-issue the exact production battery 5x with jm cache bypassed (the original
cached score is replicate 0; 5 fresh replicates follow). Report per-candidate
score stddev, worst-case spread, and within-battery vs across-battery
variation. Configured AND served model identity recorded per replicate; runs
mixing model identities are invalid and refused.

## 7. Artifacts and replay

One immutable run directory per (request-version, model-identity):
`memory-gate/runs/<stamp>-<jev-model>/` containing `candidates.jsonl`,
`scores.jsonl`, `labels.jsonl`, `report.md`. Artifact schema (validated, drift
fails offline tests): case-set fingerprint, corpus fingerprint, pausanias
revision + retrieval config + rank, untruncated-excerpt hash, presented
excerpt, canonical request hash, production-builder hash, configured + served
model ids, harness/jm revisions, coverage/failure records. Offline CI replays
committed artifacts only — it never consults the live service; a served-model
change requires a new run directory and recalibration, not a red offline test.
Excerpts originate from pausanias's own synthetic eval corpus and LoCoMo
fixtures — safe to commit; confirmed at review time.

## 8. Threshold selection and acceptance

1. Develop tau on the calibration lane: sweep observed boundaries, produce the
   metrics table, identify the Pareto frontier of (abstain any-injection rate,
   packet recall).
2. Proposed rule: smallest tau with calibration-lane abstain any-injection = 0
   observed (only 10 cases — directional) and packet recall >= 90% of its
   tau=0 value, with absolute recall also reported.
3. Validate that tau on the safety lane: REQUIRED acceptance = point estimate
   <= 2% false-injection AND 95% CI upper bound <= 4% on the 446-question set.
   (PAUS-15 lexical baseline: 1.35%.)
4. If no tau satisfies both lanes, publish the Pareto set and stop — the
   recall/abstention trade is a product call, escalated with the frontier in
   hand, not decided by the eval.
5. Enabling `memory_injection=True` (non-auto mode only) requires: acceptance
   met, RESULTS.md reviewed, and the golden request-equivalence test green at
   the harness revision being enabled.

## 9. Deliverables

```
memory-gate/
  DESIGN.md            (this doc)
  run.py               subcommands: candidates | score | label-template | report
  labeling-guide.md    written gold-label guidance + examples
  runs/<stamp>-<model>/  immutable artifacts per run (schema-validated)
  RESULTS.md           tables, chosen tau, CIs, jaggedness, cost/latency, limitations
  test_offline.py      replay + schema + golden request-equivalence tests (no network)
```

Textual tables only (no plotting dependency). Three lanes, one runner, no
other machinery.

## 10. Non-goals

- Auto-route battery calibration (follow-up; same machinery, own threshold).
- Tuning pausanias retrieval (floors, RRF, synonyms) — frozen as-is.
- Multi-message query construction for the hook path.
- Stateful-session skip semantics (fresh-session only; stated limitation).
- Any production default change before RESULTS.md exists and is reviewed.

## 11. Resolved design questions (review round 1)

1. Path-level gold rejected as primary labels -> hand-labeled presented
   excerpts with ambiguous/exclude outcome and audited subset (§4).
2. Model identity: pinned and recorded per run; offline tests fail on schema/
   hash/mixed-provenance drift, never on live-service drift (§6, §7).
3. N=2 primary (production truth), N=8 demoted to judge-stress lane that
   cannot pick the threshold (§4).
4. Import-the-builder rejected as insufficient -> reuse the complete
   production adapter for the declared mode + golden equivalence test (§3).
