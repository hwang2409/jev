# Memory-Gate Calibration Eval — Design

Status: DRAFT v5 (revised after design review rounds 1-4)
Owner: henry / zeta-orchestrated
Depends on: pausanias (frozen eval cases + locomo10 benchmark runner), jm (Jev client + cache), harness (production adapter)

## 1. Problem

zeta's memory auto-injection pipeline (pausanias search -> per-excerpt Jev
`memory_relevance` Noul -> `MEMORY_RELEVANCE_GATE` -> packet injection) is
implemented but disabled behind uncalibrated thresholds (`# TODO: calibrate`,
"thresholds do not transfer" — Jev jaggedness). PAUS-15 measured 100%
false-injection on unanswerable queries when gating on cosine floors alone
(fused mode; lexical baseline 6/446 = 1.35%); the Jev gate is the designed
fix, but `0.6` is a guess. The router subproject earned its thresholds with
RESULTS.md-grade evals; the memory gate gets the same before
`memory_injection=True` ships.

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

**Score coverage is a validity requirement, not a metric.** The production
adapter raises on missing/partial coverage and production then injects
nothing — so scoring failures LOOK like abstention. To prevent a false safety
pass: every candidate reaching step 5 in the calibration lane, and every
candidate in the safety lane, must have a complete valid score for the run to
be usable for threshold selection or gating. There is no failure budget:
any missing/invalid score, request error, or partial battery invalidates the
run for gating purposes (transient failures may be retried through the cache;
what cannot be completed invalidates). Request errors and per-lane coverage
are reported in RESULTS.md regardless.

## 4. Datasets and gold labels

**Calibration lane (development): pausanias frozen cases** (`eval/cases.json`,
59 cases; 49 answerable, 10 abstain).

- Candidates: the exact <=2 presented excerpts per case from the primary-lane
  pipeline -> at most 118 (query, presented-excerpt) pairs.
- **Gold: hand-labeled presented excerpts.** Path membership is provenance,
  not truth. Henry labels each presented (truncated) excerpt against written
  guidance (`labeling-guide.md`) as positive ("useful and safe to inject for
  this next step") / negative / ambiguous. A stratified 25% subset is
  relabeled by Henry in a later blinded session; this measures **blinded
  intra-rater agreement** (raw percent agreement reported; it is consistency,
  not independent truth — stated limitation). Ambiguous handling is defined
  in §5.
- Path annotations retained separately: `expected_source_paths` (provenance),
  `forbidden_paths` (harm flag — forbidden-injection reported as its own
  metric, never averaged away).

**Safety lane (held-out, REQUIRED): locomo10 category-5 questions.** The
committed fixture `tests/fixtures/locomo/abstention.json` is a 2-question
unit-test fixture and is NOT this lane. The safety lane uses the full pinned
locomo10 dataset via `pausanias/eval/benchmarks/locomo/run.py`, with one
prerequisite fix (upstreamed to pausanias before this eval runs): the
runner's `DATASET_URL` currently fetches mutable `main`; it must fetch the
pinned commit directly
(`https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json`),
retaining the existing SHA-256 validation, so clean regeneration cannot break
when upstream moves. The lane filters to category 5 and the memory-gate
runner enforces its own post-filter assertion of **10 conversations / 446
questions** (the pausanias runner permits conversation subsets; we refuse
them). **Candidate semantics are production semantics:** PAUS-15
rendering/ingestion and declared retrieval configuration produce the ranked
list, then the production sequence applies — skip/dedupe, top-2 cap BEFORE
scoring — exactly as in §3. The lane therefore scores at most 446 x 2 = 892
Nouls, never the retrieval top-200 (scoring candidates production never
presents would be both wasteful and non-representative). All candidates are
negative by construction (unanswerable); no hand-labeling needed. The chosen
threshold is **validated** here under the blinding protocol of §8, never
developed here. If dataset fetch/generation proves infeasible, the eval's
conclusion is downgraded to "exploratory calibration — do not enable
injection," explicitly.

**Deferred (not in this eval):** the N=8 judge-stress lane and ECE analysis
from v2 are dropped until both required lanes pass; if revisited, they get a
predeclared sample cap and a single concrete diagnostic question. Rationale:
they cannot affect the threshold and their statistics are unstable at this
sample size.

## 5. Metrics (formulas and edge rules)

Let a *case* be one calibration-lane case; *blocks* are rendered-and-injected
excerpts. The **no-gate baseline** is a separate construct (gate bypassed
entirely, all scored candidates eligible), not tau=0 (strict `>` makes tau=0
exclude exact-zero scores).

Sweep set: all unique observed scores as boundaries (plus 0.6 for reference).
Per threshold tau:

- **Any-injection rate (abstain / safety):** cases with >=1 injected block /
  all VALID cases in that lane (validity per §3 coverage rule; an invalid
  case fails the run, it never shrinks a denominator). This is the
  PAUS-15-comparable false-injection metric.
- **Packet recall (answerable):** cases with >=1 positive-labeled block
  injected / cases with >=1 positive-labeled candidate among presented
  candidates. Cases with no positive presented are excluded from recall and
  counted in a separately reported **retrieval-miss rate** = such cases /
  all answerable cases (the gate cannot fix retrieval).
- **Packet precision:** positive-labeled injected blocks / **non-ambiguous
  injected blocks** (the projected denominator; ambiguous injected blocks are
  excluded from numerator and denominator but still consume budget in the
  simulation), computed block-micro across the lane (single global ratio).
  If the projected denominator is 0 at some tau (guaranteed at the maximum
  observed boundary), precision is reported as `null`, never 0 or 1. The
  number of cases injecting nothing is reported alongside, never folded in.
- **Exact-packet rate:** cases whose injected set equals the **labeled-optimal
  set** / all answerable cases with >=1 positive presented. Labeled-optimal
  set = the result of running the production selection (order-preserving,
  budget-capped, first-over-budget break) over only the positive-labeled
  candidates. This is by construction achievable under production semantics.
- **Forbidden-injection rate:** cases injecting >=1 block whose source path is
  in `forbidden_paths` / all valid cases in the lane (case-level; block count
  also reported).
- **Ambiguous handling:** ambiguous-labeled candidates are excluded from
  recall/precision/exact-packet numerators and denominators (projection onto
  non-ambiguous candidates); they still occupy their production slot in
  pipeline simulation (they can consume budget — reality is preserved, only
  scoring is agnostic). Count of ambiguous pairs and affected cases reported.
- **Retention:** for packet recall only, recall(tau) / recall(no-gate
  baseline); undefined (and reported as such) if baseline recall = 0.
- **Judge quality (pair-micro only):** ROC-AUC and PR-AUC over all labeled
  non-ambiguous pairs, with positive prevalence stated; Brier score. No
  case-macro AUC (undefined for single-class cases); no ECE (deferred, §4).
- **Zero-denominator policy (global):** any metric whose denominator is
  empty at a given tau (or within a bootstrap replicate) is `null` for that
  evaluation; `null` replicates are dropped from the CI with the drop count
  reported. `null` is a first-class reported value, distinct from 0.
- **Confidence intervals:** answerable-lane case metrics get case-level
  bootstrap CIs (10,000 resamples, seed 20260929, percentile method),
  resampling from **each metric's eligible case set** (recall resamples
  cases with >=1 positive presented; exact-packet likewise; any-injection
  resamples the full stratum). Safety
  lane gets a Wilson binomial 95% interval for comparability, PLUS
  per-conversation rates and leave-one-conversation-out sensitivity — the
  446 questions cluster in 10 conversations and are not independent draws;
  the gate is defined as performance **on this fixed benchmark**, not a
  population claim.

## 6. Repeatability probe (renamed from "jaggedness")

Measures score repeatability, not composition sensitivity (only one battery
composition exists per case; paired-perturbation designs are future work).
For a stratified 12-case subset (4 abstain / 4 verbatim / 4 paraphrase):
re-issue the exact production battery 5x with jm cache bypassed (cached
original = replicate 0). Report per-candidate replicate stddev, worst-case
spread, and the fraction of candidates whose replicate range crosses the
chosen tau. Configured AND served model identity recorded per replicate;
runs mixing model identities are invalid and refused.

## 7. Artifacts and replay

One immutable run directory per (request-version, model-identity):
`memory-gate/runs/<stamp>-<jev-model>/` containing `candidates.jsonl`,
`scores.jsonl`, `labels.jsonl`, `report.md`. Artifact schema (validated, drift
fails offline tests): case-set fingerprint, corpus fingerprint, pausanias
revision + retrieval config + rank, untruncated-excerpt hash, presented
excerpt, canonical request hash, production-builder hash, configured + served
model ids, harness/jm revisions, request-error and coverage records. Offline
CI replays committed artifacts only — it never consults the live service; a
served-model change requires a new run directory and recalibration, not a red
offline test. Calibration-lane excerpts come from pausanias's synthetic eval
corpus (safe to commit). Safety-lane excerpts derive from the fetched locomo10
dataset: scores and hashes are committed, raw locomo text is NOT (regenerated
deterministically from the pinned fetch; regeneration documented in run.py).

## 8. Threshold selection, blinding, and acceptance

1. Develop tau on the calibration lane only: sweep observed boundaries,
   produce the metrics table, identify the Pareto frontier of (abstain
   any-injection rate, packet recall).
2. Proposed rule: smallest tau with calibration-lane abstain any-injection = 0
   observed (10 cases — directional only) and packet recall >= 90% of the
   no-gate baseline, absolute recall also reported.
3. **Blinding protocol (runner-enforced, externally witnessed):**
   `run.py lock` writes tau + the calibration artifact hashes to
   `runs/<...>/LOCK.json`, and that file is **committed and pushed to the
   remote before any safety-lane scoring** — the pushed commit hash is the
   witness that the lock preceded unblinding (a hash sitting in a mutable
   local directory proves nothing). `run.py score --lane safety` takes
   `--witness <commit>` as a REQUIRED input and refuses to run unless ALL
   hold, verified at invocation time: (a) LOCK.json exists and its hashes
   match the present calibration artifacts; (b) the exact bytes of LOCK.json
   are contained in the witness commit (`git show <commit>:<path>` compared
   byte-for-byte); (c) after a fresh `git fetch` of the designated remote,
   the witness commit is reachable from the remote-tracking ref — i.e. the
   lock is provably published, not merely local; (d) no safety outputs
   already exist. The verified witness commit hash is recorded inside the
   frozen safety artifact. The locked-tau pass/fail result is computed and
   frozen first; only then does a separate `run.py posthoc-safety-curve`
   command exist, whose artifact is permanently labeled non-authoritative —
   it cannot authorize a revised threshold; revision requires new held-out
   data. The §5 sweep applies to the calibration lane only; the safety lane
   is evaluated at the locked tau (plus the labeled post-hoc artifact).
4. Safety acceptance (REQUIRED): at the locked tau, on the 446-question lane
   with complete coverage: false-injection point estimate <= 2% AND Wilson
   95% upper bound <= 4%. (Lexical baseline: 6/446 = 1.35%, which passes
   both.) Per-conversation and leave-one-out figures reported alongside.
5. If no tau satisfies rule 2, or the locked tau fails rule 4, publish the
   frontier/results and stop — the recall/abstention trade is a product call,
   escalated with data in hand, not decided by the eval.
6. Enabling `memory_injection=True` (non-auto mode only) requires: acceptance
   met, RESULTS.md reviewed, and the golden request-equivalence test green at
   the harness revision being enabled.

## 9. Deliverables

```
memory-gate/
  DESIGN.md            (this doc)
  run.py               subcommands: candidates | score | label-template | lock | report | posthoc-safety-curve
  labeling-guide.md    written gold-label guidance + examples
  runs/<stamp>-<model>/  immutable artifacts per run (schema-validated)
  RESULTS.md           tables, locked tau, CIs, repeatability, cost/latency, limitations
  test_offline.py      replay + schema + golden request-equivalence tests (no network)
```

Textual tables only (no plotting dependency). Two required lanes, one runner.

## 10. Non-goals

- Auto-route battery calibration (follow-up; same machinery, own threshold).
- Tuning pausanias retrieval (floors, RRF, synonyms) — frozen as-is.
- Multi-message query construction for the hook path.
- Stateful-session skip semantics (fresh-session only; stated limitation).
- N=8 judge-stress lane and ECE (deferred; predeclared cap if revisited).
- Any production default change before RESULTS.md exists and is reviewed.

## 11. Resolved design questions (rounds 1-2)

1. Path-level gold rejected -> hand-labeled presented excerpts; blinded
   intra-rater agreement on 25% subset, named as consistency not truth (§4).
2. Model identity: pinned and recorded per run; offline tests fail on schema/
   hash/mixed-provenance drift, never on live-service drift (§6, §7).
3. N=2 primary (production truth); N=8 deferred entirely (§4).
4. Complete production adapter reuse + golden equivalence test (§3).
5. Safety lane = pinned full locomo10 category-5 (10 conversations / 446
   questions asserted), not the 2-question committed fixture (§4).
6. Coverage failures invalidate runs; they can never manufacture a safety
   pass (§3).
7. Safety lane blinded via locked-threshold protocol (§8).
8. Clustering acknowledged: fixed-benchmark claim + Wilson interval +
   per-conversation sensitivity, not a population claim (§5).
