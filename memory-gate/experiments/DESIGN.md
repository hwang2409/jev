# Battery Redesign Experiments — Design

Status: DRAFT v2 (after design review round 1)
Context: memory-gate RESULTS.md (2026-09-29). Target use case: POINTER
INJECTION. This is an EXPERIMENT series: non-authoritative, no lock/witness,
ranks candidate batteries only. Every artifact carries machine-readable
`"authority": "experimental"`. Graduation of a winner requires the full
production protocol ON A FRESH SAFETY CORPUS — locomo category-5 becomes
development data the moment these experiments use it for selection, and
rerunning the ceremony on it cannot grant held-out acceptance. The graduation
corpus is chosen at graduation time (options noted in §7); nothing here can
be promoted directly.

## 1. Candidate batteries

- **A (baseline): independent relevance Nouls** — current production
  question. RESCORED interleaved with B and C in this experiment (the
  existing run's scores are historically confounded — no proof of same model
  snapshot; A/B/C requests are interleaved case-by-case in one session to
  control service drift).
- **B (sharper question): answer-bearing Nouls** — independent-Noul shape,
  criteria demand contained information: true = "the excerpt CONTAINS the
  specific information the query asks for"; not_for = "topically related but
  does not contain the requested information".
- **C (forced choice): case-level Choice** — options = candidates present +
  `none_of_these_help`. Scored as a CASE-LEVEL POLICY, never per-candidate
  calibration: surface the argmax candidate iff `1 - P(none) > tau`.
  Evaluated on (a) abstention correctness and (b) selected-candidate
  correctness (was the argmax a labeled positive?). No per-candidate Brier
  or pooled per-candidate sweep from Choice probabilities — they are
  normalized against option count and not comparable to Nouls. All C
  analyses are reported within fixed-arity strata (1-candidate cases vs
  2-candidate cases separately) plus pooled-with-caveat; the arity skew
  (calibration mostly 1-candidate, safety mostly 2-candidate) makes pooled
  cross-lane C numbers descriptive only.
- **D (richer state): deferred** — real-session context capture required;
  fabricating task context onto eval queries measures nothing (unchanged
  rationale, review-endorsed).

## 2. Lanes and controls

- **Calibration lane** (reused): candidates.jsonl + adjudicated labels from
  runs/20260929-jev-b. Construct gap noted: labels judge "useful and safe to
  inject", which is broader than B's "contains the information"; B may
  legitimately score some labeled-positives low (e.g. pointer-worthy context
  that answers indirectly). Reported as a named limitation; label-vs-battery
  disagreement rows are listed in ANALYSIS.md for eyeballing, not silently
  averaged.
- **Safety lane** (reused): 446 locomo category-5 questions with verified
  retrieval (adapter v3). All candidates negative.
- **NEW — within-corpus control lane**: ~60 sampled locomo ANSWERABLE
  questions (categories 1-4, stratified across the 10 conversations), same
  rendering/index (already built), retrieval via the same pausanias
  runner path, production truncate/dedupe/cap-2. Ground truth is AUTOMATIC:
  locomo's evidence annotations — a candidate is positive iff its source
  session/turn matches the question's evidence ids (the PAUS benchmark
  machinery already maps evidence targets; reuse it). This control holds
  corpus, query style, and heading format CONSTANT between positives and
  negatives, isolating battery quality from corpus style. The cross-lane
  separation stat (calibration-positive vs safety-candidate) is demoted to
  an operational descriptive table; the WITHIN-CORPUS control is the
  battery-quality headline.

## 3. Metrics

Per battery:
- **Within-corpus control**: AUC (Noul batteries A/B, per-candidate) /
  policy accuracy by arity stratum (C); this is the headline comparison.
- **Pointer frontier**: full recall-vs-safety-pointer-rate frontier is
  PRIMARY. Illustrative cuts at 5/10/20% safety pointer-rate are display
  points only (no product budget exists yet; explicitly labeled
  illustrative). Calibration pointer-recall for A/B = >=1 labeled-positive
  candidate above tau among cases with >=1 positive; for C = policy surfaces
  a labeled positive. Safety pointer-rate = fraction of 446 cases surfacing
  >=1 pointer (A/B: any candidate above tau; C: policy surfaces).
- **L1 vs adjudicated labels** (A/B only, per-candidate): ROC/PR-AUC, Brier.
  C: abstention-probability distributions per lane + arity stratum.
- **Cost/latency**: honest accounting — A/B and C are all ~1 request per
  case (~50 calibration + 446 safety + ~60 control each); C reduces
  QUESTIONS per request, not requests. Token/latency/cache metrics recorded
  from jm; "pennies" claim replaced by measured numbers in ANALYSIS.md.

## 4. Artifacts (experiment schema, not production artifacts.py)

`experiments/results/<battery>-<stamp>/scores.jsonl` + `meta.json` with a
SMALL dedicated validator (production artifacts.py cannot represent Choice
distributions and enforces the production builder hash — not reused for
experiment rows). Fields: authority:"experimental", battery id, full question
battery hash (experimental builder source hash), per-case: request hash,
option order/count (C), full Choice distribution incl. none (C) or
per-candidate Nouls (A/B), configured+served model ids, error/coverage
records, input-artifact hashes (candidates/labels/safety/control files),
jm/harness/pausanias revisions. Coverage rule inherited: any missing/invalid
score invalidates that battery's run (all-or-nothing per battery per lane).
Same-served-model check across ALL batteries in the comparison; mismatch
refuses the comparison table.

## 5. Execution plan

1. Build: experiments/batteries.py (A/B/C builders+parsers over the
   evaluate_production seam, C parser validates option-probability sanity,
   malformed = request error), experiments/control_lane.py (answerable
   sampling + evidence-derived labels via the pausanias benchmark mappers),
   experiments/run_experiments.py (interleaved A/B/C scoring, per-lane),
   experiments/validator.py + offline tests (fake transports; fixture for
   evidence-mapping correctness; arity-stratum accounting tests).
2. Live: generate control lane (retrieval for ~60 answerable questions);
   score A/B/C interleaved on neenerair; produce ANALYSIS.md (auto tables +
   hand conclusions).
3. Decision heuristic (NOT acceptance): if neither B nor C dominates A on
   the within-corpus control AND the pointer frontier, D gets designed with
   real-session capture. Presented to the user with the frontier in hand.

## 6. Non-goals

Unchanged from v1 (no production changes, no enablement, no new hand
labeling, no D-on-fabricated-context) plus: no reuse of these lanes for
graduation acceptance (§ preamble), no per-candidate calibration claims from
Choice probabilities.

## 7. Graduation note (for later, recorded now)

A winner needs: production builder integration + golden tests, the full
DESIGN.md ceremony, and a FRESH held-out adversarial corpus — candidates:
a second adversarial memory benchmark, a newly constructed locomo-style
split from unused conversational data, or user-curated adversarial queries
over the real vault. Decided then; precommitted as required now.
