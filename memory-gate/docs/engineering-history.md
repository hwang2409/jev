# Engineering history — implementation review round 1

> **Historical document.** This records the findings from the first implementation
> review (commit 4e8205c). The blocking issues and most missing tests have been
> resolved in subsequent phases. Open follow-ups are marked below.

## OPEN follow-ups (from post-approval C2 gate review)

- ~~**T2 cache key test strengthening:** expected cache key still computed via shared
  helpers (behavioral equality with jm runner confirmed by reviewer scratch probe);
  strengthen to invoke `runner.records_for_state` directly.~~
  **DONE (2026-09-29):** reworked test_t2 to use real judge() with fake transport + tmpdir CacheStore; expected key now read from the store the runner wrote to.
- **Second-rename failure test:** trips on `freeze_safety`'s internal rename, never
  reaching run.py rollback; fail on destination path instead.
- ~~**Consolidate calibration response scoring with score_cases:** calibration
  response scoring (run.py:830-879) duplicates the structure of `score_cases`
  (run.py:280-339); consolidate so the paths cannot drift.~~
  **DONE (2026-09-29):** extracted `_score_one_request` shared helper; calibration lane and `score_cases` both call it.
- ~~**Repeatability CLI subcommand:** `run_repeatability` had no CLI entry point
  (the real run drove it via a Python heredoc); add `run.py repeatability
  --run <dir> --tau <t>` wiring subset construction and output.~~
  **DONE (2026-09-29):** added `repeatability` subcommand with stratified subset construction (`build_repeatability_subset`), overwrite refusal, and tests.

---

# Implementation review round 1 (commit 4e8205c) — NOT-MERGE-READY

Verdict: core is placeholder/divergent. Kept as working reference for the
phased re-implementation. Full missing-test inventory below is the acceptance
checklist; each phase claims its subset.

## BLOCKING
1. run.py:326-351 — candidates/score/report/posthoc are placeholders. No
   candidate generation, cached scoring, sweep/Pareto, acceptance eval,
   frozen safety artifact, or post-hoc curve. Safety must require
   --lane safety --witness before any scoring.
2. run.py:146-204 — pipeline not production-equivalent: content_hash must
   whitespace-normalize (loop/__init__.py:169-171); dedupe must be global by
   content hash (routing.py:413-460), not (path,heading,hash); supersession
   must derive from dated headings (routing.py:309-340), not a synthetic
   flag; render must use the real MEMORY_INJECTION_PREFIX. Extract/reuse
   production primitives with imported constants.
3. run.py:79-123,244-265 — §3 coverage rule unenforced: validation must
   establish one-to-one candidate/score, coverage true, finite score in
   [0,1], no request error; ANY violation aborts gating for the run —
   never shrink a denominator.
4. run.py:226-265 — §5 metrics incomplete and bootstrap wrong: missing
   exact-packet, ROC/PR-AUC, Brier, prevalence, ambiguous counts,
   per-stratum any-injection, CIs, Wilson, per-conversation + LOCO.
   Bootstrap must resample eligible cases and recompute num/denom per
   replicate (not average precomputed scalars), null replicates dropped
   and counted.

## HIGH
- test_offline.py:40-46 — selection test only covers strict > and no-gate.
- test_offline.py:70-94 — only 2 of 4 witness refusals tested; witness hash
  in frozen artifact untested.

## MEDIUM
- Network guard must be an autouse socket-refusal fixture for the whole
  suite.

## Missing design-required tests (acceptance checklist)
- Hash-before-truncation (identical first-600, different full hash)
- Whitespace-normalized hashing + cross-path content dedupe
- Top-2 cap before scoring (candidate 3 never in request)
- Production candidate-ID assignment after invalid item discard
- Derived superseded-section skip from dated headings
- Fresh-session actively-modified + invalid-candidate skips
- Retrieval order preserved against opposing score order
- Exact rendered serialization w/ real MEMORY_INJECTION_PREFIX
- Budget at exactly 1500 and 1501; first-over-budget break blocks later short candidate
- Committed golden canonical request bytes at pinned harness revision
- Golden identical parsed scores (production adapter vs eval replay)
- Each invalidity (missing/partial/malformed/out-of-range/null/request-error/coverage:false) invalidates whole run
- Missing/duplicate/extra candidate-score-label records rejected
- Hash recomputation, canonical-request-hash verify, mixed-provenance/model drift rejection
- Fixture replay asserts EVERY reported metric
- No-gate baseline distinct from tau=0 (exact-zero score case)
- Projected precision denominator; ambiguous blocks consume budget, excluded from ratio
- Precision null on zero denominator; empty-injection cases reported separately
- Recall eligibility + retrieval-miss denominators
- Exact-packet vs budgeted order-preserving positive-only optimum
- Forbidden-injection case rate + block count
- Ambiguous pair + affected-case counts
- Retention at zero and nonzero baseline recall
- ROC-AUC, PR-AUC, prevalence, Brier
- Bootstrap eligible sets, seed 20260929, percentile bounds, null-drop counts
- Wilson 95% safety interval
- Per-conversation rates + leave-one-conversation-out
- Sweep = observed boundaries + 0.6 reference
- Pareto frontier + smallest-qualifying-tau selection
- Locomo pinned URL/constant use; 10-conv/446-question assert; max-892 assert
- Acceptance at locked tau; downgrade-to-exploratory path
- Lock refusal: artifact-hash mismatch; witness unreachable after fresh fetch
- Safety CLI refusal without --witness; refusal on pre-existing outputs
- Witness hash recorded in frozen safety artifact
- Locked-tau result frozen before post-hoc; post-hoc labeled non-authoritative
- Repeatability probe: stratified 12 cases, replicate 0 + 5 bypassed, stddev/spread/crossing-fraction, model-identity refusal
- Blinded 25% intra-rater relabel + raw agreement
- Suite-wide no-live-network (autouse socket refusal)

## Post-approval follow-ups (C2 gate round 2, both minor, non-blocking)
- ~~test_contract_required T2: expected cache key still computed via shared helpers
  (behavioral equality with jm runner confirmed by reviewer scratch probe);
  strengthen to invoke runner.records_for_state directly.~~
  **DONE (2026-09-29):** test_t2 now invokes real judge() with fake transport + tmpdir CacheStore.
- Second-rename failure test trips on freeze_safety's internal rename, never
  reaching run.py:565-570 rollback; fail on destination path instead.
