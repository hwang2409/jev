# Memory-gate calibration results

**STATUS: AWAITING REAL RUN — this checkout contains no authoritative score or safety claim.**

No authoritative calibration, repeatability, or safety run has been performed in this checkout. The offline runner, artifact schema, report generation, lock/witness protocol, and safety-lane pipeline are implemented and composition-tested (see test suite). The following report structure will be emitted by an immutable run:

## Run identity and provenance

- run directory / timestamp:
- pausanias revision and exact retrieval configuration:
- harness revision / production-builder hash:
- jm revision, configured model, served model:
- case-set and corpus fingerprints:
- score coverage: complete / INVALID (request errors are listed)

## Calibration lane

- candidate and presented-pair counts (after truncate, dedupe, and top-2 cap):
- threshold table: tau, abstain any-injection, packet recall and baseline,
  precision, exact-packet, forbidden-injection, retention, ROC-AUC, PR-AUC,
  Brier, Wilson/bootstrap intervals:
- proposed locked tau and the §8.4 rationale:

## Repeatability (§6)

The report lists the stratified 12-case subset (4 abstain, 4 verbatim, 4
paraphrase), cached replicate 0 plus five cache-bypassed re-issues of the exact
production battery, model identity for every replicate, per-candidate
population standard deviation, worst spread, and fraction crossing tau.
Mixed configured/served identities invalidate the probe.

## Safety lane (§4, §8.3)

- pausanias LOCOMO entry points: `eval.benchmarks.locomo.run.load_dataset`,
  `render_session`/`_render_entries`, and the benchmark's indexed search path;
  memory-gate then filters QA category 5 and applies production skip/dedupe,
  top-2-before-scoring semantics.
- post-filter assertion: 10 conversations / 446 questions:
- scored Nouls (maximum 892):
- locked tau point estimate and Wilson 95% interval:
- acceptance: point <= 2% and upper <= 4%:
- verified witness commit and frozen artifact hash:

## Authority and limitations

`safety.json` is written once and is authoritative only when the witness,
coverage, and acceptance checks pass. `posthoc-safety-curve.json` is explicitly
non-authoritative and cannot change the locked result. The fixed LOCOMO
conversations are clustered (not an iid population); calibration labels are
human labels; this evaluates standalone fresh-session routing only, not
auto-route memory injection. No threshold is enabled by this template.
