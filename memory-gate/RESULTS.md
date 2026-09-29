# memory-gate RESULTS — first authoritative run (2026-09-29)

## Verdict
**MEMORY_RELEVANCE_GATE cannot be enabled from this eval.** At the locked
tau=0.58, the safety lane measured **126/446 = 28.25% false-injection**
(Wilson 95% [24.3%, 32.6%]) against the acceptance bar of point<=2% /
upper<=4%. Frozen artifact: runs/20260929-jev-b/safety.json (accepts: false,
witness b91a160).

## The numbers
- Ungated fused retrieval (reproduced): 100% false-injection on 446 locomo
  category-5 questions. Lexical baseline (PAUS-15): 1.35%.
- Jev-gated @ tau=0.58: 28.25% — a 3.5x improvement, far from the bar.
- Post-hoc (non-authoritative): <=2% requires tau~0.96-0.97, where
  calibration recall collapses to ~2%. **No tau on the frontier passes
  safety while retaining useful recall.**
- Calibration (62 candidates, 50 cases): recall 0.9375 / precision 1.0 /
  exact-packet 0.9375 at 0.58; ROC-AUC 0.989 vs hand-adjudicated labels.
- Repeatability (12 cases x 6 replicates): per-candidate stddev ~0.01,
  worst spread 0.04, no tau crossings. The failure is calibration of the
  judgment, not jaggedness.

## Interpretation
The standalone memory_relevance Noul separates useful-from-useless well
WITHIN a curated corpus (calibration AUC 0.989) but scores adversarial
same-corpus garbage nearly as high as genuine memories (locomo). The
question, not the threshold, is the limiting factor. Candidate next steps
(future work, not authorized here): richer state (auto-route battery with
task/last_assistant context), a dedicated abstention question, or
retrieval-side hardening before the gate.

## Deviations and incidents (full trail in run dirs + engineering-history)
- Labels are model-annotated (user-delegated): opus-4-6 primary, gpt-5.5
  independent full second pass (96.8% raw agreement), sonnet-4-6 tiebreak
  on 2 rows, pre-committed majority rule. annotation-provenance.json.
- Calibration abstain stratum empty (retrieval-level abstention on the eval
  corpus) -> amended selection rule, documented in report.md.
- Run 20260929-jev safety artifact RETRACTED (vacuous: adapter shape bug
  discarded all candidates; zero batteries scored). Root cause + structural
  guards: jev#79. Re-run in 20260929-jev-b with verified batteries (886).
