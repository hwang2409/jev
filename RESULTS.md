# Phase 1 results — 2026-09-17

Live run: 60 cases, 0 errors, model `jev-1.13.0` (`results/20260917-202530.json`).

| Metric | Value | Bar | Verdict |
|---|---|---|---|
| Top-1 accuracy (46 clear cases) | 1.00 | >= 0.90 | pass |
| Top-3 accuracy | 1.00 | — | pass |
| Confusions | 0 | — | pass |
| Mean confidence on correct routes | 0.978 | — | — |
| Mean confidence on incorrect routes | n/a (no misses) | below correct | not measurable |
| needs_tool AUC (tool vs no-tool cases) | 0.995 | separation | pass |
| step_clarity AUC (clear vs vague) | 1.00 | separation | pass |
| Tokens (60 routes) | 45,721 in / 10,948 out | — | ~762/182 per route |

Notes:

- Success criterion 2 (wrong routes get lower confidence) is vacuously
  untestable at 100% accuracy. The 15-tool catalog is too easy to produce
  misses. Phase 2's larger catalog is required to test calibration.
- `step_clarity` is conservative in absolute terms (mean 0.51 on clear
  steps) but ranks perfectly (AUC 1.0). A harness should threshold it
  relatively, not at 0.5.
- `needs_tool` separates cleanly: mean 0.72 on tool cases vs 0.12 on
  no-tool cases.

Conclusion: phase 1 passes. Proceed to phase 2 (50-150 synthetic tool
catalog, degradation curve, calibration measurement).
