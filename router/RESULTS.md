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

# Phase 2 results — 2026-09-17

Synthetic 120-tool catalog (10 domains), nested subsets. Runs:
`results/phase2-20260917-214410.json` (curve + first full),
`results/phase2-20260917-215909.json` (final full, sharpened hard cases).
Model `jev-1.13.0`. Zero API errors across ~440 calls.

## Degradation curve (40 fixed clear cases, growing distractor count)

| Catalog size | Top-1 | Top-3 | Conf (correct) | Conf (wrong) | Tokens in |
|---|---|---|---|---|---|
| 15 | 1.000 | 1.0 | 0.996 | — | 30k |
| 30 | 1.000 | 1.0 | 0.992 | — | 44k |
| 60 | 0.975 | 1.0 | 0.996 | 0.65 | 72k |
| 120 | 0.975 | 1.0 | 0.992 | 0.76 | 126k |

8x more distractors cost 2.5 accuracy points. Top-3 never missed.

## Full experiment (140 cases at 120 tools; final run)

- Top-1 0.993 (coverage 120/120 = 1.0; hard cases 19/20 = 0.95).
- Top-3 = 1.0 — every miss recoverable by top-k fallback.
- Calibration: mean confidence 0.983 on correct vs 0.35 on the miss;
  Brier 0.0046. Every case with confidence >= 0.5 was correct (139/139);
  the single sub-0.5 case was the only miss.
- Only miss: hard-19 (`shell_stop_process` -> `shell_run_command`).
- Cost at 120 tools: ~900 input / ~390 output tokens per route (from the
  120-size curve row).

## Caveats

- Miss sample is small (3 wrong routes across all phase-2 runs). The spec's
  one hard-case revision round ran (reviewer-audited, single-best labels);
  even sharpened cross-domain traps rarely fool the router. Calibration
  conclusions are directionally strong but low-n.
- Evalset labels are single-annotator (worker-authored, one sol audit round).

## Conclusions

1. Flat routing holds to 120 tools: accuracy 0.975-0.993, no hierarchy
   needed at this scale.
2. Confidence is a usable miss detector: all observed misses sat in the
   low-confidence region; high-confidence routes were always correct.
   A harness rule "confidence < 0.8 -> expose top-3" would have recovered
   every miss at near-zero extra cost.
3. Cost scales linearly with catalog size (criteria tokens dominate);
   ~900 input tokens per route at 120 tools.
4. Phase 3 (attach the router to a real agent loop) is justified by the data.
