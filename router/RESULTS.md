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

# Phase 2 crossover extension — 2026-09-24

This run extends the deterministic catalog generator to 180 and 250 tools.
It compares flat Choice with a category Choice followed by a within-category
Choice. It uses 20 fixed cases from `evalset_curve.jsonl` at each size. The
full artifact is `results/phase2-20260924-crossover.json`.

The router uses `jm.client` through the Vercel AI Gateway. Calls were paced at
2.1 seconds, below the free-tier request rate. The run made 367 planned calls.

## Accuracy and cost

| Size | Flat top-1/top-3 | Flat input tokens | Flat p50 ms | Flat route errors | Hierarchical top-1/top-3 | Hierarchical input tokens | Hierarchical p50 ms | Hierarchical route errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 1.00 / 1.00 | 750 | 194 | 0 | 1.00 / 1.00 | 877 | 400 | 0 |
| 30 | 1.00 / 1.00 | 1,106 | 196 | 0 | 1.00 / 1.00 | 919 | 373 | 0 |
| 60 | 1.00 / 1.00 | 1,789 | 189 | 0 | 1.00 / 1.00 | 988 | 366 | 0 |
| 120 | 0.95 / 0.95 | 3,158 | 241 | 1 | 1.00 / 1.00 | 1,114 | 362 | 0 |
| 180 | 0.70 / 0.70 | 4,672 | 1,030 | 6 | 1.00 / 1.00 | 1,348 | 400 | 0 |
| 250 | 0.45 / 0.45 | 6,438 | 436 | 11 | 1.00 / 1.00 | 1,442 | 402 | 0 |

Flat routing used one call per route. Hierarchical routing used two calls per
route. Hierarchical input cost grows slowly because each tool Choice sees at
most 25 tools. Its p50 latency is about twice flat latency at small sizes,
then becomes lower than flat latency once flat requests start failing.

## Errors and confidence

Flat transport 503 responses increased with catalog size: 0, 0, 0, 13, 28,
and 38 observed attempts at sizes 15 through 250. Final route error rates were
0%, 0%, 0%, 5%, 30%, and 55%. Hierarchical routing had zero final route errors;
it saw one retryable 503 attempt at size 180 and recovered it.

There were no incorrect successful routes in this focused run. Therefore, the
run cannot add a new wrong-route calibration sample. The confidence rescue
rule recovered all successful misses, but it cannot recover a failed API call.
The result records this distinction as `rescued_misses` and
`unrescued_misses`. Each size also records confidence bins and Brier score.
Hierarchical confidence stayed between 0.925 and 0.932 on correct routes; its
three low-confidence cases at each size were all present in top-3.

## Choice option limit

| Options | Result |
|---:|---|
| 250-255 | accepted when the request reached the service; some 503 retries occurred |
| 256 | rejected with `JevError`, `http_status=400`, `attempts=1`, and message `request failed with HTTP status 400` |

The empirical gateway limit is 255 Choice options. The 503 responses near the
limit are transient service failures, not the option-count validation shape.

## Verdict

Use flat routing through 120 tools when one-call latency and simple operation
matter. At 180 tools, hierarchy is already the better default: flat routing
had 30% route errors in this run, while hierarchy stayed at 100% accuracy with
no final errors. At 250 tools, hierarchy is required for reliable operation:
flat routing had 55% route errors and 6.4k input tokens per successful route.

The hard API ceiling is 255 options. Arc-3 should group page elements into
categories and route within the selected category. Keep each category at 25
tools or fewer where possible, and apply the confidence-below-0.8 top-3
fallback to successful low-confidence routes. Do not treat that fallback as a
recovery path for gateway errors.
