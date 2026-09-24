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

# Phase 2 crossover extension — corrected 2026-09-24

The corrected artifact is `results/phase2-20260924-crossover.json`.

The run uses 20 cases per catalog size. At 180 and 250 tools, 10 cases target
generated tools. Category names no longer expose tool-name prefixes. Some
actions appear in two honest categories, such as email and chat replies.

Flat and hierarchical variants alternate for each case. Catalog sizes use a
fixed-seed shuffle. One global throttle waits 2.1 seconds before every
transport attempt, including retries. Concurrency is one. The run made 367
planned gateway requests before retries.

## Accuracy and cost

The p50 column is paced route wall time. It includes the required throttle.

| Size | Flat top-1/top-3 | Flat input tokens | Flat p50 ms | Flat errors | Hierarchical top-1/top-3 | Hierarchical input tokens | Hierarchical p50 ms | Hierarchical errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 1.00 / 1.00 | 750 | 2,120 | 0 | 1.00 / 1.00 | 998 | 4,200 | 0 |
| 30 | 1.00 / 1.00 | 1,106 | 2,127 | 0 | 1.00 / 1.00 | 1,040 | 4,186 | 0 |
| 60 | 1.00 / 1.00 | 1,789 | 2,120 | 0 | 1.00 / 1.00 | 1,112 | 4,204 | 0 |
| 120 | 0.95 / 0.95 | 3,158 | 2,129 | 1 | 1.00 / 1.00 | 1,276 | 4,203 | 0 |
| 180 | 0.55 / 0.70 | 4,672 | 4,268 | 6 | 0.95 / 1.00 | 1,769 | 4,144 | 0 |
| 250 | 0.25 / 0.35 | 6,437 | 4,378 | 13 | 0.85 / 1.00 | 1,827 | 4,154 | 0 |

Hierarchy keeps the tool Choice small and reduces input tokens. Its two calls
per route double the paced wall time at small sizes. At 180 and 250 tools, it
has much better accuracy and no final route errors.

## Rescue and calibration

The category stage keeps its full probability map. Below 0.8 category
confidence, the tool stage expands across the three most probable categories.
The reported tool probabilities are end-to-end values:

`P(tool) = P(category) × P(tool | candidate categories)`

The Brier score uses that end-to-end probability. Hierarchical Brier scores
were 0.0022, 0.0019, 0.0017, 0.0031, 0.0278, and 0.0289 from 15 through 250
tools. Low-confidence hierarchical rows had 1.00 top-3 accuracy at every
size. At 180, one wrong top-1 route was still in top-3. At 250, three wrong
top-1 routes were still in top-3.

The 180 and 250 cases expose a real synthetic limit. Generated tools reuse
action nouns across services, so category grouping remains favorable when the
prompt supplies workspace context. This is an upper bound under favorable
grouping, not evidence that arbitrary page-element categories will be as
clean. Real Arc-3 data needs a separate category ambiguity evaluation.

## Transport and Choice limit

Flat 503 attempts by size were 2, 3, 2, 9, 28, and 47. Hierarchical 503
attempts were 3, 7, 1, 0, 2, and 1. Retries stayed inside the global throttle.
Final route errors count only requests that exhausted their retry budget.

| Options | Corrected run result |
|---:|---|
| 250-255 | final responses were transient 503 errors in this run |
| 256 | rejected with `http_status=400`, `attempts=1` |

The corrected run confirms the 256-option validation boundary. Earlier
successful 255-option probes remain consistent with a 255-option ceiling.

## Verdict

The round-1 crossover claim was too strong. Its flat-versus-hierarchical
comparison was confounded by block ordering, route-level pacing, and retries.
Its rescue and Brier metrics ignored category uncertainty. Its hierarchy also
used favorable prefix categories and only core tools as targets.

After correction, flat routing remains simpler through 120 tools. At 180 and
250 tools, hierarchy has the better observed tradeoff: lower input cost,
higher top-1 accuracy, and fewer final gateway errors. The result is still an
upper bound under favorable grouping. Arc-3 should use semantic categories,
retain stage probabilities, expand across top categories below 0.8 confidence,
and treat top-3 as a rescue for successful low-confidence routes only.
