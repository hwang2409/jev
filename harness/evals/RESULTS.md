# offline browser evaluation baseline — 2026-09-25

The fixed runner compares routed and stock arms through one shared stage
pipeline. Both arms use the same fixtures, prompts, budgets, page gates, and
safety policy.

The corpus has 60 tasks. It covers the full 5 x 3 x 4 matrix:

- catalog sizes: 10, 40, 120, 500, and 2,000;
- churn: static, moderate, and full;
- task shapes: clear target, repeated label, search triage, and multi-step form;
- mutation-before-action stale rejection and recovery;
- all five denied safety classes: destructive, payment, authentication,
  external-origin, and download.

## headline

The routed arm completes 40/60 tasks (66.67%). The stock arm completes 28/60
(46.67%). Both arms have zero risky false approvals and cover all five denied
classes.

The matrix has 14 qualifying routed cells and one no-win cell. The 10-element
static cell has zero successful tasks in both arms, so it does not qualify as a
win. This is the fixed runner's result, not a forced routing claim.

| metric | routed | stock large-tool |
| --- | ---: | ---: |
| top-1 accuracy | 0.7500 | 0.8000 |
| top-3 coverage | 0.7667 | 0.8167 |
| page-state accuracy | 1.0000 | 1.0000 |
| task success rate | 0.6667 | 0.4667 |
| risky false-approval rate | 0.0000 | 0.0000 |
| Jev tokens | 39,055 | 133,761 |
| Jev cost | 0.042451 | 0.136469 |
| Jev cost per successful step | 0.000663 | 0.003102 |
| model input tokens | 35,659 | 131,053 |
| model output tokens | 3,396 | 2,708 |
| provider turns | 222 | 180 |
| stale recovery rate | 0.7000 | 0.5000 |
| stale rejections | 30 | 24 |
| time per successful step | 0.128953 s | 0.387318 s |

The runner enforces these budgets in both arms: 8 page Jev calls, 12,000 Jev
tokens, 20 browser actions, and 120 wall-clock seconds. The stock arm reaches
the Jev budget on 15 tasks. The routed arm reaches none.

Time, token use, and cost are modeled offline values. The mock has no provider
key and receives no target id or target label. It ranks from the task prompt,
action type, and visible catalog. Search triage has 93.33% routed accuracy and
91.67% stock accuracy. The ambiguous search task fails when triage chooses the
wrong result.

Failure categories remain separate. Routed: 15 Jev selection misses. Stock:
12 Jev selection misses. Adapter and page failures are zero in this corpus and
have dedicated regression coverage.

The crossover matrix uses task-success and safety parity first. A token
reduction alone cannot qualify. The 10-element static cell is the no-win cell;
the other 14 cells qualify on modeled cost or modeled time with the reported
quality and safety results.

Command:

`uv run --frozen python -m evals.browser_eval --tasks evals/browser_tasks.jsonl --output /tmp/browser-eval-fixed.json`

Targeted verification:

`uv run --frozen pytest -q --tb=short tests/test_browser_evals.py tests/test_evals.py tests/test_module_limits.py::test_module_limits`

Result: 49 passed.
