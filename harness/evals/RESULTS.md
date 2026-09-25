# offline browser evaluation results — 2026-09-25

The fixed offline runner compares routed and stock arms on the same seeded
fixtures, observable goals, page states, budgets, and safety policy.

The corpus has 60 tasks. It covers the full 5 x 3 x 4 matrix:

- catalog sizes: 10, 40, 120, 500, and 2,000;
- churn: static, moderate, and full;
- task shapes: clear target, repeated label, search triage, and multi-step form;
- seeded stale mutation and recovery;
- nine seeded moderate mutations across 30 moderate steps;
- five denied safety classes in the task truth.

The seed is 17. The four shared budget settings are:

- page Jev calls: 8;
- page Jev tokens: 12,000;
- browser actions: 20;
- wall clock: 120 seconds.

## headline

The routed arm completes 36/60 tasks (0.6000). The stock arm completes 26/60
(0.4333). Both arms have zero false approvals over 4/4 denied-class risky
attempts. The routed arm wins 14 of 15 matrix cells and loses one. The loss is
the 10-element static cell, where both arms complete 0/4 tasks.

These are the reported numbers. They do not force a routing-win claim.

| metric | routed | stock large-tool |
| --- | ---: | ---: |
| top-1 accuracy, attempted | 45/47 (0.9574) | 36/47 (0.7660) |
| top-3 coverage, attempted | 46/47 (0.9787) | 37/47 (0.7872) |
| page-state accuracy, attempted | 60/60 (1.0000) | 60/60 (1.0000) |
| search-triage accuracy, attempted | 14/15 (0.9333) | 11/12 (0.9167) |
| task success | 36/60 (0.6000) | 26/60 (0.4333) |
| selection unattempted | 13/60 | 13/60 |
| prefilter recall | 111/123 (0.9024) | n/a, no prefilter |
| false approval rate | 0/4 (0.0000) | 0/4 (0.0000) |
| Jev tokens (modeled) | 35,431 | 129,799 |
| Jev cost (modeled) | 0.038723 | 0.132601 |
| Jev cost per successful step (modeled) | 0.000587 | 0.002763 |
| provider turns | 218 | 187 |
| stale recovery | 23/23 (1.0000) | 18/18 (1.0000) |
| stale rejections | 33 | 24 |
| budget-exhausted tasks | 5 | 18 |
| time per successful step (modeled) | 0.120833 s | 0.349542 s |

Accuracy rates exclude unattempted records. The denominator appears beside each
rate. The runner reports 13 unattempted selection records in each arm.

## exclusive failure counts

Each record has one first-failure category. Safety denials remain separate from
this table. A prefilter miss stops attribution before later selection stages.

| first failure category | routed | stock large-tool |
| --- | ---: | ---: |
| pre_filter_miss | 12 | 0 |
| search_triage_miss | 1 | 1 |
| jev_selection_miss | 2 | 11 |
| adapter_failure | 0 | 0 |
| page_failure | 0 | 0 |
| budget_exhausted | 5 | 18 |

## caveats

- The fake adapter uses deterministic pages. This ceiling inflates both arms.
- Token counts are modeled from the offline transport formula.
- Cost is modeled from input and output token formulas.
- Time is modeled from shared stage-cost formulas.
- The stock arm has no prefilter, so its prefilter recall is not applicable.
- The fixed call budget prevents one denied-class task from reaching its final
  safety stage. False-approval accounting uses only runtime attempts.

The runner uses no network, browser provider, or provider key.

Command:

`uv run --frozen python -m evals.browser_eval --tasks evals/browser_tasks.jsonl --output /tmp/browser-eval-fixed.json`

Targeted verification:

`uv run --frozen pytest -q --tb=short tests/test_browser_evals.py tests/test_module_limits.py::test_module_limits`

Result: 21 passed.
