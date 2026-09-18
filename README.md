# jev

Experiments with TypeSafe's Jev (System One structured-decision model).

- `router/` — standalone router research: phase 1 (15-tool routing eval),
  phase 2 (120-tool degradation curve + calibration), phase 3 (dual-harness
  sim comparison). This repo tracks it.
- `harness/` — local fork of zeta with Jev integrated (router v1/v2,
  Jev-triage compaction, evals). Its OWN independent git repo (full zeta
  history, no remotes) — intentionally untracked here; never pushed, never
  merged back into upstream zeta.
