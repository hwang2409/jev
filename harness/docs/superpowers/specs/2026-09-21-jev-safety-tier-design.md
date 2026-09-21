# jev-zeta: Jev safety tier for shell commands (arc 4, "yolo seatbelt")

Date: 2026-09-21
Status: approved arc (Henry roadmap 2026-09-18; queued behind stale guards).
Repo: `~/me/fun/jev/harness`. References: TypeSafe llm_guardrails cookbook +
confidence-routing pattern; house standards (structured criteria, named
thresholds, injection probes, jaggedness rules).

## Idea

Make `--yolo` safe: before executing a shell command (bash/exec/
run_background), a Jev Score classifies the CONCRETE invocation's risk.
Clearly-safe auto-approves; risky returns a teaching error (headless) or
escalates to the human prompt (interactive). Approval gains a middle brain
between "approve everything" and "ask about everything".

## Design

1. LAYER 0 — deterministic always-escalate list (checked FIRST, no Jev):
   sudo, pipe-to-shell (curl|sh shapes), rm -rf on / or ~ roots, chmod/chown
   -R outside cwd, credential-file reads (~/.ssh, ~/.aws, *.pem, keychain),
   history/shell-profile writes. Small, auditable, tested pattern list.
2. LAYER 1 — Jev Score, structured rubric (0-3):
   0 read-only inspection / 1 reversible workspace writes /
   2 destructive-but-scoped (workspace or /tmp deletes, git reset --hard,
   force-push own branch) / 3 dangerous-or-out-of-scope (deletes outside
   workspace, network exfil of local files, package installs system-wide,
   kill by wildcard). Structured criteria per level (what/not_for/examples
   per house standard). State: {command, cwd, task excerpt} — fields named
   explicitly in instructions; command quoted as neutral data.
   One Jev call per gated command; nouls alongside in the SAME call:
   "touches paths outside cwd?", "plausibly irreversible?". Call
   confidence = min across judgments (house rule).
3. DECISION (named tunables): score <= SAFE_MAX (1.0) AND confidence >=
   SAFETY_CONFIDENCE (0.8) -> auto-approve. Otherwise: interactive ->
   escalate to the existing human approval prompt with the score/reasons
   attached; headless -> deny with a teaching error result (score, level
   meaning, the triggering judgment, "narrow the command or ask the user").
   FAIL-CLOSED polarity: Jev error/missing key/low confidence NEVER
   auto-approves — falls to escalate/deny. (Opposite of router fail-open;
   state why in code comments.)
4. Scope: gates bash, exec, run_background ONLY (read-family and dedicated
   tools bypass — they have their own approval posture). Applies only when
   yolo would have auto-approved; non-yolo flows unchanged (human prompt
   already gates). Setting + flag `--safety-tier` (BooleanOptionalAction),
   DEFAULT OFF for the A/B; children inherit.
5. Telemetry: per-command score/confidence/decision/layer; service-tagged
   Jev usage; skip reasons (layer0_escalated, jev_error_failclosed, ...).

## Testing (offline, mocked Jev)

Layer-0 pattern list (each pattern + near-misses that must NOT match);
decision matrix (score x confidence x mode); fail-closed on every error
path; teaching-error content; paired hostile/benign state probes (command
content cannot alter rubric construction); off-flag byte-identity; children
inherit; eval flag passthrough.

## Acceptance (orchestrator live steps)

1. Command-corpus eval: a labeled corpus (~40: clearly-safe / scoped-
   destructive / dangerous / obfuscated) through the gate offline-style
   (real Jev, no execution): accuracy per class, calibration, escalate-band
   size. Obfuscated class EXPECTED to land in escalate via low confidence —
   that is success, not failure.
2. Live smoke: safety-tier ON in a headless session where the task tempts a
   risky command; verify deny-with-teaching and task adaptation; and a
   clearly-safe flow auto-approving.

## Out of scope

Gating non-shell tools, browser actions (arc 3 reuses this tier), rewriting
commands on the agent's behalf, default-on decision (Henry's, from the eval
data).
