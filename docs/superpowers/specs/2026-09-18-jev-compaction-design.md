# jev-zeta: Jev-triage compaction design

Date: 2026-09-18
Status: approved (Henry: "drive this till completion")
Repo: `~/me/fun/jev-zeta` (local-only fork; never merged upstream).

## Idea

Stock zeta compaction summarizes the old conversation range with a full
Claude call (`CompactionPolicy.summarize`, `src/zeta/core/context.py`).
Jev-triage compaction inserts a cheap classification stage BEFORE that:
one Jev systemone call marks each old item keep/drop; unimportant tool
results are tombstoned in place. If triage alone recovers the budget, the
summarize call is skipped entirely — better-preserved context AND a saved
Claude call. Otherwise stock summarization runs on the smaller range.

## Algorithm (stage 0 inside the existing compaction path)

1. Trigger: unchanged — `ContextAssembler` decides compaction is needed and
   computes the compactible range exactly as today. Recent turns are outside
   the range and therefore never triaged.
2. Candidate set: tool-result entries in the compactible range whose content
   exceeds a size floor (default 200 chars; smaller results are not worth a
   tombstone). V1 triages TOOL RESULTS ONLY — user messages, assistant
   decisions, and tool_call entries are never dropped.
3. One Jev call (`providers/jev.py`, reuse the existing client; add a
   `triage(...)` helper):
   - state: `{"task": <latest user objective: the most recent user message
     text, truncated 500 chars>, "latest_assistant_text": <latest assistant
     text, truncated 300 chars>, "recent_tool_actions": [<last 3
     non-candidate tool outcomes>], "items": [{"id", "kind", "tool",
     "excerpt"}...]}`. Each recent action is one line, such as
     `write keyfacts.txt: ok`. Excerpt = first 200 chars of the result text.
   - questions: per item id, one Noul: "Will the details of item <id> be
     needed to finish the task, beyond what the excerpt already shows?"
   - Same auth/retry/error conventions as `route_step`.
4. Drop rule: keep-probability < 0.35 -> tombstone. Tombstone REPLACES the
   tool result's content blocks with one text block:
   `[dropped by jev-compaction: <tool> result, ~<n> tokens]` — the
   tool_call/tool_result pairing and entry structure survive untouched
   (provider message validity depends on this).
5. Recount assembled tokens. Under budget -> emit compaction events as usual
   and SKIP summarize. Still over -> run the existing summarize on the
   (post-tombstone) range, unchanged.
6. Fail-safe: any Jev error / missing JEV_API_KEY / empty candidate set ->
   stage 0 is a no-op and stock compaction proceeds. Never wedge, never
   double-charge (one triage attempt per compaction).
7. Persistence: tombstoning mutates what the PROVIDER sees. Follow zeta's
   existing durable-store conventions for compaction (the store already
   records compaction entries): record dropped item ids + token estimates in
   the compaction entry metadata so a session transcript shows what was
   dropped. Do not destroy the durable original entries if the store design
   keeps them; mirror how summarize treats originals.

## Config

Default ON in the fork. `--no-jev-compaction` CLI flag (BooleanOptionalAction
like `--router`) + matching setting. Composes with `--no-router` freely.
Observability: emit a `compaction_start`-adjacent event or data field with
`{"jev_triage": {"candidates": n, "dropped": n, "tokens_recovered": n,
"skipped_summarize": bool}}` so headless streams show triage behavior; Jev
usage from triage is surfaced the same way route usage is (JEV-20's
mechanism).

## Testing

Python only, no network, no cargo/GUI. Mock Jev HTTP + fake backend. Cover:
triage request shape (ids, excerpts, task truncation); size-floor filtering;
tombstone replacement preserves pairing + entry structure; budget-recovered
skip-summarize path; still-over -> summarize on smaller range; Jev-error and
missing-key fallbacks (stock behavior byte-identical); recent-range
protection (nothing outside the compactible range is touched); flag/setting
plumbing; store metadata recording. Targeted test files only.

## Live demo (orchestrator step, after review)

Headless session with a small `--token-budget` forcing compaction mid-task:
verify triage fires, unimportant results tombstone, the task still completes
correctly, and the event stream reports triage stats.

## Out of scope

Summarize-prompt changes, mid-range user/assistant message triage, Score-
based multi-level importance (Noul v1 only), GUI.
