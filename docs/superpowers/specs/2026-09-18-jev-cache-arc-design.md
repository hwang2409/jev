# jev-zeta: in-turn history caching design (cache-reads arc)

Date: 2026-09-18
Status: approved (Henry: "go on an arc to improve the cache reads")
Repo: `~/me/fun/jev/harness`. Motivated by the three-way eval diagnosis
(`evals/RESULTS.md` findings 2-3): agentic tasks are single user turns, so
zeta's ZETA-39 marker policy ("cache only completed history before the
active user turn") never caches history in ANY mode, and auto's static
prefix alone sits under Anthropic's ~1024-token cacheable minimum.

## Change

In `anthropic_payload.py`, advance the message-level `cache_control` marker
INTO the active user turn: place it on the last COMPLETED message excluding
the newest message (the newest may still be revised for schema delivery —
the store's newest-only revision guard, added in JEV-29's hardening, is the
safety property that makes every earlier message immutable once sent).

Rules:
1. Breakpoints stay <= 4 per request (system[-1], tools[-1], one advancing
   message marker — unchanged count, new placement).
2. The marker never lands on or after the newest active message.
3. If no completed message exists yet (first call of a session), fall back
   to current behavior (system/tools markers only).
4. Compaction rewrites earlier history (tombstones, summary markers): the
   post-compaction request naturally re-creates cache (one-time
   cache_creation spike). No special handling; document it.
5. Codex path: no change (automatic prefix caching server-side; prefix
   stability already verified by the router-v2 probes).
6. TTL stays default ephemeral (5 minutes); note as a tunable only.

This helps ALL modes (stock, tool, auto) — it is a general zeta improvement
and an upstream candidate; fork-only for now (never merge upstream without
Henry's explicit say).

## Testing (offline, mocked)

- Marker placement: multi-call single-user-turn run — marker advances each
  call to the last completed message, never the newest; breakpoint count
  <= 4; first-call fallback.
- Revision interplay: after marking, a revision attempt on a marked
  (non-newest) message is rejected by the store guard (regression pairing).
- Compaction interplay: post-compaction request serializes markers sanely
  (no marker on a removed entry).
- Prefix stability: the router-v2 annotation-stripped prefix-extension
  probes still pass; additionally assert the marked span is byte-stable
  across calls (the content under the marker never changes).
- No behavior change to message content, routing, or compaction.

## Acceptance (live, orchestrator step)

Rerun the three-way eval (6 tasks x auto/tool/stock). Require:
cache_creation > 0 and cache_read > 0 in EVERY mode; auto's effective cost
drops materially vs the 2026-09-18 baseline; quality metrics unchanged.
Update `evals/RESULTS.md` with the redrawn cost table.

## Out of scope

TTL tuning, codex-side changes, upstreaming, compaction-aware cache
preservation (accept the one-time invalidation).
