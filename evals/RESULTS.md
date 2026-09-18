# Router vs stock — live eval results (2026-09-18)

Head `ba0564d7`, 6 tasks x 2 modes, claude-sonnet-4-6 via subscription OAuth,
jev-1.13.0 routing. Full records: `results/20260918T142954577340Z.json`.

| Metric | Router | Stock |
|---|---|---|
| Tasks passing all checks | 4/6 | 5/6 |
| Raw tokens (Claude in/out + Jev) | 58.2k/6.4k + 18.2k Jev | 21.9k/2.9k |
| Cache reads | 11.6k | 98.5k |
| Est. cost (sonnet pricing, cache-adjusted) | ~$0.27 + Jev | ~$0.14 |
| Wall clock (total) | 89s | 68s |

## Findings

1. Dynamic per-turn tool injection DEFEATS prompt caching: the tools list
   changes every turn, invalidating the prefix. Stock's static 10-tool
   prompt cached 98.5k tokens at 0.1x; the router paid fresh input on
   almost every turn. At this catalog size the router is ~2x the dollar
   cost despite 38 percent fewer raw combined tokens.
2. Route round-trips consume the turn budget: 3 router runs finished their
   work (checks green) but ran out of turns before a final message.
3. One genuine routing-quality miss: in-place-edit routed to bash (the
   agent phrased its step shell-style) and botched the edit; stock used
   read+edit cleanly.
4. chain-and-verify failed strict checks in BOTH modes (formatting drift) —
   an eval-strictness issue, not a router issue. Manual audit of both event
   streams: no checker gaming; the router version executed the full chain
   routed (read manifest -> reads -> writes) but skipped the final verify
   re-read.
5. Router mechanics were flawless: 0 unrouted attempts, 0 router errors,
   12 runs clean.

## Implications for the one-tool harness thesis

The thesis holds only where catalog schema bloat exceeds cache savings:
big catalogs (phase 2: 120+ tools), cache-hostile contexts, or providers
without prefix caching. At ~10 cached tools, stock wins on cost and turn
efficiency. Design leads if pursued further: cache-stable injection
(append-only tool exposure; or deliver routed schemas via user-message
content instead of the tools param) and a turn-budget-aware route protocol.
