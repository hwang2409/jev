# Jev-Navigated Browsing: Arc 3 Implementation Plan

Date: 2026-09-25  
Status: implementation plan for review  
Scope: one experimental, Playwright-backed browsing arc in `harness/`

This plan turns the merged arc-3 design into dependency-ordered, PR-sized
lanes. The stable browser tool surface stays separate from the live element
catalog. Jev receives bounded page state as neutral data, and every action
passes identity, page-state, and safety checks before execution.

The normal lanes use fake browser adapters and mocked Jev responses. They do
not call a real site, the Gateway, or a Playwright browser binary. The final
lane alone runs the gated headless smoke against a local disposable fixture.

## Scope

In scope:

- one browser session and one page per agent session;
- `browser_navigate`, `browser_state`, `browser_click`, `browser_type`,
  `browser_select`, `browser_extract`, and `browser_submit`;
- bounded element catalogs, cheap local pre-filtering, and Jev element choice;
- page-state Nouls, stale-snapshot recovery, search-result triage, and
  injection-resistant state handling;
- shared safety-tier reuse for risky submits, clicks, and external navigation;
- offline evaluation against a stock large-tool baseline;
- one explicitly enabled, headless-only live smoke lane.

Out of scope:

- tabs, uploads, user-facing downloads, arbitrary JavaScript, extensions,
  authentication management, CAPTCHA solving, cookie export, screenshot
  control, and a general browser API;
- external smoke sites, personal credentials, production data, or real
  payment and download flows;
- a second risk policy, a second Jev transport, or page-specific tool schemas;
- making browsing default-on while this arc remains experimental.

## Locked decisions

Each decision is locked with the stated veto window: **locked (veto window:
this PR review)**.

1. **Feature flag:** Browsing is flag-gated, not default-on. The named setting
   is `browser_enabled`, with a matching `--browser` flag. The default is
   `false` until the arc leaves the experimental phase. **locked (veto window: this PR review).**
2. **Smoke site:** The smoke uses a disposable fixture site served on localhost
   from test assets. It has deterministic navigation, search or filtering, a
   harmless form submit, one deliberate external link, and hostile-text
   fixtures. It uses no external site and no personal credentials. **locked (veto window: this PR review).**
3. **Run mode:** Headless is the only supported run mode. A headed run is
   debug-only and manual. The final lane does not make headed mode a CI path.
   **locked (veto window: this PR review).**
4. **Budgets:** Use named settings with these proposed defaults:

   | setting | default | rationale |
   | --- | ---: | --- |
   | `browser_page_jev_call_budget` | `8` calls | Covers state, selection, gates, and two bounded recoveries on one page. |
   | `browser_page_jev_token_budget` | `12,000` tokens | Bounds catalog and gate input while allowing 24 filtered entries across the page budget. |
   | `browser_task_action_budget` | `20` actions | Covers navigation, search or filter, form entry, submit, and recovery without allowing loops. |
   | `browser_task_wall_clock_seconds` | `120` seconds | Gives a headless local task bounded startup and action time without an open-ended wait. |

   Exhaustion stops the task with a structured teaching error. The handler
   fails closed and never continues silently. **locked (veto window: this PR review).**
5. **Origin policy:** Use an allowlist. The smoke allowlist contains only the
   ephemeral localhost fixture origin. Any external-origin navigation requires
   approval in this arc, even when the user explicitly requested it. **locked (veto window: this PR review).**

## Package layout and module limits

The module-limit test is policy, not a lane target. Every lane must keep
`MAX_FILE_LINES=1250` and `MAX_FILES_PER_DIRECTORY=17` green. The `tui`
directory is already at its 17-file cap, so no lane adds a file there.

Reserve the direct `harness/src/zeta/tools/browser/` layout before lane 1. The
planned final direct-file count is 11, below the cap:

```text
browser/
  __init__.py       public registration and stable exports
  adapter.py        BrowserAdapter protocol and plain observation types
  fake.py           deterministic FakeBrowserAdapter
  playwright.py     optional Playwright implementation only
  catalog.py        snapshot catalog builder and bounded serialization
  prefilter.py      deterministic candidate filtering
  session.py        lazy session, snapshot identity, and cleanup
  gates.py          page-state gate decisions and recovery policy
  triage.py         search-result records, scoring, and tie policy
  handlers.py       browser tool handlers and structured errors
  schemas.py        stable tool schemas and shared payload builders
```

Tests stay under `browser/tests/` and are split by responsibility. The target
set is `test_adapter.py`, `test_session.py`, `test_catalog.py`,
`test_prefilter.py`, `test_provider.py`, `test_tools.py`, `test_gates.py`, and
`test_triage.py`. Injection probes stay in the existing top-level
`harness/tests/test_browser_injection.py` because they cross provider,
handler, and safety boundaries.

If an implementation would exceed 1,250 lines, split the responsibility into
one of these reserved files before adding more behavior. Do not raise either
limit. Each lane's exit gate includes `harness/tests/test_module_limits.py`.

## Shared contracts

These contracts apply to every lane:

- Browser framework objects stay behind `BrowserAdapter`. Structured results
  contain plain dataclasses or dictionaries only.
- A catalog entry carries adapter-observed role, affordance, destination,
  form context, and risk facts. Visible page text is data, never policy.
- `snapshot_id` and a generation counter identify the page state. Navigation,
  reload, or frame replacement invalidates prior element ids.
- Stale, detached, ambiguous, timeout, navigation-race, provider, and safety
  errors have stable kinds, bounded text, and machine-readable hints.
- Identity and safety errors fail closed. A routing error can request fresh
  state, but it cannot guess an element or silently retry a risky action.
- The browser surface remains static. The live element catalog travels in
  structured state for Jev and does not become a changing provider tool list.
- Normal lanes run offline with fake adapters and mocked Jev transport.

## Dependency-ordered lane ladder

Every lane is one focused PR. A lane may add a small follow-up test to an
earlier family when integration exposes a missing contract, but it may not
skip its stated non-goals or pull live dependencies into normal tests.

### Lane 1: adapter seam and fake adapter

Depends on: none.

Scope:

- Define the narrow `BrowserAdapter` protocol for launch, navigation,
  observation, click, type, select, extraction, search extraction, and close.
- Define plain observation values, snapshot limits, element references, action
  results, extraction results, and stable adapter error types.
- Add `FakeBrowserAdapter` with scripted observations and controls for detach,
  timeout, navigation race, launch failure, and cleanup failure.
- Keep Playwright imports and locator conversion in `playwright.py`, behind the
  protocol. Do not expose page objects or locators in results.

Files touched:

- `harness/src/zeta/tools/browser/adapter.py`
- `harness/src/zeta/tools/browser/fake.py`
- `harness/src/zeta/tools/browser/playwright.py` only for the adapter seam
- `harness/src/zeta/tools/browser/tests/test_adapter.py`
- `harness/pyproject.toml` only if the optional browser extra needs the seam

Exit criteria:

- Fake and real adapters share the same action contract.
- The fake can produce every offline failure condition without network or a
  Playwright binary.
- Adapter values have bounded fields and no browser framework objects.
- `test_adapter.py` and `test_module_limits.py::test_module_limits` pass.

Targeted tests:

- `harness/src/zeta/tools/browser/tests/test_adapter.py`
- adapter construction, action recording, bounds, cleanup, timeout, and race
  cases from the design's fake-adapter family.

Non-goals:

- Tool registration, Jev calls, page-state gates, safety policy, and live
  smoke execution.
- A complete DOM catalog or a real browser task.

### Lane 2: session lifecycle

Depends on: lane 1.

Scope:

- Add lazy `BrowserSession` ownership for one adapter, context, page, and
  current state.
- Start on first browser use and close from registry or loop cleanup.
- Make launch single-flight and cleanup best effort without hiding the original
  failure.
- Track snapshot ids, generation changes, recent actions, action and
  navigation timeouts, and the current element-reference map.
- Reject use after close and reject an element from an old snapshot.

Files touched:

- `harness/src/zeta/tools/browser/session.py`
- `harness/src/zeta/tools/registry.py`
- `harness/src/zeta/tools/browser/tests/test_session.py`
- `harness/tests/test_session_lifecycle.py`
- `harness/tests/test_session_shutdown.py`

Exit criteria:

- One registry owns one browser session for one agent session.
- Start and close are idempotent under success, cancellation, and launch
  failure.
- Snapshot and generation checks prevent stale ids from reaching the adapter.
- The two session integration tests and the module-limit test pass.

Targeted tests:

- `browser/tests/test_session.py`
- `harness/tests/test_session_lifecycle.py`
- `harness/tests/test_session_shutdown.py`
- stale-id and navigation-race cases from the fake-adapter family.

Non-goals:

- Candidate filtering, Jev selection, risk scoring, or loop routing.
- Multiple pages, tabs, shared contexts, or authentication state.

### Lane 3: catalog builder and cheap pre-filter

Depends on: lane 2.

Scope:

- Build bounded entries for links, buttons, textboxes, comboboxes, checkboxes,
  radios, tabs, headings, articles, and submit controls.
- Normalize accessible names and text. Preserve landmark context and adapter
  facts. Exclude unsupported, hidden, disabled, duplicate, and un-actionable
  entries as specified.
- Add deterministic lexical and role filtering using the goal and action.
- Enforce `ELEMENT_PREFILTER_K=40` and `ELEMENT_CATALOG_MAX=24` as named
  settings, byte caps, diversity quotas, prior-goal retention, and explicit
  `no_candidate` results.
- Cover 100, 500, and 2,000 actionable-element fixtures.

Files touched:

- `harness/src/zeta/tools/browser/catalog.py`
- `harness/src/zeta/tools/browser/prefilter.py`
- `harness/src/zeta/routing.py` for named browser catalog settings
- `harness/src/zeta/tools/browser/tests/test_catalog.py`
- `harness/src/zeta/tools/browser/tests/test_prefilter.py`

Exit criteria:

- Catalog output is deterministic, bounded, and free of selectors, XPath,
  hidden HTML, cookies, scripts, or policy claims.
- Snapshot ids are monotonic. Generation changes invalidate prior ids.
- Prefilter output is capped, role-aware, diverse on ties, and never calls Jev
  when it has no candidates.
- Large-page tests stay within time and byte bounds.
- Catalog, pre-filter, and module-limit tests pass.

Targeted tests:

- `browser/tests/test_catalog.py`
- `browser/tests/test_prefilter.py`
- catalog and routing fixture family from design section 9.1.

Non-goals:

- Provider request construction, page-state Nouls, search-result ranking, or
  external-origin policy.

### Lane 4: provider element-choice call shape

Depends on: lane 3.

Scope:

- Add the browser element Choice call to `providers/jev.py`.
- Send named neutral state fields: goal, current origin and path, recent
  actions, title, bounded page summary, and filtered catalog.
- Add `goal_element_present`, `page_loaded_and_stable`, and
  `action_is_the_next_step` Nouls to the same call.
- Parse selected id, affordance, probabilities, Nouls, usage, and confidence.
- Use the candidate `0.8` top-1 threshold and expose up to three candidates
  below it. Keep this threshold separate from page-state, safety, and triage
  thresholds.
- Preserve provider retry, malformed-response, missing-key, and usage rules.

Files touched:

- `harness/src/zeta/providers/jev.py`
- `harness/src/zeta/protocol/jev.py` or the existing browser result module if
  the result type needs a dedicated seam
- `harness/tests/test_jev_provider.py`
- `harness/src/zeta/tools/browser/tests/test_provider.py`

Exit criteria:

- The request has stable Choice criteria and separate structured Noul criteria.
- State fields are quoted as data and page text cannot alter instructions,
  policy, thresholds, or tool schemas.
- Confidence uses the existing least-confident judgment convention.
- Low confidence exposes candidates and performs no action.
- Provider tests and module limits pass without network access.

Targeted tests:

- mocked Jev request-shape, response parsing, usage, retry, malformed response,
  missing-key, confidence, and threshold tests.
- Choice, page-state, risk, and relevance fixture construction tests that are
  owned by later lanes and remain offline.

Non-goals:

- Calling the handler, executing browser actions, wiring the registry, or
  implementing the shared safety-tier decision.

### Lane 5: tools, registry, and loop integration

Depends on: lanes 2, 3, and 4.

Scope:

- Register the seven stable browser tools with stable schemas.
- Wire lazy session construction, catalog publication, and cleanup through
  `ToolRegistry` and the loop.
- Validate the selected id, snapshot, role, and affordance against the exact
  current catalog before any adapter call.
- Add static browser tool routing to the existing router-v2 invoke surface.
- Store current catalog and snapshot context in the router context. Reject an
  unrouted browser element with a structured teaching error.
- Return bounded state, extraction, success, top-3, and recovery results.

Files touched:

- `harness/src/zeta/tools/browser/__init__.py`
- `harness/src/zeta/tools/browser/handlers.py`
- `harness/src/zeta/tools/browser/schemas.py`
- `harness/src/zeta/tools/registry.py`
- `harness/src/zeta/runtime/loop/__init__.py`
- `harness/src/zeta/runtime/loop/routing.py`
- `harness/src/zeta/tools/route/__init__.py`
- `harness/src/zeta/tools/browser/tests/test_tools.py`
- `harness/tests/test_tools_integration.py`
- `harness/tests/test_router_auto_integration.py`

Exit criteria:

- The seven tool names and schemas remain stable across page changes.
- `browser_state` is the recovery tool after navigation, changing actions,
  stale ids, and unclear page progress.
- Stale, mismatched, detached, and unrouted element identities never execute.
- Session close is wired through the normal registry and loop cleanup paths.
- Tool, loop, route, and module-limit tests pass offline.

Targeted tests:

- stable schema, lazy start, state recovery, extraction bounds, stale identity,
  top-3 expansion, unrouted rejection, batching, and close tests.
- design section 9.4 tool and loop family.

Non-goals:

- Page-state Noul decisions, safety approval, search relevance acceptance,
  injection corpus expansion, live smoke, or an external site.

### Lane 6: page-state Noul gates

Depends on: lanes 4 and 5.

Scope:

- Implement the six gates: `page_loaded_and_stable`, `goal_element_present`,
  `action_is_the_next_step`, `action_succeeded`, `dead_end`, and
  `needs_different_approach`.
- Combine deterministic adapter evidence with bounded Jev judgments.
- Block on negative load or goal presence. Treat uncertain action success as
  unknown and request a fresh state without claiming reversal.
- Bound recovery attempts and return stable teaching errors at the cap.
- Keep provider failure polarity conservative for each active gate.

Files touched:

- `harness/src/zeta/tools/browser/gates.py`
- `harness/src/zeta/providers/jev_browser.py`
- `harness/src/zeta/providers/jev.py` only for shared result plumbing
- `harness/src/zeta/tools/browser/handlers.py`
- `harness/src/zeta/tools/browser/tests/test_gates.py`
- `harness/src/zeta/tools/browser/tests/test_tools.py`

Exit criteria:

- Each gate has a named criterion, threshold, safe direction, and structured
  failure result.
- Negative load and goal-presence judgments block action execution.
- Uncertain post-action success requests state and never reports false success.
- Provider errors use the gate's safe direction and never bypass identity or
  safety checks.
- Gate, handler, provider, and module-limit tests pass offline.

Targeted tests:

- all page-state Noul tests from design section 9.2 and the full gate family in
  `browser/tests/test_gates.py`.
- recovery-cap and changed-versus-unchanged-state cases.

Non-goals:

- Risk classification, external navigation approval, search-result scoring,
  or live browser startup.

### Lane 7: shared safety-tier wiring

Depends on: lane 5. It may use lane 6 gate results but does not redefine them.

Scope:

- Reuse the safety-tier shape from
  `2026-09-21-jev-safety-tier-design.md` for risky browser actions.
- Classify submits and risky clicks from adapter facts: role, text, current
  origin, destination, form action origin, payment or auth language, download,
  and durable-state change.
- Require the gate before `browser_submit` and before classifier-marked risky
  clicks. Classify external-origin navigation as risky.
- Keep layer 0 deterministic. Send only analyzable actions to Jev safety
  scoring. Fail closed on errors, missing keys, malformed responses, and low
  confidence: escalate in interactive mode and deny with a teaching error in
  headless mode.
- Add the `browser_enabled` flag and named budget settings without enabling
  browsing by default.

Files touched:

- `harness/src/zeta/core/safety/_browser.py`
- `harness/src/zeta/core/safety/_tier.py`
- `harness/src/zeta/core/safety/_types.py`
- `harness/src/zeta/core/safety/__init__.py`
- `harness/src/zeta/tools/browser/handlers.py`
- `harness/src/zeta/providers/jev.py`
- CLI and settings modules that own `browser_enabled` and budget values
- `harness/tests/test_safety.py`
- `harness/tests/test_safety_eval.py`
- `harness/src/zeta/tools/browser/tests/test_tools.py`

Exit criteria:

- Browser actions use the shared safety policy and no parallel browser policy.
- Page labels such as `safe` never bypass layer 0 or the safety tier.
- External-origin navigation requires approval, including explicit user
  requests.
- Budget exhaustion stops with a stable teaching error and no silent retry.
- Safety decision, fail-closed, headless, interactive, flag-off, and module
  limit tests pass offline.

Targeted tests:

- risk-class matrix, score and confidence matrix, approval handoff, denial,
  missing-key, malformed-response, and flag-off byte-identity tests.
- shared safety-tier tests from section 9.2 and tool handoff tests.

Non-goals:

- Changing shell safety policy, adding new safety levels, approving external
  origins, or making the browser flag default-on.

### Lane 8: search-result triage

Depends on: lanes 3 through 6.

Scope:

- Extract bounded search records with result id, title, snippet, displayed URL,
  source section, and position.
- Add Jev Score relevance ranking over records as neutral data.
- Apply separate relevance threshold, tie margin, relevance floor, call
  confidence, top-3 exposure, and source-diversity settings.
- Use relevance only as a pre-filter feature. Never auto-click from score alone.
- Send an external result through the lane-7 safety policy.

Files touched:

- `harness/src/zeta/tools/browser/triage.py`
- `harness/src/zeta/tools/browser/catalog.py` only for shared result records
- `harness/src/zeta/providers/jev.py`
- `harness/src/zeta/tools/browser/handlers.py`
- `harness/src/zeta/tools/browser/tests/test_triage.py`
- `harness/src/zeta/tools/browser/tests/test_tools.py`

Exit criteria:

- Clear winners are accepted only above the relevance threshold and tie margin.
- Close ties, low confidence, and below-floor results expose candidates or
  request a narrower goal.
- Equal scores preserve source diversity.
- The handler never treats page result text as an instruction.
- Triage, handler, safety handoff, and module-limit tests pass offline.

Targeted tests:

- clear winner, tie, low score, sponsored result, pagination, source diversity,
  threshold boundary, and external-result tests from section 9.1 and 9.2.

Non-goals:

- General search-provider integration, external-origin approval changes,
  arbitrary result scraping, or live search traffic.

### Lane 9: injection-probe suite

Depends on: lanes 5 through 8.

Scope:

- Add paired benign and hostile fixtures at the real boundaries.
- Cover buttons, links, snippets, input values, hidden accessible names, URLs,
  search results, extraction output, and tool-result carryover.
- Assert hostile text remains named state data and never changes criteria,
  policy, origin allowlists, thresholds, tool schemas, or approval behavior.
- Assert actual adapter role, destination, form facts, and risk facts win over
  page claims such as `safe`, `approved`, `urgent`, or instruction text.
- Keep fail-closed identity, provider, and safety behavior mutation-proven.

Files touched:

- `harness/tests/test_browser_injection.py`
- `harness/src/zeta/tools/browser/handlers.py`
- `harness/src/zeta/tools/browser/catalog.py`
- `harness/src/zeta/providers/jev.py`
- `harness/src/zeta/core/safety/_browser.py`
- `harness/src/zeta/tools/browser/tests/test_tools.py`

Exit criteria:

- Every injection probe fails if page content is promoted to instruction data.
- Hostile content cannot create tools, change origins, grant approval, or
  select an element outside the current catalog.
- Extracted text stays bounded and inert on the next provider turn.
- The full injection suite, targeted browser tests, and module limits pass.

Targeted tests:

- all probes in `harness/tests/test_browser_injection.py`.
- injection cases from design section 9.1 through 9.4, including transport
  capture and message-assembly boundaries.

Non-goals:

- Claiming perfect model resistance, adding a model-based security scanner, or
  changing the shared shell safety policy.

### Lane 10: offline evaluation harness

Depends on: lanes 3 through 9.

Scope:

- Add an offline browser task corpus and runner that compares the stable routed
  arm with an equivalent stock large-tool baseline.
- Use the same fake adapter, page fixtures, task prompts, timeouts, and safety
  policy in both arms.
- Cover catalog sizes 10, 40, 120, 500, and 2,000; static, moderate churn,
  and full churn; clear targets, repeated labels, search triage, and forms.
- Report top-1 accuracy, top-3 coverage, page-state gate accuracy, task
  success, risky false approvals, cost and tokens per step, provider turns,
  stale recovery, retries, and time per successful step.
- Report pre-filter recall and separate routing misses from adapter or page
  failures. Record threshold versions and the expected small-catalog outcome.

Files touched:

- `harness/evals/browser_tasks.jsonl`
- `harness/evals/browser_eval.py`
- `harness/evals/RESULTS.md` only for offline baseline results
- `harness/tests/test_browser_evals.py`
- `harness/tests/test_evals.py` only for shared runner plumbing

Exit criteria:

- The eval runs without network, a Gateway key, or a Playwright binary.
- Routed and stock arms use equivalent fixtures and policy.
- Reports include all primary metrics and confidence calibration by primitive.
- The report distinguishes a pre-filter miss from a Jev selection miss.
- Eval tests, targeted browser tests, and module limits pass.

Targeted tests:

- offline task loading, fixture generation, metric calculation, cost accounting,
  safety parity, catalog-size matrix, churn matrix, and report schema tests.
- design section 10's queued big-catalog crossover matrix.

Non-goals:

- Live Gateway calls, threshold auto-tuning, production traffic, or declaring
  a routing win from token reduction alone.

### Lane 11: gated live smoke

Depends on: lanes 1 through 10.

Scope:

- Serve a disposable local fixture from test assets on an ephemeral localhost
  port. Include deterministic navigation, search or filter, harmless submit,
  deliberate external link, hostile text, stale state, and low-confidence
  controls.
- Run only when the explicit smoke gate is set. Require network access, a
  Playwright browser binary, and a Vercel AI Gateway key.
- Run headless only. Keep headed mode as a manual debug option outside the
  supported smoke contract.
- Verify navigate -> state -> select/type -> submit, stale-id recovery,
  low-confidence top-3 exposure, and risky external or submit approval.
- Enforce the localhost allowlist, per-page Jev call and token budgets, task
  action cap, wall-clock budget, cleanup, and no-secret logging.

Files touched:

- `harness/tests/browser_fixture/index.html`
- `harness/tests/browser_fixture/fixture_server.py`
- `harness/tests/browser_fixture/assets/*`
- `harness/tools/browser_live_smoke.py`
- `harness/tests/test_browser_live_smoke.py`
- `harness/pyproject.toml` and `harness/uv.lock` for the optional Playwright
  extra and pinned browser test support

Exit criteria:

- The smoke refuses to run without its explicit flag and complete policy
  configuration.
- The only allowed smoke origin is the ephemeral localhost fixture origin.
- External-origin navigation reaches approval and cannot auto-proceed merely
  because the user asked for it.
- Budget exhaustion returns a structured teaching error and stops the task.
- Headless smoke passes with cleanup. No personal credentials, production data,
  real downloads, or secrets appear in logs.
- The smoke test and module-limit test pass. Headed mode is not a CI claim.

Targeted tests:

- gated live smoke only; all normal lane tests remain offline.
- fixture navigation, filter, harmless submit, external link, hostile text,
  stale snapshot, top-3, safety approval, budget, and cleanup checks.

Non-goals:

- External websites, headed support, auth, payments, real downloads, or a
  general browser compatibility matrix.

## Test-family ownership map

Every family from the merged design's section 9 has an owning lane:

| design section 9 family | owning lane | evidence |
| --- | --- | --- |
| catalog and routing fixtures | lane 3 | bounded catalogs, caps, roles, ids, diversity, and injection-neutral filtering |
| mocked Jev provider | lane 4, with gate additions in lane 6 | request shape, criteria, parsing, usage, retries, malformed responses, keys, and per-primitive thresholds |
| fake browser adapter | lane 1 | action recording, scripted observations, detach, timeout, race, and cleanup |
| tool and loop tests | lane 5, with safety and gate cases in lanes 6-8 | schemas, ownership, recovery, truncation, static routing, top-3, batch behavior, and unrouted rejection |
| gated live smoke | lane 11 | local fixture, headless end-to-end flow, stale recovery, top-3, approval, budgets, and cleanup |

The injection-probe suite in lane 9 is an additional cross-boundary family
required by section 7 and is included in the lane map even though it has no
separate numbered subsection in section 9.

## Failure polarity and recovery rules

The plan uses one polarity for safety and identity:

- stale id: return `stale_snapshot`; take no action;
- detached or ambiguous locator: return `element_unavailable`; take no action;
- navigation race: return `navigation_race`; observe before deciding;
- adapter timeout: return `browser_timeout`; allow one bounded observation;
- provider timeout or malformed result: return a routing error; do not guess;
- page-load failure: return `page_load_failed`; do not infer success;
- safety error or low confidence: escalate or deny per shared tier; never
  auto-retry the same risky action;
- extraction truncation: return success with `truncated=true` and request a
  narrower target when needed;
- any budget exhaustion: stop with a structured teaching error.

The router may use its existing fail-open behavior for general tool discovery
when that contract requires a usable session. Browser element actions never
fail open into an arbitrary element. Safety and identity always fail closed.

## Self-review

- The plan has eleven dependency-ordered, PR-sized lanes.
- Each lane names scope, files, exit criteria, targeted tests, and non-goals.
- All section 9 test families map to an owning lane.
- The browser package has a reserved 11-file direct layout, below the
  17-file directory cap. The 1,250-line file cap remains enforced.
- Normal lanes are offline-only. The final smoke alone needs network, the
  Playwright binary, and a Vercel AI Gateway key.
- The five open decisions are recorded as locked with the veto window.
- Budgets have named settings, concrete defaults, rationales, and a fail-closed
  exhaustion rule.
- The smoke uses a local disposable fixture, headless execution, localhost
  allowlisting, and no personal credentials.
- No lane adds tabs, arbitrary JavaScript, authentication management, or a
  second safety policy.
- Failure handling is explicit and consistent for safety, identity, routing,
  timeout, and extraction errors.
- The document has no unfinished placeholders.
