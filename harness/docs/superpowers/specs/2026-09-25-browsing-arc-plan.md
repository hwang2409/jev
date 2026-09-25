# Jev-navigated browsing: arc 3 gap plan

Date: 2026-09-25
Status: audit-first implementation plan for review
Scope: close the remaining gaps around the merged experimental browser surface

The design spec remains authoritative:
`harness/docs/superpowers/specs/2026-09-22-jev-browsing-arc-design.md`.
The safety contract remains authoritative for shared risk decisions:
`harness/docs/superpowers/specs/2026-09-21-jev-safety-tier-design.md`.

This is a gap plan. It does not rebuild the browser adapter, catalog, provider,
page-state gates, safety handoff, or router surface already present on the
branch. The audit below maps the design to current evidence. The lane ladder
contains only partial or missing rows.

## audit baseline

The audit used the branch files, not the earlier plan's inventory. Current
browser code has five direct Python files. The largest files are
`browser/__init__.py` at 1,045 lines and `browser/adapter.py` at 949 lines.
Both remain below `MAX_FILE_LINES=1250`. The directory remains below
`MAX_FILES_PER_DIRECTORY=17`.

### design surface audit

| design surface | status | current evidence | disposition |
| --- | --- | --- | --- |
| seven stable browser tools and stable schemas | implemented | `harness/src/zeta/tools/browser/__init__.py:55-70,627-710` registers `browser_navigate`, `browser_state`, `browser_click`, `browser_type`, `browser_select`, `browser_extract`, and `browser_submit`; `harness/src/zeta/tools/route/__init__.py:156` adds their static criteria | retain; lane 1 gates registration |
| one session per agent, lazy adapter launch, close and clone handling | implemented | `harness/src/zeta/tools/browser/session.py:40-250` owns one adapter, state, refs, generation, and cleanup; `harness/src/zeta/tools/registry.py:249-258,427-440` owns lazy access and close; `registry.py:345-380` resets the session on clone | retain; test the gated path in lane 1 |
| framework-neutral adapter seam | implemented | `harness/src/zeta/tools/browser/adapter.py:18-142` defines plain values and `BrowserAdapter`; `adapter.py:284-518` contains the Playwright implementation; `adapter.py:703-813` contains the deterministic fake | retain; do not split or re-plan these modules |
| bounded page observations and element references | implemented | `harness/src/zeta/tools/browser/adapter.py:18-83` defines byte-bounded observations and risk facts; `adapter.py:177-282` extracts plain DOM data | retain; add only policy limits in lane 2 |
| snapshot-local identity and generation invalidation | implemented | `harness/src/zeta/tools/browser/session.py:167-213,234-249` rejects stale or mismatched ids; `harness/src/zeta/tools/browser/catalog.py:160-219` builds and checks snapshots | retain; existing tests remain regression gates |
| catalog roles, affordances, text bounds, and byte bounds | implemented | `harness/src/zeta/tools/browser/catalog.py:24-55,160-480` builds normalized entries and bounded payloads; `catalog.py:253-480` enforces byte fitting | retain |
| cheap pre-filter and no-candidate behavior | implemented | `harness/src/zeta/tools/browser/catalog.py:457-637` defines `ELEMENT_PREFILTER_K=40`, `ELEMENT_CATALOG_MAX=24`, role and lexical scoring, diversity, prior-id retention, and empty results | retain |
| Jev element Choice and least-confidence result | partial | `harness/src/zeta/providers/jev.py:492-702` builds, parses, retries, and confidence-gates element choice; `harness/src/zeta/routing.py:7-12` has the top-1 and top-3 constants, but named-setting configurability and threshold-version telemetry are missing | add named settings, versioned threshold telemetry, and threshold-isolation tests in lane 2 |
| six page-state Nouls and conservative recovery | implemented | `harness/src/zeta/providers/jev_browser.py:14-325` defines and parses all six gates; `harness/src/zeta/tools/browser/gates.py:25-212` applies safe directions and the recovery cap | retain |
| shared safety-tier handoff for risky browser actions | partial | `harness/src/zeta/core/safety/_browser.py:43-83` classifies browser evidence; `harness/src/zeta/core/safety/_tier.py:86-128` evaluates it through the shared tier; `browser/__init__.py:316-339,798-838` calls the tier | add explicit origin-allowlist enforcement and budget tests in lane 2 |
| origin allowlist and approval-gated external navigation | partial | `browser/__init__.py:745-751` only parses an origin; `adapter.py:625-636` accepts any absolute HTTP or HTTPS URL; `_browser.py:57-68` detects cross-origin evidence but has no allowlist; `runtime/composition.py:141-150` creates `SafetyTier` only when safety-tier and yolo are both enabled, so `_check_browser_safety()` denies instead of asking in a normal interactive run | add one shared policy in lane 2, wire browser safety independently of shell yolo, and test the flag/yolo/safety-tier/headless matrix |
| named browser budgets | partial | `session.py:24-25` has only navigation and action timeouts; `adapter.py:18-24` has byte caps; no page Jev call, Jev token, task action, or task wall-clock budget exists | add named settings, accounting, and fail-closed exhaustion in lane 2 |
| default-off experimental flag | missing | `harness/src/zeta/tools/_discovery.py:14-50` imports browser and calls `register`; `browser/__init__.py:627-710` registers unconditionally; `harness/src/zeta/config/settings.py:50-68,85-129` has no browser setting; `harness/src/zeta/cli/main.py:54-100` has no browser flag | make lane 1 the registration-time gate |
| prompt-injection boundary rules | implemented | `harness/src/zeta/providers/jev.py:492-607` names state fields and neutral criteria; `harness/tests/test_browser_injection.py:201-692` covers hostile fields, URLs, hidden names, extraction, tool results, safety, and static schemas | retain; rerun as a gate regression |
| stable structured errors and recovery hints | partial | `harness/src/zeta/tools/browser/__init__.py:1004-1037` maps known adapter and routing failures, but navigation failures fall through to `browser_start_failed`; design section 8 requires `page_load_failed` for page-load failures | add the page-load error-kind mapping and regression in lane 2 |
| search-result extraction and triage | partial | `adapter.py:75-92,454-482` returns bounded result records; `catalog.py:55-158` applies score, tie, floor, confidence, and source diversity; `routing.py` has constants only, without named-setting configurability or threshold-version telemetry; `test_tools.py:1055-1112` exercises handler results | add named settings, versioned threshold telemetry, and threshold-isolation tests in lane 2 |
| headless Playwright path | partial | `adapter.py:284-324` supports headless launch; `harness/tools/browser_live_smoke.py:125-190` runs handlers, but its policy values are unset at lines 21-24 and the URL comes from an environment variable at lines 27-32 | replace the external-site smoke with the locked local fixture in lane 3 |
| local fixture smoke site | missing | no `harness/tests/browser_fixture/` directory exists; `harness/tests/test_browser_live_smoke.py:8-16` only tests enablement and does not start a site | add the disposable localhost fixture and smoke assertions in lane 3 |
| offline routed-versus-stock browser evaluation | missing | `harness/evals/run_evals.py` covers existing router tasks; no browser task corpus or browser-specific runner exists | add the browser eval harness in lane 4 |

### design section 9 test-family audit

| design section 9 family | status | current evidence | remaining work |
| --- | --- | --- | --- |
| 9.1 catalog and routing fixtures | implemented | `harness/src/zeta/tools/browser/tests/test_catalog.py:125-387` covers bounds, hidden entries, generations, and large inputs; `test_prefilter.py:113-239` covers 100 and 500 element pages, diversity, and empty candidates; `harness/src/zeta/tools/route/tests/test_router_auto.py:164-403` covers static schemas and unrouted elements | keep as regression coverage; add flag-off schema assertions in lane 1 |
| 9.2 mocked Jev provider | partial | `harness/tests/test_jev_provider.py:451-960` covers neutral Choice, page-state, search scoring, malformed responses, missing keys, bounds, usage, and confidence; `test_browser_choice_delegates_retries_to_client` feeds one successful response and asserts one request, so it does not test retry behavior; `harness/tests/test_safety.py:63-316` covers safety decisions and failure polarity | add real retryable responses through the client seam, request-count and usage assertions, named budget accounting, and allowlist cases in lane 2 |
| 9.3 fake browser adapter | implemented | `harness/src/zeta/tools/browser/tests/test_adapter.py:64-465` covers action recording, bounds, failures, launch, cleanup, and contract parity; `test_session.py:46-78` covers launch failure and cancellation cleanup | keep as regression coverage |
| 9.4 tool and loop tests | partial | `harness/src/zeta/tools/browser/tests/test_tools.py:803-1112` covers schemas, lazy use, stale identity, cleanup, extraction, triage, and safety; `router_auto.py:238-403` covers loop routing and fail-open identity rejection, but no test maps the design's required browser batch behavior | add a concrete multi-call browser batch test, plus default-off and flag-on cases, in lane 1 |
| 9.5 gated live smoke | partial | `harness/tools/browser_live_smoke.py:42-79` has policy checks and a budget helper; `test_browser_live_smoke.py:8-16` checks only enablement | add local fixture, complete flow, stale recovery, low-confidence response, external approval, cleanup, and no-secret checks in lane 3 |

## locked decisions

These decisions are closed for this plan. Each is locked for this PR review.

1. Browsing uses one named `browser_enabled` setting and matching
   `--browser`/`--no-browser` flags. The default is `false`. Registration is
   gated before tool schemas, adapter factories, or sessions are exposed.
2. The live smoke uses a disposable fixture served on localhost. It includes
   deterministic navigation, search or filtering, a harmless form submit, a
   deliberate external link, stale state, low-confidence controls, and
   hostile text. It uses no external site or personal credential.
3. The supported smoke mode is headless. A headed run is a manual debug aid,
   not a CI path or acceptance mode.
4. The named budgets have these defaults:

   | setting | default | exhaustion behavior |
   | --- | ---: | --- |
   | `browser_page_jev_call_budget` | 8 calls | stop with `browser_budget_exhausted` |
   | `browser_page_jev_token_budget` | 12,000 tokens | stop with `browser_budget_exhausted` |
   | `browser_task_action_budget` | 20 actions | stop with `browser_budget_exhausted` |
   | `browser_task_wall_clock_seconds` | 120 seconds | stop with `browser_budget_exhausted` |

   The handler fails closed and never continues silently after exhaustion.
5. Origin handling uses an allowlist. The smoke allowlist contains only its
   ephemeral localhost origin. External-origin navigation always reaches the
   shared safety approval policy, even when the user requested it. No page
   label can grant approval.

## gap lane ladder

The current browser surface is default-on. Therefore lane 1 is the only first
lane. Every later lane depends on lane 1 and keeps the default-off gate. No
lane may widen the exposed surface before its gating and safety dependency.

### lane 1: gate the existing browser surface

Depends on: none.

Scope:

- Add `browser_enabled` to the validated settings and resolved runtime config.
- Add `--browser` and `--no-browser` with the same precedence rules as the
  other Boolean settings. Keep the built-in default false.
- Pass the resolved value into the registry or discovery context.
- Make `browser.register()` return without registering tools, criteria, an
  adapter factory, or a session when the flag is off.
- Keep flag-on registration byte-compatible with the existing seven schemas.
- Allow browser unit tests to construct an explicitly enabled registry.
- Add a browser loop test for a multi-call batch that preserves result order
  and applies routing, identity, and safety checks to each call.

Files touched:

- `harness/src/zeta/config/settings.py`
- `harness/src/zeta/cli/main.py`
- `harness/src/zeta/runtime/composition.py`
- `harness/src/zeta/runtime/tool_setup.py` if that path builds a separate registry
- `harness/src/zeta/tools/registry.py`
- `harness/src/zeta/tools/browser/__init__.py`
- `harness/src/zeta/tools/route/__init__.py` only if disabled discovery needs a guard
- `harness/tests/test_settings.py`
- `harness/tests/test_cli.py`
- `harness/tests/test_tool_discovery.py`
- `harness/src/zeta/tools/browser/tests/test_tools.py`
- `harness/src/zeta/tools/route/tests/test_router_auto.py`

Exit criteria:

- A default registry exposes no `browser_*` tools, schemas, or browser adapter.
- `--browser` exposes exactly the existing seven tools and stable schemas.
- `--no-browser` overrides settings and keeps the surface hidden.
- Flag-off discovery does not construct a browser session or Playwright object.
- Existing browser tests set the flag explicitly and retain their behavior.
- The module-limit test remains green.

Targeted test set:

- settings and CLI precedence tests;
- discovery default-off and explicit-enable tests;
- browser schema and lazy-session tests;
- browser loop batch-behavior test for multi-call dispatch and per-call results;
- router catalog tests for hidden and enabled browser tools;
- `harness/tests/test_module_limits.py::test_module_limits`.

Non-goals:

- Do not change browser handlers, catalog behavior, provider criteria, safety
  scoring, budgets, origin policy, or live smoke behavior.
- Do not rename any browser tool or alter its schema.

### lane 2: add named budgets and origin policy

Depends on: lane 1 and the existing shared safety tier.

Scope:

- Add the four locked browser budgets to settings, resolved config, the
  registry, and the browser session context.
- Count every browser page Jev call and provider-reported input and output
  token. Bound the task action count and wall clock with a monotonic clock.
- Apply the budgets to element choice, page-state gates, search triage, and
  recovery. Preserve existing navigation and action timeout settings.
- Promote element-selection and search-triage thresholds to named settings.
  Emit a threshold version with each decision and keep primitive thresholds
  separate.
- Return one stable `browser_budget_exhausted` teaching error. Do not retry a
  risky action after exhaustion.
- Add one origin policy that normalizes scheme, host, and effective port.
  Permit only configured origins without external approval. Classify every
  cross-origin target through the shared safety tier. Deny or escalate an
  origin that is not allowlisted; never let a page label change the result.
- Keep the policy in the existing browser safety path. Do not create a second
  browser risk policy.
- Wire the browser safety tier independently of shell yolo. Interactive
  external navigation must reach the existing approval prompt. Headless
  external navigation must deny with a teaching error. Test every combination
  of browser flag, yolo flag, safety-tier flag, and interactive or headless
  mode.
- Map adapter navigation failures to `page_load_failed` and add a regression
  test for the structured error kind.
- Drive provider retry tests with retryable responses followed by success
  through the client seam. Assert request count, final result, and usage.

Files touched:

- `harness/src/zeta/config/settings.py`
- `harness/src/zeta/cli/main.py` only for budget overrides if required by the
  settings contract
- `harness/src/zeta/runtime/composition.py`
- `harness/src/zeta/tools/registry.py`
- `harness/src/zeta/tools/browser/session.py`
- `harness/src/zeta/tools/browser/__init__.py`
- `harness/src/zeta/routing.py`
- `harness/src/zeta/providers/jev.py`
- `harness/src/zeta/tools/browser/catalog.py`
- `harness/src/zeta/core/safety/_browser.py`
- `harness/src/zeta/core/safety/_tier.py` only for shared origin outcome wiring
- `harness/src/zeta/tools/_results.py`
- `harness/tests/test_settings.py`
- `harness/tests/test_safety.py`
- `harness/tests/test_jev_provider.py`
- `harness/src/zeta/tools/browser/tests/test_gates.py`
- `harness/src/zeta/tools/browser/tests/test_tools.py`
- `harness/tests/test_browser_injection.py`

Exit criteria:

- All four names and defaults are visible in the resolved browser context.
- Call, token, action, and wall-clock exhaustion stop the task with the same
  structured error and safe recovery hint.
- Provider usage is charged once, including retries, with no negative or
  unbounded counter.
- Element-selection and search-triage thresholds are named settings, remain
  separate by primitive, and emit threshold-version telemetry.
- The localhost fixture origin can pass the allowlist.
- An external target cannot auto-proceed, including after an explicit user
  request. Browser safety reaches approval in interactive mode without shell
  yolo, and denies with a teaching error in headless mode. The full
  browser/yolo/safety-tier/headless matrix passes.
- Retry tests exercise real retryable responses through the client seam and
  assert the final result, request count, and usage.
- Navigation failures return `page_load_failed` with a regression test.
- Hostile page text cannot add an origin, raise a budget, or change a
  threshold.
- Existing shell safety behavior remains unchanged.
- The module-limit test remains green.

Targeted test set:

- settings defaults and CLI precedence;
- browser call and token accounting, action cap, and wall-clock cap;
- named threshold settings, threshold-version telemetry, and primitive
  threshold-isolation tests;
- origin normalization, localhost allowlist, external approval, and headless
  denial;
- browser/yolo/safety-tier/headless matrix tests for external navigation;
- provider retry tests with retryable responses through the client seam;
- navigation failure mapping and `page_load_failed` regression;
- missing-key, malformed, and low-confidence shared safety failures;
- injection probes for hostile URLs, labels, and extracted text;
- gate recovery at and below the budget cap;
- `harness/tests/test_module_limits.py::test_module_limits`.

Non-goals:

- Do not add a browser-specific safety scorer or new safety tier.
- Do not enable browsing by default.
- Do not add a live site, Playwright smoke fixture, or browser evaluation
  corpus.

### lane 3: replace the external smoke with a localhost fixture

Depends on: lanes 1 and 2.

Scope:

- Add a disposable fixture server under `harness/tests/browser_fixture/`.
- Serve deterministic pages with navigation, search or filtering, a harmless
  form, an external link, hostile page text, a stale-state transition, and a
  low-confidence control.
- Start the server on an ephemeral localhost port and pass only that origin
  to the browser origin policy.
- Update `harness/tools/browser_live_smoke.py` to require the explicit smoke
  gate, enable the browser flag, use headless Playwright, and use the named
  budgets. Remove the unset external-site policy path.
- Verify navigate, state, type or select, submit, stale-id recovery,
  low-confidence top-three exposure, external approval or denial, cleanup,
  and bounded no-secret logging.
- Keep a headed option only for manual debugging. It is not a supported smoke
  acceptance path.

Files touched:

- `harness/tests/browser_fixture/index.html`
- `harness/tests/browser_fixture/fixture_server.py`
- `harness/tests/browser_fixture/assets/*`
- `harness/tools/browser_live_smoke.py`
- `harness/tests/test_browser_live_smoke.py`
- `harness/pyproject.toml` and `harness/uv.lock` only if the browser extra
  needs a lockfile update

Exit criteria:

- The smoke refuses to run without its explicit gate and required provider
  configuration.
- The smoke uses only the ephemeral localhost origin in its allowlist.
- The headless flow proves the complete form path and cleans up the server,
  browser context, and registry.
- External navigation reaches shared safety and cannot proceed from a user
  request alone.
- Stale and low-confidence paths return their stable structured outcomes.
- Logs exclude passwords, cookies, authorization headers, full HTML, and
  unbounded extraction.
- No normal unit test needs a network connection, provider key, or browser
  binary.
- The module-limit test remains green.

Targeted test set:

- fixture server lifecycle and deterministic route tests;
- smoke disabled, missing-config, wrong-origin, headless-flow, stale-state,
  low-confidence, external-approval, cleanup, and log-bound tests;
- existing browser adapter and safety tests;
- `harness/tests/test_module_limits.py::test_module_limits`.

Non-goals:

- Do not use an external website, personal credential, payment account,
  production data, real download, or headed CI run.
- Do not add tabs, uploads, arbitrary JavaScript, authentication management,
  CAPTCHA handling, cookie export, or screenshot control.

### lane 4: add the offline browser evaluation harness

Depends on: lanes 1 and 2. It may land before lane 3, but the live smoke does
not prove the evaluation metrics.

Scope:

- Add a browser task corpus and offline runner using the existing fake adapter
  and mocked Jev transport.
- Compare the stable routed arm with an equivalent stock large-tool arm using
  the same pages, prompts, timeouts, budgets, and safety policy.
- Cover catalog sizes 10, 40, 120, 500, and 2,000; static, moderate churn,
  and full churn; clear targets, repeated labels, search triage, and forms.
- Report top-1, top-3, page-state accuracy, task success, risky false
  approvals, Jev cost and tokens per step, provider turns, stale recovery,
  retries, and time per successful step.
- Separate pre-filter misses, Jev selection misses, adapter failures, and
  page failures. Record threshold versions and confidence by primitive.

Files touched:

- `harness/evals/browser_tasks.jsonl`
- `harness/evals/browser_eval.py`
- `harness/evals/RESULTS.md` only for committed offline baseline results
- `harness/tests/test_browser_evals.py`
- `harness/tests/test_evals.py` only for shared runner plumbing

Exit criteria:

- The runner uses no network, provider key, or Playwright binary.
- Routed and stock arms use equivalent fixtures and safety policy.
- The report contains all primary metrics and failure categories.
- The large-catalog crossover matrix is present.
- The runner cannot report a routing win from token reduction alone.
- Eval, browser, safety, injection, and module-limit tests pass.

Targeted test set:

- task loading and safe prompt validation;
- fake fixture generation and catalog-size matrix;
- churn matrix, search triage, form tasks, and stale recovery;
- metric calculation, cost accounting, confidence calibration, and report
  schema;
- safety parity and false-approval accounting;
- `harness/tests/test_module_limits.py::test_module_limits`.

Non-goals:

- Do not call the live Gateway or tune thresholds from eval output.
- Do not add production traffic, a live browser dependency, or a new tool
  surface.

## ordering and acceptance gate

Lane 1 is mandatory before any other lane. Lane 2 supplies the policy needed
by the live smoke and evaluation runner. Lane 3 validates the real headless
path. Lane 4 measures offline quality and cost. Existing implementation rows
remain acceptance gates, even when their lane has no new code.

Each lane must preserve:

- default-off browser registration;
- fail-closed identity, provider, gate, budget, origin, and safety behavior;
- stable seven-tool schemas;
- plain browser data with no framework objects or page instructions;
- `MAX_FILE_LINES=1250` and `MAX_FILES_PER_DIRECTORY=17`.

## self-review

This revision starts from current code evidence. It does not re-plan merged
browser modules. It places the default-off gate before every later surface
change. It records the missing threshold, approval-wiring, retry, batch,
error-kind, budget, origin, local-smoke, and eval work.
Every lane has scope, files, exit criteria, targeted tests, and non-goals.
The design-spec test families are mapped to existing coverage or a named gap.
Failure paths use one conservative polarity: uncertainty stops or escalates;
no uncertain browser action proceeds silently.
