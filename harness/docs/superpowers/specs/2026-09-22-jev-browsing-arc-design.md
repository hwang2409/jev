# jev-navigated browsing: arc 3 design

Date: 2026-09-22  
Status: proposed design for review  
Scope: one Playwright-backed browsing arc in the experimental `harness/` fork

## 1. goal and scope

Jev-navigated browsing lets the agent operate a live web page through a small,
stable harness surface. Jev chooses from the current page's elements. The page
does not become a second instruction channel.

This arc covers the smallest useful task loop:

1. navigate to a URL;
2. read a bounded page-state snapshot;
3. click, type, or select one page element;
4. extract bounded text or structured attributes;
5. submit a form or other goal action after the safety gate.

The arc also covers page-state Nouls, result triage, stale-state recovery, and
the Playwright adapter seam needed for offline tests.

This arc does not cover a general browser product. It does not include tabs,
downloads as a user-facing feature, uploads, arbitrary JavaScript execution,
browser extensions, authentication management, CAPTCHA solving, cookie export,
visual screenshot control, or a broad browser API. A later arc may add these
surfaces after this arc has evidence.

The harness owns one browser session for one agent session. The session starts
on first browser use and closes with the agent session. It does not share a
browser context across unrelated agents. The default run is headless. A
headed smoke run is an explicit test choice.

## 2. toolset and ownership

### 2.1 Stable harness tools

The registry exposes these tools. Their schemas stay stable across turns. The
live element catalog travels as structured state for Jev, not as a changing
provider tool list.

| tool | purpose | side effect |
| --- | --- | --- |
| `browser_navigate` | Open an allowed URL in the session page. | Navigation and page state change. |
| `browser_state` | Return the current bounded page snapshot and element catalog. | None. |
| `browser_click` | Click one catalog element by stable snapshot id. | Page state may change. |
| `browser_type` | Replace or append text in one input by snapshot id. | Page state may change. |
| `browser_select` | Select one option in a select control by snapshot id and value. | Page state may change. |
| `browser_extract` | Return bounded text or selected attributes from one element or the page. | None. |
| `browser_submit` | Submit a form or click the identified submit control after safety approval. | External or durable state may change. |

The model sees the tool descriptions and the current snapshot result. It does
not receive one tool schema per page element. A browser action names an
element id from the latest snapshot. The handler rejects ids from an older
snapshot and asks for a fresh state.

`browser_state` is the recovery tool. The agent calls it after navigation,
after an action that changes the page, after a stale-id error, and when a page
state Noul says that progress is unclear.

`browser_navigate` accepts only an absolute URL with an explicit scheme. The
first implementation allows `http` and `https`, subject to the smoke policy.
It does not accept a page-provided URL as an instruction. A user task or an
explicit agent decision supplies the target.

### 2.2 Browser session lifecycle

`BrowserSession` owns one Playwright browser, one browser context, and one
page. The session is lazy and is closed from the tool registry or loop cleanup
path. A failed start returns a structured tool error. Cleanup is best effort
and never hides the original tool error.

The session owns these limits:

- navigation timeout and action timeout are named settings;
- one page is in scope for this arc;
- page content, extracted text, and catalog text have byte caps;
- every snapshot gets a monotonically increasing `snapshot_id`;
- every action records the snapshot id used for its element lookup;
- a navigation or frame change invalidates all prior element ids.

Playwright is an implementation detail behind the adapter. Tool handlers do
not import Playwright types or call a browser method directly.

### 2.3 Adapter seam

Mirror the calendar EventKit pattern in `src/zeta/tools/calendar.py`. That
module keeps framework types inside `EventStoreAdapter`, defines a narrow
`CalendarAdapter` protocol, injects the adapter into tool logic, and uses
`FakeCalendarAdapter` in `tests/test_calendar_tools.py`. Browsing follows the
same shape:

- define a narrow `BrowserAdapter` protocol for launch, navigation, snapshot
  inspection, actions, extraction, and close;
- put Playwright imports and locator conversion inside
  `PlaywrightBrowserAdapter`;
- pass an adapter or session factory into browser handlers;
- use a deterministic fake adapter for catalog, stale-state, timeout, and
  navigation-race tests;
- keep browser framework objects out of structured tool results.

The adapter returns plain dataclasses or dictionaries. It does not return
locators, element handles, page objects, or arbitrary browser objects. This
keeps the catalog builder and policy code deterministic and testable.

An illustrative seam is:

```python
class BrowserAdapter(Protocol):
    async def navigate(self, url: str, timeout_ms: int) -> PageObservation: ...
    async def observe(self, limits: SnapshotLimits) -> PageObservation: ...
    async def click(self, element_ref: ElementRef, timeout_ms: int) -> ActionObservation: ...
    async def type_text(self, element_ref: ElementRef, text: str, replace: bool, timeout_ms: int) -> ActionObservation: ...
    async def select(self, element_ref: ElementRef, value: str, timeout_ms: int) -> ActionObservation: ...
    async def extract(self, target: ElementRef | None, attributes: list[str], limit: int) -> ExtractedData: ...
    async def close(self) -> None: ...
```

This snippet illustrates the boundary only. It is not an implementation
plan.

### 2.4 Registry and loop integration

The tools use the existing `ToolRegistry.register` contract and structured
tool results. The browser session binds to the agent session, like the
calendar adapter binds to the session's tool calls. The loop does not own
Playwright objects.

The existing router supports a static invoke surface and provider-side
selection. Arc 3 uses that stable surface for browser operations. The page
element catalog is a Jev input catalog, not a registry catalog. This avoids
the prompt-cache failure seen when per-turn tool schemas change.

The browser tools record the current catalog and snapshot id in the registry's
router context. A route decision must select an element and an affordance from
that exact catalog. If Jev returns a tool-level choice without an element
choice, the handler returns a low-confidence routing error and requests a
fresh, more specific step.

## 3. element-as-catalog routing

### 3.1 Snapshot to Choice catalog

Each `browser_state` observation builds a bounded catalog. One entry represents
one actionable or extractable page element. The entry has stable fields:

```json
{
  "element_id": "e17",
  "role": "button",
  "text": "Continue to checkout",
  "affordance": "click",
  "name": "continue",
  "value_hint": null,
  "landmark": "main",
  "disabled": false,
  "visible": true
}
```

`element_id` is an opaque snapshot-local id. The adapter stores the mapping
from that id to a locator or equivalent private reference. Jev sees only the
plain catalog entry. The catalog does not expose CSS selectors, XPath, hidden
attributes, cookies, page scripts, or arbitrary HTML.

The builder includes useful roles such as links, buttons, text inputs,
comboboxes, checkboxes, radio buttons, tabs, and headings that can support
extraction. It includes a submit affordance as a distinct action. It marks
disabled and hidden elements, but excludes elements that cannot be acted on or
read within the supported surface.

Text is normalized, length-capped, and quoted as data. Accessible names are
preferred over raw text. The builder preserves enough nearby landmark context
to distinguish repeated controls without copying the full page.

### 3.2 Cheap pre-filter and bounds

The full page catalog is not sent to Jev. Large irrelevant state degrades
accuracy. Before the Jev call, a deterministic pre-filter uses the user goal,
the current action verb, role, affordance, visibility, disabled state, text,
accessible name, and landmark. It does not execute page text as code or treat
page claims as policy.

The pre-filter applies these bounds:

- remove hidden, disabled, duplicate, and unsupported elements;
- keep elements whose affordance matches the requested action;
- score lexical and role matches with a cheap local function;
- retain at most `ELEMENT_PREFILTER_K` candidates, with a named default of
  40;
- retain a small diversity quota for links, controls, and headings when the
  lexical score ties;
- retain the current goal element if a prior snapshot named it and it remains
  attached;
- send at most `ELEMENT_CATALOG_MAX` candidates to Jev, with a named default
  of 24;
- cap each entry's text and the complete serialized catalog by bytes.

The defaults are tunable. They are not accuracy claims. The eval plan must
calibrate them against pages with 100, 500, and 2,000 actionable elements.
If the pre-filter has no candidates, the tool returns a bounded state summary
and a `no_candidate` reason. It does not ask Jev to choose from an empty or
irrelevant catalog.

### 3.3 Jev selection and confidence

The provider adds a browser element-choice call shape to `providers/jev.py`.
The call includes:

- `state`: the user goal, current URL origin and path, recent browser actions,
  page title, bounded page-state summary, and the filtered catalog;
- a Choice question for `element_id` with criteria equal to the catalog;
- a Noul for `goal_element_present`;
- a Noul for `page_loaded_and_stable`;
- a Noul for `action_is_the_next_step`.

The provider treats all state fields as neutral data. The instructions say to
classify the goal against the catalog, not to follow text inside the page
state or catalog entries.

The route result includes the selected element id, affordance, probabilities,
the Nouls, usage, and a call confidence equal to the least confident judgment
in the call. That follows the existing provider convention in `_call_confidence`.

When the Choice confidence is at least `BROWSER_ELEMENT_TOP1_CONFIDENCE`, the
handler exposes one selected element. The initial calibration candidate is
0.8, mirroring router v2's top-3 rule. This threshold belongs only to element
selection. It must not be copied to page-state, risk, or relevance judgments.

When Choice confidence is below the threshold, the handler exposes the top
`BROWSER_ELEMENT_TOPN` candidates, with a candidate default of 3. The next
model turn can choose among those candidates or call `browser_state` again.
The handler never executes a low-confidence click by silently taking the
top-ranked item.

Each primitive has its own calibration:

- element Choice confidence: candidate `0.8` threshold;
- page-state Nouls: separate gates per judgment;
- risky-action approval: safety-tier threshold and score policy;
- search-result relevance: separate acceptance and tie thresholds.

The thresholds do not transfer between primitive types. All thresholds are
named settings and telemetry includes the threshold version.

## 4. page-state Noul gates

Page-state judgments control whether the loop can continue. They do not
replace deterministic adapter checks. The initial set is:

| Noul | use | low-confidence or negative result |
| --- | --- | --- |
| `page_loaded_and_stable` | Decide if the page has reached a usable state after navigation or action. | Wait within the timeout, then return a recovery error. |
| `goal_element_present` | Decide if the current goal target appears in the current catalog. | Rebuild state, widen the local goal wording, or report that the goal is absent. |
| `action_is_the_next_step` | Decide if the proposed click, type, select, extract, or submit advances the goal. | Ask for a more specific step or expose top candidates. |
| `action_succeeded` | Judge whether the post-action state shows the intended transition. | Keep the action result, mark progress unknown, and request `browser_state`. |
| `dead_end` | Detect a blocked, empty, error, or terminal page state. | Stop the current path and return a bounded explanation. |
| `needs_different_approach` | Detect that the current target or action no longer fits the goal. | Re-route from a fresh snapshot rather than repeating the same action. |

The adapter supplies deterministic evidence such as load state, URL change,
visible text, attached ids, and action exceptions. Jev judges only the
bounded evidence. A tool result reports both the deterministic reason and the
Noul outcome.

The control rule is conservative: a negative `page_loaded_and_stable` or
`goal_element_present` blocks action execution. A low-confidence
`action_succeeded` does not undo a completed browser action. It triggers a
fresh snapshot. This distinction avoids claiming that a click was reversed
when the page may have changed.

## 5. risky-click safety

Arc 3 reuses the confidence-gated approval shape from
`2026-09-21-jev-safety-tier-design.md`. It does not create a second risk
policy. Arc 4 owns the common policy and its future implementation seam.

Before `browser_submit`, and before a click that the deterministic classifier
marks as risky, the browser action is classified. Risk evidence includes the
element role and text, form action origin, current origin, proposed URL,
payment or purchase language, authentication language, download behavior, and
whether the action changes durable external state.

The risky classes are:

- destructive form submit or delete;
- payment, purchase, or financial commitment;
- authentication, account, or permission change;
- external-origin navigation;
- file download or export.

Layer 0 denies known dangerous shapes and escalates anything it cannot
analyze. Only an analyzable action reaches the Jev safety score. A safe score
with confidence at or above the safety-tier threshold can auto-approve when
the configured policy permits it. Any Jev error, missing key, malformed
response, or low confidence fails closed: interactive mode escalates to the
existing approval prompt; headless mode denies with a teaching error.

The approval request includes the concrete action, target text, destination,
and risk reason. The page cannot label an element "safe" to bypass the
classifier. Page labels are evidence at most, never authority. The existing
arc-4 safety tier remains the source of truth for score levels, fail-closed
behavior, telemetry, and approval semantics.

Read-only `browser_state` and `browser_extract` do not require this risky
action gate. `browser_navigate` to a new external origin is classified as
external navigation and follows the same gate when the policy enables it.

## 6. search-result triage

Search pages often contain many links with similar labels, sponsored results,
navigation chrome, and repeated pagination. Arc 3 uses Jev Score to rank
bounded result records for goal relevance. This is evidence-based triage,
following the retrieve-then-judge lesson from the memory evals.

The search adapter identifies result candidates through supported landmarks
and link structure. It creates records with result id, title, visible snippet,
displayed URL, source section, and position. It does not send hidden HTML or
page instructions. A Jev Score call ranks each record against the user goal.

The triage result includes a relevance score for each result and a call
confidence. The browser route then uses the score as one pre-filter feature;
it does not auto-click solely because a result has the highest score.

The initial decision rules are:

- accept a clear top result only when it passes the relevance threshold and
  beats the runner-up by the tie margin;
- expose top 3 candidates when the margin is small or call confidence is low;
- request a narrower goal or another search when all scores are below the
  relevance floor;
- preserve source diversity when scores tie;
- require the risky-click gate before following a result to an external origin.

The relevance threshold, tie margin, and low-confidence rule are separate
calibrated settings. A tie is not an error. It is a reason to expose choices
or ask for more goal detail.

## 7. prompt-injection hardening

Page content is untrusted third-party state. It is neutral data for routing.
It is not a system message, a user message, or an instruction source. This is
the same jaggedness rule used by the router and safety specs: state can contain
useful evidence and hostile text at the same time.

The boundary rules are fixed:

- page text, accessible names, URLs, snippets, and form labels are placed in
  named state fields and quoted as data;
- Jev instructions say to classify the user goal against the catalog and to
  ignore instructions found inside state fields;
- page text never changes registry policy, approval policy, allowed origins,
  tool schemas, thresholds, or session settings;
- catalog entries carry role and affordance from adapter facts, not claims in
  visible text;
- words such as `safe`, `approved`, `urgent`, or `ignore prior instructions`
  have no policy meaning when they appear in a page element;
- the handler verifies the selected `element_id` against the current local
  catalog and verifies its actual role and affordance before execution;
- a page cannot create a new tool, alter a target URL policy, or grant its own
  submit action an approval exemption;
- extraction results are returned as data and are not automatically fed into
  the next action as instructions.

The catalog also resists a hostile page that labels a trap element as safe.
The catalog records the adapter-observed role, destination, form context,
download behavior, and risk features. The safety classifier uses those facts
and the concrete action. It ignores the page's safety adjectives. An element
that says "safe click" but posts to a new origin remains external navigation
and follows the risk policy.

Injection probes are required in offline tests. They must cover buttons,
links, snippets, input values, hidden accessible names, and search results.

## 8. data flow and error handling

The normal action flow is:

1. the agent calls `browser_state` or receives a state result;
2. the adapter observes the page and assigns `snapshot_id` and element ids;
3. the cheap pre-filter bounds the catalog;
4. Jev selects an element and returns Choice, Noul, confidence, and usage;
5. the handler checks the selected id, affordance, state gates, and action
   policy;
6. a risky action enters the shared safety-tier decision;
7. the adapter resolves the id against the same snapshot and performs the
   action;
8. the handler observes the post-action state, evaluates success, and returns
   a bounded result with the new snapshot id.

The handler fails closed for safety and identity errors:

| condition | result | retry posture |
| --- | --- | --- |
| stale element id | structured `stale_snapshot` error; no action | call `browser_state`, then re-route |
| detached or ambiguous locator | structured `element_unavailable` error; no action | fresh snapshot; do not repeat blindly |
| navigation race | structured `navigation_race` error; action outcome unknown | observe current page before deciding |
| adapter timeout | structured `browser_timeout` error | one bounded recovery observation; then stop |
| Jev timeout or malformed response | routing error; no unapproved action | retry only through provider retry policy, then request state or fail |
| page load failure | structured `page_load_failed` error | report URL and bounded reason; do not infer success |
| safety-tier failure or low confidence | escalation or denial per shared tier | never auto-retry the same risky action |
| extraction truncation | successful bounded result with `truncated=true` | ask for a narrower target if more data is needed |

The router may fail open for discovery only when the existing router contract
requires a usable session. Browser actions do not fail open into an arbitrary
element. A Jev routing error can expose `browser_state` and return a teaching
error. It cannot execute a guessed click. Safety always fails closed.

Navigation races use a generation check. The handler captures the snapshot
generation before Jev selection and compares it with the generation at action
time. Any navigation, reload, or frame replacement changes the generation.
The action stops before execution when generations differ.

Every error includes a stable kind, a short human-readable message, and a
machine-readable hint. Logs record session id, snapshot id, action kind,
confidence, gate outcomes, and timing. Logs do not record passwords, full
page HTML, cookies, authorization headers, or unbounded extracted text.

## 9. testing strategy

All normal tests are offline. No test calls a real site or Jev.

### 9.1 Catalog and routing fixtures

Fixtures cover:

- a small page with buttons, links, inputs, selects, and duplicate labels;
- a 100-element page with a clear target;
- a 500-element page with irrelevant navigation and repeated controls;
- a dynamic page whose target disappears between observation and action;
- hostile labels such as `safe`, `click me`, and prompt-injection text;
- search results with a clear winner, ties, low scores, sponsored results,
  and pagination.

Tests assert deterministic filtering, role and affordance fields, text caps,
catalog caps, diversity behavior, snapshot ids, and injection neutrality.

### 9.2 Mocked Jev provider

Provider tests mock the HTTP layer and assert request shape, neutral state
fields, structured criteria, catalog bounds, response parsing, usage capture,
retry behavior, malformed responses, missing API keys, and the confidence
calculation. Separate fixtures cover element Choice, page-state Nouls, risk
classification, and result relevance. Threshold tests prove that each
primitive uses its own setting.

### 9.3 Fake browser adapter

A `FakeBrowserAdapter` mirrors `FakeCalendarAdapter`. It records navigation,
element actions, selected values, and extraction calls. It can return a new
snapshot after each action, detach an element, delay beyond a timeout, or
raise a navigation race. Tests prove that stale ids never reach the adapter's
action method and that cleanup runs after both success and failure.

### 9.4 Tool and loop tests

Tool tests cover stable schemas, structured errors, state recovery, result
truncation, safety handoff, and session ownership. Loop tests cover the
static browser tool surface, page catalog routing, top-3 expansion, unrouted
element rejection, batch behavior, and router fail-open compatibility.

### 9.5 Gated live smoke

The smoke uses a disposable, non-production site with deterministic controls.
It needs network access, a real Playwright browser binary, and a Vercel AI Gateway key.
It must not use personal credentials, payment accounts, production data, or
real downloads. The site should support navigation, a search or filter, a
form with a harmless submit, a deliberate external link, and a hostile-text
fixture.

The smoke runs headless first. A headed run is optional for debugging. It
verifies a complete navigate -> state -> select/type -> submit flow, a stale
id recovery, a low-confidence top-3 response, and a risky external or submit
action that reaches the shared approval policy. The smoke records no secrets
and cleans up its browser context.

## 10. eval plan

Arc 3 compares Jev-navigated browsing with a stock large-tool baseline. The
baseline exposes equivalent browser operations with a page-specific tool or
element schema catalog. The routed arm exposes the stable browser toolset and
passes the page catalog to Jev. Both arms use the same adapter, pages, task
prompts, timeouts, and safety policy.

The comparison must include the queued big-catalog crossover eval. It should
vary catalog size and churn, because the existing results show that routing
wins only when the catalog is large or cache-hostile. A useful matrix is:

- 10, 40, 120, 500, and 2,000 actionable elements;
- static, moderately changing, and fully changing page catalogs;
- clear target, repeated labels, search triage, and multi-step forms;
- cached and cache-less provider conditions where available.

Primary metrics:

- element-selection top-1 accuracy;
- element-selection top-3 coverage;
- page-state gate accuracy;
- end-to-end task success;
- risky-action false-approval rate, with a zero-approval target for denied
  classes;
- cost per navigation step, including Jev usage;
- model input tokens, cache reads, and provider turn count;
- stale-id recovery rate and action retry count;
- time per successful navigation step.

The eval reports confidence calibration by primitive. It separates routing
misses from page or adapter failures. It also reports catalog pre-filter
recall so a cheap filter miss is not misread as a Jev selection miss.

The routing win criterion is not token reduction alone. Arc 3 wins a cell
when it maintains task success and safety parity while lowering cost per
navigation step or preserving quality as the stock catalog becomes too large
or too dynamic for caching. At small cached catalogs, stock may remain the
better result. That is an expected outcome, not a failed thesis.

## 11. open questions and Henry decisions

These choices remain open before implementation:

1. **default-on:** should Jev-navigated browsing be default-on in this fork,
   or should a browser flag enable it while the arc is experimental?
2. **smoke sites:** which disposable site should host the headless smoke and
   hostile-content fixtures? The choice must not require personal credentials.
3. **headless versus headed:** is headless the only supported run mode for the
   first arc, with headed mode only for debugging?
4. **budget:** what are the per-page Jev call, token, action, and wall-clock
   budgets? What should happen when the budget is exhausted?
5. **origin policy:** which origins may the smoke and future sessions visit,
   and should external navigation always require approval even when the user
   asked for it explicitly?

The design defaults are headless operation, one page, no personal credentials,
bounded catalogs, fail-closed risky actions, and explicit approval for
external navigation. Henry can change these defaults before implementation.

## self-review

The document was checked for unresolved placeholders, scope drift, and
contradictory failure polarity. It has no `TODO`, `TBD`, or unfinished
placeholder markers. It keeps the browser surface to six operations plus
state, uses the existing registry and adapter patterns, gives Jev only a
bounded page catalog, and reuses the arc-4 safety policy. The eval plan
measures both routing quality and the catalog economics that motivated arc 3.
