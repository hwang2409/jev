# Harness source reorg implementation plan

## Goal

Reorganize `harness/src/zeta/` without changing harness behavior. Each
registered tool becomes a package. Shared helpers and framework support move
to their owning packages. Tool-private tests move beside their owners.

This plan translates the merged design spec into four independently reviewable
stage PRs. It does not change tool logic, schemas, assertions, provider
behavior, storage formats, public APIs, or browser scope.

## Architecture summary

`zeta.tools` remains the built-in discovery root. `pkgutil.iter_modules` still
finds top-level tool packages, and each discovered package exposes the same
synchronous `register(registry)` contract. `_shared` is hidden by the leading
underscore rule. `agent_send` remains a support package and keeps its explicit
registration call from `agent.register()`.

The final layout groups agent lifecycle under `zeta.agent`, runtime support
under `zeta.runtime`, user-facing entry points under `zeta.cli`, settings
under `zeta.config`, image helpers under `zeta.media`, model metadata under
`zeta.models`, shared protocol values under `zeta.protocol`, submission code
under `zeta.submission`, and TUI persistence under `zeta.tui`.

Stage 1 converts registered tools and the browser foundation. Stage 2 moves
shared helpers and framework support after the runtime lazy initializer is in
place. Stage 3 co-locates tests and applies pytest and wheel configuration.
Stage 4 groups the remaining loose modules in the verified dependency order.

## Spec path

`harness/docs/superpowers/specs/2026-09-22-harness-source-reorg-design.md`

## Global constraints

- Preserve tool names, schemas, handler behavior, approval behavior, result
  shapes, registration order, and the synchronous `register(registry)` API.
- Preserve `zeta.tools.<name>` for every tool package.
- Keep `zeta.cli:main` valid until the final CLI package move, then preserve it
  through `zeta/cli/__init__.py` exports.
- Do not change assertions. Change only file locations, imports, package
  exports, and the specified pytest or packaging configuration.
- Use the exact importer inventory searches from the spec at implementation
  time. Count distinct importer files after resolving relative and root-package
  aliases.
- Run every fresh-process import check in a new Python process. A warm pytest
  process is not evidence of cycle safety.
- Record the pre-stage and post-stage collected test count, ordered tool names,
  and ordered schemas. A mismatch fails the stage.
- Workers run only the targeted tests named by their task. The orchestrator
  runs the full harness suite once per PR after the worker handoff.
- Each task is sized as one implement-lane PR. A worker may split its task
  into local commits, but each commit must stay behavior-preserving and green.
- Use `uv run --frozen` for pytest commands. Run `uv build` only in the
  packaging task and inspect its wheel contents.
- Keep compatibility shims for moved high-fanout modules until a repository
  search finds no importer. A shim imports the single new implementation.
- Run a subtractive simplification pass over the final diff before handoff.

## File structure

```text
harness/src/zeta/tools/<tool>/__init__.py
harness/src/zeta/tools/browser/{__init__.py,adapter.py,catalog.py}
harness/src/zeta/tools/_shared/{__init__.py,process.py,sandbox.py,user_discovery.py}
harness/src/zeta/agent/{__init__.py,background.py,budget.py,receipt.py,runner.py,presets.py,plan_mode.py}
harness/src/zeta/cli/{__init__.py,main.py,session.py}
harness/src/zeta/config/{__init__.py,settings.py}
harness/src/zeta/media/{__init__.py,images.py}
harness/src/zeta/models/{__init__.py,catalog.py}
harness/src/zeta/protocol/{__init__.py,types.py}
harness/src/zeta/runtime/{__init__.py,execution.py,headless.py,loop.py,tool_setup.py}
harness/src/zeta/submission/{__init__.py,model.py,pipeline.py}
harness/src/zeta/tui/{__init__.py,persistence.py}
harness/src/zeta/tools/<tool>/tests/
harness/tests/zeta_test_plugin.py
```

The source tree retains `harness/tests/` for cross-cutting integration tests.
No co-located test directory gets an `__init__.py`.

## Implementation tasks

### 1. convert registered tools to packages

This is the first migration stage. Do not create `_shared` or move framework
support in this task. Keep all tests in `harness/tests/`.

#### 1a. Convert the registered tool modules

Exact moves and renames:

- `harness/src/zeta/tools/agent.py` ->
  `harness/src/zeta/tools/agent/__init__.py`;
- `harness/src/zeta/tools/bash.py` ->
  `harness/src/zeta/tools/bash/__init__.py`;
- `harness/src/zeta/tools/calendar.py` ->
  `harness/src/zeta/tools/calendar/__init__.py`;
- `harness/src/zeta/tools/edit.py` ->
  `harness/src/zeta/tools/edit/__init__.py`;
- `harness/src/zeta/tools/exec.py` ->
  `harness/src/zeta/tools/exec/__init__.py`;
- `harness/src/zeta/tools/fetch.py` ->
  `harness/src/zeta/tools/fetch/__init__.py`;
- `harness/src/zeta/tools/memory.py` ->
  `harness/src/zeta/tools/memory/__init__.py`;
- `harness/src/zeta/tools/read.py` ->
  `harness/src/zeta/tools/read/__init__.py`;
- `harness/src/zeta/tools/route.py` ->
  `harness/src/zeta/tools/route/__init__.py`;
- `harness/src/zeta/tools/skill.py` ->
  `harness/src/zeta/tools/skill/__init__.py`;
- `harness/src/zeta/tools/todo.py` ->
  `harness/src/zeta/tools/todo/__init__.py`;
- `harness/src/zeta/tools/websearch.py` ->
  `harness/src/zeta/tools/websearch/__init__.py`;
- `harness/src/zeta/tools/write.py` ->
  `harness/src/zeta/tools/write/__init__.py`.

Move each implementation directly into `__init__.py`. Do not add an `impl.py`
wrapper. Preserve module-global patch seams on the new package path. Update
absolute imports from `zeta.tools.<name>` only when the importer needs a
browser submodule path. Existing package directories are checked in place:
`tools/automation/`, `tools/zeta_background/`, and `tools/agent_send/` keep
their implementations in `__init__.py`. `agent_send` remains outside
discovery. `tools/plan_mode/` stays in place for stage 2.

Use these interim stage-1 relative imports because `_shared` does not exist:

- `agent/__init__.py`: `from ..registry`, `from ..agent_presets`, and
  `from ..agent_send`;
- `bash/__init__.py` and `exec/__init__.py`: `from .._process` and
  `from .._sandbox`;
- `memory/__init__.py`: `from .._process`;
- `read/__init__.py`, `write/__init__.py`, and `edit/__init__.py`:
  `from .._sandbox`;
- `calendar/__init__.py`, `fetch/__init__.py`, `route/__init__.py`,
  `skill/__init__.py`, `todo/__init__.py`, and `websearch/__init__.py`:
  `from ..registry`;
- `route/__init__.py`: `from ..calendar` and `from ..memory`;
- `websearch/__init__.py`: `from ..fetch`.

The registry remains at `zeta.tools.registry`. No stage-1 import may reference
`zeta.tools._shared`.

#### 1b. Move the browser foundation

Exact moves and renames:

- `harness/src/zeta/tools/browser_adapter.py` ->
  `harness/src/zeta/tools/browser/adapter.py`;
- `harness/src/zeta/tools/browser_catalog.py` ->
  `harness/src/zeta/tools/browser/catalog.py`;
- create `harness/src/zeta/tools/browser/__init__.py`.

The browser package may export foundation types without a `register()` until
browser handlers exist. Change catalog imports to `from .adapter`. Update
current browser test imports and monkeypatch targets to
`zeta.tools.browser.adapter` or `zeta.tools.browser.catalog`. Keep the
adapter and catalog as separate internal modules.

#### 1c. Boundary scan, cycle audit, and verification

There is no pytest or packaging configuration change in stage 1. Keep the
existing central test paths and discovery configuration. Verify that
`pkgutil.iter_modules(zeta.tools.__path__)` still sees every real registered
tool package, skips underscore-prefixed helpers, and keeps the `.agent` sort
special case. Do not add a discovery wrapper around `agent_send`.

Cycle audit for this task:

- modules moved: `tools/agent`, `bash`, `browser/{adapter,catalog}`,
  `calendar`, `edit`, `exec`, `fetch`, `memory`, `read`, `route`, `skill`,
  `todo`, `websearch`, and `write`;
- gate: `zeta.tools.__init__` imports only the registry; the registry does not
  import discovered children during package import; each new tool package has
  no import cycle with its newcomer;
- verdict: cycle-free with the newcomer after each package conversion.

Before each move, run the affected existing test subset. After each package
conversion, run a fresh process such as:

```sh
cd harness && for module in agent bash calendar edit exec fetch memory read route skill todo websearch write; do
  PYTHONPATH=src python -c "import zeta.tools.${module}"
done
cd harness && PYTHONPATH=src python -c 'import zeta.tools.browser.adapter; import zeta.tools.browser.catalog'
cd harness && PYTHONPATH=src python -c 'import zeta.tools.registry'
```

Use the matching package name for every changed tool. The targeted behavior
parity gate runs before and after the task:

```sh
cd harness && uv run --frozen pytest -q \
  tests/test_tool_discovery.py tests/test_calendar_tools.py \
  tests/test_memory_tools.py tests/test_read_images.py tests/test_todo.py \
  tests/test_webtools.py tests/test_tools.py tests/test_sandbox.py \
  tests/test_background.py tests/test_session_safety.py \
  tests/test_browser_adapter.py tests/test_browser_catalog.py \
  tests/test_browser_prefilter.py
```

Also run the current agent, route, and registry tests when those packages
change. Record collection count, ordered tool names, and ordered schemas.
The task is green only when the targeted suite, discovery output, and schema
order match exactly, with no assertion changes. The orchestrator runs the
full harness suite once for the PR.

Rollback: revert the stage-1 commit. Restore the original single-file module
paths and browser files if a partial revert is needed. No config restoration
is needed.

Worker lane: one implement-lane PR. Keep the two sub-tasks in one PR and use
targeted tests only.

### 2. move shared helpers and framework support

Do the lazy runtime initializer first. Do not move a runtime child before
that initializer passes its fresh-process checks.

#### 2a. Install the runtime lazy initializer and shared helpers

First replace `harness/src/zeta/runtime/__init__.py` with the lazy initializer
from the spec. It must not eagerly import `composition`, `zeta.tools.registry`,
`execution`, or `loop`.

Exact helper moves:

- `harness/src/zeta/tools/_process.py` ->
  `harness/src/zeta/tools/_shared/process.py`;
- `harness/src/zeta/tools/_sandbox.py` ->
  `harness/src/zeta/tools/_shared/sandbox.py`;
- `harness/src/zeta/tools/_user_discovery.py` ->
  `harness/src/zeta/tools/_shared/user_discovery.py`;
- create `harness/src/zeta/tools/_shared/__init__.py` as an empty initializer.

Rewrite every stage-1 helper import to `from .._shared.process`,
`from .._shared.sandbox`, or the matching user-discovery path. Update
`zeta.tools.registry` from `._process` and `._sandbox` to
`._shared.process` and `._shared.sandbox`. Update runtime, TUI, and tests that
import user discovery to `zeta.tools._shared.user_discovery`. Do not copy the
helpers into individual tool packages.

#### 2b. Move agent policy and tool setup

Create `harness/src/zeta/agent/__init__.py` as empty or lazy before moving its
children. Then make these exact moves:

- `harness/src/zeta/tools/agent_presets.py` ->
  `harness/src/zeta/agent/presets.py`;
- `harness/src/zeta/tools/plan_mode/` ->
  `harness/src/zeta/agent/plan_mode/`;
- `harness/src/zeta/tools/loop_setup.py` ->
  `harness/src/zeta/runtime/tool_setup.py`.

Update all source and test importers to the new paths. Update `loop.py` to
import `zeta.runtime.tool_setup`. Keep `tools/registry.py` at its current
path. Keep `agent_send` out of discovery and preserve the explicit call from
`agent.register()` after `agent`, `agent_status`, and `agent_output`.

#### 2c. Boundary scan, cycle audit, and verification

There is no pytest or packaging configuration change in stage 2. The boundary
scan must prove that `_shared` does not appear in `_discover_tool_modules()`.
The leading underscore remains the discovery boundary. Re-run the importer
inventory searches after all rewrites and update the importer counts.

Cycle audit for this task:

- modules moved: `_shared/{process,sandbox,user_discovery}`;
- gate: `_shared/__init__.py` is empty;
- verdict: cycle-free.
- modules moved: `zeta.agent/{presets,plan_mode}`;
- gate: `agent/__init__.py` is empty or lazy before either child move;
- verdict: cycle-free.
- module moved: `zeta.runtime/tool_setup`;
- gate: the runtime initializer is lazy before this move;
- verdict: cycle-free; this is the child that exposed the old runtime cycle.

Run these fresh-process checks immediately after the relevant changes:

```sh
cd harness && PYTHONPATH=src python -c 'import zeta.runtime'
cd harness && PYTHONPATH=src python -c 'import zeta.runtime.tool_setup'
cd harness && PYTHONPATH=src python -c 'import zeta.loop'
cd harness && PYTHONPATH=src python -c 'import zeta.tools.registry'
cd harness && PYTHONPATH=src python -c 'import zeta.tools._shared.process; import zeta.tools._shared.sandbox; import zeta.tools._shared.user_discovery'
```

The targeted behavior parity gate runs before and after the task:

```sh
cd harness && uv run --frozen pytest -q \
  tests/test_background.py tests/test_sandbox.py \
  tests/test_user_tool_discovery.py tests/test_agent.py \
  tests/test_agent_output.py tests/test_agent_status.py \
  tests/test_plan_mode.py tests/test_memory_tools.py \
  tests/test_router_auto.py tests/test_router_mode.py \
  tests/test_tool_discovery.py tests/test_import_boundaries.py \
  tests/test_tools.py tests/test_session_safety.py
```

Record collection count, ordered tool names, and ordered schemas. Compare the
same values before and after. The stage is green only when the targeted tests
pass, `_shared` stays undiscovered, and no assertion changes. The orchestrator
runs the full harness suite once for the PR.

Rollback: revert the stage-2 commit. Restore `_process.py`, `_sandbox.py`, and
`_user_discovery.py` at the tools root, restore `agent_presets.py`,
`plan_mode/`, and `loop_setup.py` to their old paths, and restore the previous
runtime initializer. Config restoration is not needed.

Worker lane: one implement-lane PR. Keep the lazy initializer and moves in
dependency order. Run targeted tests only.

### 3. co-locate tool tests and update pytest packaging

This is the designated pytest and packaging stage. Runtime source imports do
not change in this task.

#### 3a. Move the shared pytest plugin and configure discovery

Move the body of `harness/tests/conftest.py` to
`harness/tests/zeta_test_plugin.py`. The plugin owns HOME isolation, network
blocking, terminal defaults, the live-home guard, and `stock_router_mode`.
Leave `harness/tests/conftest.py` as the one-line compatibility loader:

```python
pytest_plugins = ["zeta_test_plugin"]
```

Edit `harness/pyproject.toml` so pytest uses:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "src/zeta/tools"]
pythonpath = ["tests", "src"]
addopts = ["--import-mode=importlib", "-p", "zeta_test_plugin"]
asyncio_mode = "auto"
```

Keep `harness/` as the root directory. Do not add a source-tree conftest or
duplicate fixture code. Do not add `__init__.py` to test directories. Tool
tests use absolute public package imports, with internal paths only for owned
private seams such as `zeta.tools.browser.adapter`.

#### 3b. Move and split tests by behavior owner

Move the following tool-private tests under the named directories. Preserve
their current basenames unless the spec gives a split name. A listed source
file means only the named tool-owned portions move when the source file is
mixed.

- `tools/agent/tests/`: agent portions of `test_agent.py`,
  `test_agent_output.py`, `test_agent_status.py`, `test_agents.py`,
  `test_automations.py`, `test_session.py`, `test_session_resilience.py`,
  and `test_session_safety.py`;
- `tools/agent_send/tests/`: direct send portions from `test_agent.py` and
  `test_session_shutdown.py`, including the named direct-send tests and the
  direct send assertion in `test_recovery_and_send_release_borrowed_child_stores`;
- `tools/automation/tests/`: automation tool portions of `test_automations.py`;
- `tools/browser/tests/`: `test_browser_adapter.py` -> `test_adapter.py`,
  `test_browser_catalog.py` -> `test_catalog.py`, and
  `test_browser_prefilter.py` -> `test_prefilter.py`;
- `tools/calendar/tests/`: `test_calendar_tools.py`;
- `tools/memory/tests/`: memory portions of `test_memory_tools.py`;
- `tools/read/tests/`: read portions of `test_read_images.py` and
  `test_tools.py`;
- `tools/todo/tests/`: tool-handler portions of `test_todo.py`;
- `tools/websearch/tests/`: websearch portions of `test_webtools.py`;
- `tools/fetch/tests/`: fetch portions of `test_webtools.py`;
- `tools/skill/tests/`: skill assertions from `test_skills.py:77-105`;
- `tools/route/tests/`: route-only portions of `test_evals.py` and router tests;
- `tools/bash/tests/`: bash portions of `test_tools.py` and `test_safety.py`;
- `tools/exec/tests/`: exec portions of `test_tools.py`, `test_commands.py`,
  `test_safety.py`, and `test_session_safety.py`;
- `tools/edit/tests/`: edit portions of `test_tools.py`;
- `tools/write/tests/`: write portions of `test_tools.py`;
- `tools/zeta_background/tests/`: tool portions of `test_background.py`;
- `tools/_shared/tests/`: sandbox and process portions of `test_sandbox.py`
  and `test_background.py`.

Apply the spec's exact mixed-file splits:

- all agent lifecycle and `agent_output` tests from `test_agent_output.py` go
  to `tools/agent/tests/test_agent_output.py`, except
  `test_foreground_receipt_shows_lifecycle_stats`, which goes to
  `harness/tests/test_agent_output_tui.py`;
- tool-handler, schema, validation, mutation, and transcript tests in
  `test_todo.py` through `test_todo_empty_list_clears_state_and_does_not_pollute_transcript`
  go to `tools/todo/tests/test_todo.py`;
- `test_todo_items_persist_across_store_resume` goes to
  `harness/tests/test_todo_persistence.py`;
- all `TodoWidget` and `TUIApp` rendering, layout, status, truncation,
  pinning, and terminal-size tests go to `harness/tests/test_todo_tui.py`;
- `test_import_boundaries.py` stays whole in `harness/tests/`;
- the direct-send test names and the two later agent lifecycle tests stay
  assigned exactly as specified by the behavior-owner rule;
- `test_runs_and_send_commands_drive_a_live_run` stays in `harness/tests/`;
- only lines 77-105 of `test_skills.py` move to the skill owner; the rest
  remains central;
- session and CLI lifecycle portions of `test_session_resilience.py` remain
  central.

Rebuild direct-send ownership with:

```sh
rg -n --glob 'test_*.py' 'agent_send|send_to_run' harness/tests
```

#### 3c. Update boundary scans and wheel packaging

Update both production-module scans in
`harness/tests/test_import_boundaries.py` to exclude every path matching
`**/tests/**`: the fresh-process module list and the forbidden-import scan.
Do not weaken the forbidden dependency rules.

Edit the wheel target in `harness/pyproject.toml` to add the explicit test
exclusion while keeping existing `force-include` entries unchanged:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/zeta"]
exclude = ["src/zeta/**/tests/**"]
```

#### 3d. Cycle audit and verification

No production module moves in this task. The cycle audit is therefore a
collection audit: co-located tests import the same production package paths,
and test directories are never package initializers. Verdict: no production
cycle change and no test directory enters built-in tool discovery.

Run collection in importlib mode before and after each ownership split. The
targeted parity gate is:

```sh
cd harness && uv run --frozen pytest --collect-only -q
cd harness && uv run --frozen pytest -q \
  src/zeta/tools/agent/tests src/zeta/tools/agent_send/tests \
  src/zeta/tools/automation/tests src/zeta/tools/browser/tests \
  src/zeta/tools/calendar/tests src/zeta/tools/memory/tests \
  src/zeta/tools/read/tests src/zeta/tools/todo/tests \
  src/zeta/tools/websearch/tests src/zeta/tools/fetch/tests \
  src/zeta/tools/skill/tests src/zeta/tools/route/tests \
  src/zeta/tools/bash/tests src/zeta/tools/exec/tests \
  src/zeta/tools/edit/tests src/zeta/tools/write/tests \
  src/zeta/tools/zeta_background/tests src/zeta/tools/_shared/tests \
  tests/test_import_boundaries.py tests/test_todo_persistence.py \
  tests/test_todo_tui.py tests/test_agent_output_tui.py
cd harness && uv build
```

Inspect the wheel file list. It must contain runtime modules and packaged
markdown, and no `zeta/tools/**/tests/` files. Run fresh processes for the
boundary checks after the scan exclusions and confirm that test paths are not
treated as production modules. Compare collected node counts, ordered tool
names, and ordered schemas with the pre-stage baseline. Run each moved test at
its new path. The orchestrator runs the full harness suite once for the PR.

Rollback: revert the stage-3 commit and restore the previous `pyproject.toml`,
central test files, and `conftest.py` body. This task needs config restoration:
restore `testpaths`, `pythonpath`, `addopts`, wheel `exclude`, and the original
boundary-scan path filters.

Worker lane: one implement-lane PR. Run targeted collection, moved tests,
boundary tests, and wheel inspection only.

### 4. group loose top-level modules

Keep the stage-2 runtime moves complete. Verify the lazy runtime package in a
fresh process before moving any runtime child. Move runtime children in the
specified order, then move the remaining groups.

#### 4a. Move runtime children in dependency order

First verify `zeta.runtime.__init__` and `zeta.runtime.tool_setup` from stage 2.
Then make these exact moves:

- `harness/src/zeta/execution.py` ->
  `harness/src/zeta/runtime/execution.py`;
- `harness/src/zeta/headless.py` ->
  `harness/src/zeta/runtime/headless.py`;
- `harness/src/zeta/loop.py` ->
  `harness/src/zeta/runtime/loop.py`.

Update `tools.registry` and every other importer to the new relative depth.
Move `headless.py` after `execution.py`, updating its runtime driver import.
Move `loop.py` last among runtime children, updating
`runtime.composition` and all other importers. Keep `zeta.execution` and
`zeta.loop` as re-export shims until their importer counts reach zero.

Before each child move, run a fresh process. Immediately after each move run:

```sh
cd harness && PYTHONPATH=src python -c 'import zeta.runtime.execution'
cd harness && PYTHONPATH=src python -c 'import zeta.runtime.headless'
cd harness && PYTHONPATH=src python -c 'import zeta.runtime.loop'
cd harness && PYTHONPATH=src python -c 'import zeta.runtime; from zeta.runtime import compose_runtime'
cd harness && PYTHONPATH=src python -c 'import zeta.tools.registry'
```

Use the command matching the moved child before a later test process loads the
runtime package.

#### 4b. Move the remaining loose modules

Create each target package initializer as empty or lazy before its first child
move. Make these exact moves and path changes:

- `agent_background.py` -> `zeta/agent/background.py`;
- `agent_budget.py` -> `zeta/agent/budget.py`;
- `agent_receipt.py` -> `zeta/agent/receipt.py`;
- `agent_runner.py` -> `zeta/agent/runner.py`;
- `cli.py` -> `zeta/cli/main.py`, with `zeta/cli/__init__.py` exporting
  `main`, `build_parser`, and current public names;
- `session_cli.py` -> `zeta/cli/session.py`;
- `settings.py` -> `zeta/config/settings.py`;
- `images.py` -> `zeta/media/images.py`;
- `model_catalog.py` -> `zeta/models/catalog.py`;
- `types.py` -> `zeta/protocol/types.py`;
- `submission.py` -> `zeta/submission/model.py`;
- `submission_pipeline.py` -> `zeta/submission/pipeline.py`;
- `persistence.py` -> `zeta/tui/persistence.py`.

Update every source and test importer from the spec's move table to the new
path. Keep the current root `zeta` re-exports. Preserve the console entry
point `zeta.cli:main`. Keep explicit re-export shims for `zeta.loop`,
`zeta.settings`, and `zeta.types` until repository searches show zero
importers. Remove each shim only after its replacement passes the boundary
suite and its importer count is zero. Do not maintain two implementations.

Before moving `persistence.py`, replace the eager `zeta.tui.__init__` import
of `.app` with a lazy initializer. Verify `import zeta.tui` and
`import zeta.tui.persistence` in separate fresh processes before and after the
move.

#### 4c. Boundary scan, cycle audit, and verification

There is no new pytest or packaging configuration change in stage 4. Keep the
stage-3 importlib mode, plugin loading, testpaths, and wheel exclusion.
Re-run the complete importer inventory after each high-fanout move. Boundary
scans must cover the complete `src/zeta` tree and must continue excluding
`**/tests/**`.

Cycle audit for this task:

- modules moved: `zeta.agent/{background,budget,receipt,runner}`;
- gate: `agent/__init__.py` is empty or lazy before the first child;
- verdict: cycle-free.
- modules moved: `zeta.cli/{main,session}`;
- gate: `cli/__init__.py` preserves lazy exports and `zeta.cli:main`;
- verdict: cycle-free and entry-point compatible.
- modules moved: `zeta.config/settings`, `zeta.media/images`,
  `zeta.models/catalog`, `zeta.protocol/types`,
  `zeta.submission/{model,pipeline}`;
- gate: each grouping initializer is empty or lazy before its first child;
  `types`, `settings`, and `loop` use one-way shims during migration;
- verdict: cycle-free after importer rewrites and shim checks.
- module moved: `zeta.tui/persistence`;
- gate: `zeta.tui.__init__` is lazy before the move because the old initializer
  eagerly imported `.app`;
- verdict: cycle-free.
- runtime modules moved: `execution`, `headless`, and `loop`;
- gate: `zeta.runtime.__init__` remains lazy throughout, and each fresh
  process import passes before the next runtime child move;
- verdict: cycle-free with runtime composition and registry.

The targeted parity gate runs before and after each sub-task:

```sh
cd harness && uv run --frozen pytest -q \
  tests/test_import_boundaries.py tests/test_cli.py tests/test_headless.py \
  tests/test_agent.py tests/test_agents.py tests/test_session.py \
  tests/test_session_lifecycle.py tests/test_session_resilience.py \
  tests/test_session_safety.py tests/test_session_shutdown.py \
  tests/test_settings.py tests/test_types.py tests/test_loop.py \
  tests/test_checkpoint.py tests/test_commands.py tests/test_mcp.py \
  tests/test_mcp_oauth.py tests/test_server.py tests/test_tui.py \
  tests/test_anthropic.py tests/test_codex.py tests/test_model_picker.py
```

Also run package import smoke checks for every target package in a new Python
process. Compare collection count, ordered tool names, and ordered schemas.
Run the boundary suite after each shim update. The stage is green only when
the targeted tests pass, the entry point and root exports remain usable, no
shim hides a remaining importer, and no assertion changes. The orchestrator
runs the full harness suite once for the PR.

Rollback: revert the stage-4 commit. Restore every old top-level module and
its importer paths. Restore the old `zeta.runtime.__init__` and `zeta.tui.__init__`
only if the rollback includes their initializer changes. This task needs
config restoration only if a rollback also crosses the stage-3 commit; stage
4 itself changes no pytest or wheel configuration.

Worker lane: one implement-lane PR, with runtime and remaining-group sub-tasks
kept in the verified order. Run targeted tests only.

## Arc 3 handoff

Browsing arc 3 tasks 5-12 resume only after this plan completes. New browser
handlers, including task 12's real Playwright adapter, must build in
`harness/src/zeta/tools/browser/` and use its package `register()` contract
and co-located browser tests. Do not resume those tasks against the old flat
`browser_adapter.py` or `browser_catalog.py` paths.

## Blocked on Henry decisions

No decision blocks the staged implementation plan. The spec records one open
choice: remove temporary old-path shims in the final reorg PR, or remove them
in a follow-up cleanup PR. This plan assumes removal in stage 4 after each
importer count reaches zero. If Henry chooses the follow-up, leave only the
verified one-way shims, record the remaining paths, and move shim removal to
that follow-up without changing the stage order.

## Self-review checklist

- [x] all four spec stages appear as ordered implementation tasks;
- [x] stage 1 package conversion precedes stage 2 helper moves;
- [x] the runtime lazy initializer precedes `tool_setup`, `execution`, and
  `loop` moves;
- [x] every stage lists exact source moves, import rewrites, boundary changes,
  fresh-process checks, targeted verification, and parity gates;
- [x] stage 1 includes all thirteen single-file tools and the browser move;
- [x] stage 2 includes all three helpers, agent policy, plan mode, and tool
  setup, while keeping registry and agent-send behavior stable;
- [x] stage 3 includes the shared pytest plugin, importlib collection,
  pyproject changes, mixed-test ownership, boundary scan exclusions, and wheel
  exclusion;
- [x] stage 4 includes every move-table module, runtime ordering, lazy TUI
  initializer, entry-point export, and compatibility shim rule;
- [x] each task includes its per-stage cycle audit and verdict;
- [x] each task includes a rollback, including config restoration where needed;
- [x] worker lanes run targeted tests and the orchestrator owns full suites;
- [x] no task depends on a later task;
- [x] the arc-3 tasks 5-12 handoff is explicit;
- [x] no implementation code, assertion changes, or file moves are included
  in this plan document;
- [x] all code fences have language tags;
- [x] the only spec ambiguity is recorded with a default resolution;
- [x] the final diff receives a subtractive simplification pass before PR
  handoff.
