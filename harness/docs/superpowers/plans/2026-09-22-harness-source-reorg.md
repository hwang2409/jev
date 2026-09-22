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
- Use the importer inventories listed in the relevant stage task at
  implementation time. Count distinct importer files after resolving relative
  and root-package aliases.
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
harness/src/zeta/agent/{__init__.py,background.py,budget.py,receipt.py,runner.py,presets.py}
harness/src/zeta/agent/plan_mode/__init__.py
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
- `bash/__init__.py`: `from .._process` and `from .._sandbox`;
- `exec/__init__.py`: `from .._process`;
- `memory/__init__.py`: `from .._process`;
- `read/__init__.py`, `write/__init__.py`, and `edit/__init__.py`:
  `from .._sandbox`;
- `calendar/__init__.py`, `fetch/__init__.py`, `route/__init__.py`,
  `skill/__init__.py`, `todo/__init__.py`, and `websearch/__init__.py`:
  `from ..registry`;
- `route/__init__.py`: `from ..calendar` and `from ..memory`;
- `websearch/__init__.py`: `from ..fetch`.

Also increase the root-relative depth for every `from ..` import in a moved
tool. The complete verified rewrite list is:

- `tools/agent/__init__.py`: `from ..agent_receipt` ->
  `from ...agent_receipt`; `from ..core.approval` -> `from ...core.approval`;
  `from ..core.checkpoints` -> `from ...core.checkpoints`;
  `from ..core.session_files` -> `from ...core.session_files`;
  `from ..core.store` -> `from ...core.store`; `from ..model_catalog` ->
  `from ...model_catalog`; `from ..types` -> `from ...types`.
- `tools/bash/__init__.py`: `from ..core.abort` -> `from ...core.abort`;
  `from ..types` -> `from ...types`.
- `tools/calendar/__init__.py`: `from ..types` -> `from ...types`.
- `tools/edit/__init__.py`: `from ..types` -> `from ...types`.
- `tools/exec/__init__.py`: `from ..core.abort` -> `from ...core.abort`;
  `from ..types` -> `from ...types`.
- `tools/fetch/__init__.py`: `from ..core.abort` -> `from ...core.abort`;
  `from ..types` -> `from ...types`.
- `tools/memory/__init__.py`: `from ..core.abort` -> `from ...core.abort`;
  `from ..execution` -> `from ...execution`; `from ..types` ->
  `from ...types`.
- `tools/read/__init__.py`: `from ..core.abort` -> `from ...core.abort`;
  `from ..images` -> `from ...images`; `from ..types` -> `from ...types`.
- `tools/route/__init__.py`: `from ..execution` -> `from ...execution`;
  `from ..providers.jev` -> `from ...providers.jev`; `from ..types` ->
  `from ...types`.
- `tools/skill/__init__.py`: `from ..skills` -> `from ...skills`.
- `tools/todo/__init__.py`: `from ..core.todo` -> `from ...core.todo`;
  `from ..types` -> `from ...types`.
- `tools/websearch/__init__.py`: `from ..core.abort` -> `from ...core.abort`;
  `from ..types` -> `from ...types`.
- `tools/write/__init__.py`: `from ..types` -> `from ...types`.

The list contains 30 individual root-relative import statements. The sibling
rewrites are the interim imports listed above: `from .registry` becomes
`from ..registry` where present, and `from ._process`, `from ._sandbox`,
`from .agent_presets`, `from .agent_send`, `from .calendar`, `from .memory`,
and `from .fetch` become their matching `from ..` imports.

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

Extend `tests/test_tool_discovery.py` with a package fixture. The fixture must
create a temporary `fixture_package/` directory containing `__init__.py` with
a callable `register(registry)`, a temporary `_shared/` package containing
`__init__.py` with a register function that raises if called, and a temporary
`agent/` package containing `__init__.py` with a register function. Point the
patched discovery path at that directory and assert that the package is
discovered, `_shared` is excluded, and the sorted result places the
`zeta.tools.agent` special case after the other discovered names. Keep the
existing module fixture as a separate test so both module and package
discovery remain covered.

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

First replace `harness/src/zeta/runtime/__init__.py` with this lazy initializer.
It must not eagerly import `composition`, `zeta.tools.registry`, `execution`,
or `loop`:

```python
__all__ = [
    "DENIAL_MARKER",
    "TOOL_RESULT_MAX_BYTES",
    "RuntimeComposition",
    "build_unattended_loop",
    "compose_runtime",
    "drive_turn",
]

_EXPORTS = {
    "DENIAL_MARKER": (".driver", "DENIAL_MARKER"),
    "TOOL_RESULT_MAX_BYTES": (".driver", "TOOL_RESULT_MAX_BYTES"),
    "RuntimeComposition": (".composition", "RuntimeComposition"),
    "build_unattended_loop": (".unattended", "build_unattended_loop"),
    "compose_runtime": (".composition", "compose_runtime"),
    "drive_turn": (".driver", "drive_turn"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    module = importlib.import_module(module_name, __name__)
    return getattr(module, attribute_name)
```

Add `import importlib` at the top. Verify these exact exports in separate
fresh processes: `import zeta.runtime`; `from zeta.runtime import
RuntimeComposition, build_unattended_loop, compose_runtime, drive_turn`;
`from zeta.runtime import DENIAL_MARKER, TOOL_RESULT_MAX_BYTES`; and assert
that importing `zeta.runtime` alone does not load `zeta.runtime.composition`,
`zeta.tools.registry`, `zeta.runtime.execution`, or `zeta.runtime.loop`.

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

Use this complete importer rewrite list:

- `tools/bash/__init__.py`: `from .._process` ->
  `from .._shared.process`; `from .._sandbox` -> `from .._shared.sandbox`.
- `tools/edit/__init__.py`: `from .._sandbox` ->
  `from .._shared.sandbox`.
- `tools/exec/__init__.py`: `from .._process` ->
  `from .._shared.process`.
- `tools/memory/__init__.py`: `from .._process` ->
  `from .._shared.process`.
- `tools/read/__init__.py`: `from .._sandbox` ->
  `from .._shared.sandbox`.
- `tools/write/__init__.py`: `from .._sandbox` ->
  `from .._shared.sandbox`.
- `tools/zeta_background/__init__.py`: `from .._sandbox` ->
  `from .._shared.sandbox`.
- `tools/registry.py`: `from ._process` -> `from ._shared.process`;
  `from ._sandbox` -> `from ._shared.sandbox`.
- `mcp/stdio.py`: `from ..tools._process` ->
  `from ..tools._shared.process`.
- `runtime/composition.py`: `from ..tools._user_discovery` ->
  `from ..tools._shared.user_discovery`.
- `tui/app.py`: `from ..tools._user_discovery` ->
  `from ..tools._shared.user_discovery`.
- `tui/slash_handlers/__init__.py`: `from ...tools._user_discovery` ->
  `from ...tools._shared.user_discovery`.
- `tests/test_background.py`: `from zeta.tools._process` ->
  `from zeta.tools._shared.process`.
- `tests/test_sandbox.py`: `import zeta.tools._sandbox as sandbox_module` ->
  `import zeta.tools._shared.sandbox as sandbox_module`.
- `tests/test_session_safety.py`: `from zeta.tools._process` ->
  `from zeta.tools._shared.process`.
- `tests/test_tools.py`: `import zeta.tools._sandbox as sandbox_module` ->
  `import zeta.tools._shared.sandbox as sandbox_module`; `from
  zeta.tools._process` -> `from zeta.tools._shared.process`.
- `tests/test_user_tool_discovery.py`: `from zeta.tools._user_discovery` ->
  `from zeta.tools._shared.user_discovery`.

Move `agent_presets.py`, `plan_mode/`, and `loop_setup.py` only after these
helper rewrites pass. Their complete importer rewrites are:

- `tools/agent/__init__.py`: `from ..agent_presets` ->
  `from ...agent.presets`.
- `agent_runner.py`: `from .tools.agent_presets` -> `from .agent.presets`.
- `loop.py`: `from .tools.agent_presets` -> `from .agent.presets`;
  `from .tools.loop_setup` -> `from .runtime.tool_setup`.
- `skills/agent_catalog.py`: `from ..tools.agent_presets` ->
  `from ..agent.presets`.
- `tui/agent_card.py`: `from ..tools.agent_presets` ->
  `from ..agent.presets`.
- `tools/plan_mode/__init__.py`, moved to `agent/plan_mode/__init__.py`:
  `from ..agent_presets` -> `from ..presets`.
- `loop.py`: `from .tools.plan_mode` -> `from .agent.plan_mode`.
- `tests/test_agent.py`: `from zeta.tools.agent_presets` ->
  `from zeta.agent.presets` at the module import and both function-local
  imports.
- `tests/test_plan_mode.py`: `from zeta.tools.plan_mode` ->
  `from zeta.agent.plan_mode`.

The post-stage-2 importer search must return no old helper, preset, plan-mode,
or tool-setup path:

```sh
rg -n --glob '*.py' \
  'zeta\.tools\._(process|sandbox|user_discovery)|zeta\.tools\.agent_presets|zeta\.tools\.plan_mode|tools\.(agent_presets|loop_setup|plan_mode)' \
  harness/src harness/tests
```

The relative-import audit was regenerated with `rg 'from \.{1,3}tools\.'`.
It found the listed `agent_presets`, `loop_setup`, and `plan_mode` importers;
the other matches are registry or tool imports that stay in place until their
later stage-4 moves.

#### 2b. Move agent policy and tool setup

Create `harness/src/zeta/agent/__init__.py` as an empty initializer before
moving its children. Then make these exact moves:

- `harness/src/zeta/tools/agent_presets.py` ->
  `harness/src/zeta/agent/presets.py`;
- `harness/src/zeta/tools/plan_mode/` ->
  `harness/src/zeta/agent/plan_mode/`, retaining
  `harness/src/zeta/agent/plan_mode/__init__.py`;
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
their current basenames unless a destination basename appears below. A listed
source file means only the named tool-owned portions move when the source file
is mixed.

- `tools/agent/tests/`: the agent-owned destinations named in the split table
  below, plus `test_agent_status.py`, `test_agents.py`,
  `test_session.py`, `test_session_resilience.py`, and
  `test_session_safety.py`;
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
- `tools/route/tests/`: route portions of `test_evals.py` ->
  `test_route_evals.py`, and router tests -> `test_router.py`;
- `tools/bash/tests/`: bash portions of `test_tools.py` and `test_safety.py`;
- `tools/exec/tests/`: exec portions of `test_tools.py`, `test_commands.py`,
  `test_safety.py`, and `test_session_safety.py`;
- `tools/edit/tests/`: edit portions of `test_tools.py`;
- `tools/write/tests/`: write portions of `test_tools.py`;
- `tools/zeta_background/tests/`: tool portions of `test_background.py`;
- `tools/_shared/tests/`: sandbox and process portions of `test_sandbox.py`
  and `test_background.py`.

Apply these exact mixed-file splits and destination basenames:

| source file | destination files | ownership split |
| --- | --- | --- |
| `test_agent.py` | `src/zeta/tools/agent/tests/test_agent.py`, `src/zeta/tools/agent_send/tests/test_agent_send.py`, `tests/test_agent_integration.py` | agent lifecycle and agent behavior; direct `agent_send`; live-run slash-command integration remains central |
| `test_agent_output.py` | `src/zeta/tools/agent/tests/test_agent_output.py`, `tests/test_agent_output_tui.py` | all agent lifecycle and `agent_output`; the TUI receipt test remains central |
| `test_automations.py` | `src/zeta/tools/agent/tests/test_automations.py`, `src/zeta/tools/automation/tests/test_automation.py`, `tests/test_automations_integration.py` | agent lifecycle; automation tool; cross-layer automation integration |
| `test_session.py` | `src/zeta/tools/agent/tests/test_session.py`, `tests/test_session_integration.py` | agent-owned session behavior; session and CLI integration |
| `test_session_resilience.py` | `src/zeta/tools/agent/tests/test_session_resilience.py`, `tests/test_session_resilience_integration.py` | agent lifecycle; session and CLI lifecycle |
| `test_session_safety.py` | `src/zeta/tools/agent/tests/test_session_safety.py`, `src/zeta/tools/exec/tests/test_session_safety.py`, `tests/test_session_safety_integration.py` | agent lifecycle; exec abort behavior; cross-layer session safety |
| `test_session_shutdown.py` | `src/zeta/tools/agent_send/tests/test_agent_send.py`, `tests/test_session_shutdown.py` | direct-send assertion; session shutdown and lease recovery |
| `test_memory_tools.py` | `src/zeta/tools/memory/tests/test_memory_tools.py`, `tests/test_memory_tools_integration.py` | memory tool; settings and session integration |
| `test_read_images.py` | `src/zeta/tools/read/tests/test_read_images.py`, `tests/test_read_images_integration.py` | read tool; loop and image integration |
| `test_tools.py` | `src/zeta/tools/read/tests/test_read.py`, `src/zeta/tools/bash/tests/test_bash.py`, `src/zeta/tools/exec/tests/test_exec.py`, `src/zeta/tools/edit/tests/test_edit.py`, `src/zeta/tools/write/tests/test_write.py`, `tests/test_tools_integration.py` | read, bash, exec, edit, and write handlers; registry and loop integration |
| `test_todo.py` | `src/zeta/tools/todo/tests/test_todo.py`, `tests/test_todo_persistence.py`, `tests/test_todo_tui.py` | handler through the named empty-list test; store-only persistence; TUI widgets and app |
| `test_webtools.py` | `src/zeta/tools/fetch/tests/test_fetch.py`, `src/zeta/tools/websearch/tests/test_websearch.py` | fetch and websearch |
| `test_skills.py` | `src/zeta/tools/skill/tests/test_skill_tool.py`, `tests/test_skills.py` | lines 77-105; all other skill and catalog behavior |
| `test_evals.py` | `src/zeta/tools/route/tests/test_route_evals.py`, `tests/test_evals.py` | route-owned eval cases; remaining eval harness |
| `test_safety.py` | `src/zeta/tools/bash/tests/test_bash_safety.py`, `src/zeta/tools/exec/tests/test_exec_safety.py`, `tests/test_safety.py` | bash safety; exec safety; shared safety policy |
| `test_commands.py` | `src/zeta/tools/exec/tests/test_exec_commands.py`, `tests/test_commands.py` | exec macro tests; command and submission integration |
| `test_background.py` | `src/zeta/tools/zeta_background/tests/test_background.py`, `src/zeta/tools/_shared/tests/test_process.py`, `tests/test_background_integration.py` | background tool; process helper; registry and integration tests |

The central remainder basenames in this table are part of the plan. Do not
leave any split as an unnamed subset.

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
- these direct-send tests from `test_agent.py` go to
  `src/zeta/tools/agent_send/tests/test_agent_send.py`:
  `test_agent_send_waits_for_blocked_append_before_cancellation`,
  `test_tool_registry_reports_agent_send_result_after_cleanup_cancellation`,
  `test_agent_send_aborts_before_append_when_store_lock_is_held`,
  `test_agent_send_reports_when_the_run_just_closed`,
  `test_a_follow_up_reaches_the_run_at_its_next_turn`,
  `test_send_to_run_rejects_unknown_and_finished_runs`, and
  `test_send_to_run_rejects_non_run_children`;
- `test_recovery_and_send_release_borrowed_child_stores` in
  `test_session_shutdown.py` contributes its direct-send assertion to the same
  `test_agent_send.py`; its shutdown and lease-recovery assertions stay in
  `tests/test_session_shutdown.py`;
- `test_a_queued_prompt_stays_out_of_the_run_context` and
  `test_a_run_with_an_empty_queue_finishes_normally` go to
  `src/zeta/tools/agent/tests/test_agent.py`;
- `test_runs_and_send_commands_drive_a_live_run` stays in
  `tests/test_agent_integration.py`;
- only lines 77-105 of `test_skills.py` move to the skill owner; the rest
  remains central;
- session and CLI lifecycle portions of `test_session_resilience.py` remain
  central.

Rebuild direct-send ownership with:

```sh
rg -n --glob 'test_*.py' 'agent_send|send_to_run' harness/tests
```

The search must find the seven direct-send tests and the one direct-send
assertion listed above. It must not move the two agent lifecycle tests or the
live-run slash-command integration test into the direct-send file.

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
pre-move gate uses the current central files:

```sh
cd harness && uv run --frozen pytest --collect-only -q \
  tests/test_agent.py tests/test_agent_output.py tests/test_automations.py \
  tests/test_session.py tests/test_session_resilience.py \
  tests/test_session_safety.py tests/test_session_shutdown.py \
  tests/test_memory_tools.py tests/test_read_images.py tests/test_tools.py \
  tests/test_todo.py tests/test_webtools.py tests/test_skills.py \
  tests/test_evals.py tests/test_safety.py tests/test_commands.py \
  tests/test_background.py
```

The post-move gate executes every destination in the split table:

```sh
cd harness && uv run --frozen pytest --collect-only -q \
  src/zeta/tools/agent/tests/test_agent.py \
  src/zeta/tools/agent_send/tests/test_agent_send.py \
  tests/test_agent_integration.py \
  src/zeta/tools/agent/tests/test_agent_output.py tests/test_agent_output_tui.py \
  src/zeta/tools/agent/tests/test_automations.py \
  src/zeta/tools/automation/tests/test_automation.py \
  tests/test_automations_integration.py \
  src/zeta/tools/agent/tests/test_session.py tests/test_session_integration.py \
  src/zeta/tools/agent/tests/test_session_resilience.py \
  tests/test_session_resilience_integration.py \
  src/zeta/tools/agent/tests/test_session_safety.py \
  src/zeta/tools/exec/tests/test_session_safety.py \
  tests/test_session_safety_integration.py tests/test_session_shutdown.py \
  src/zeta/tools/memory/tests/test_memory_tools.py \
  tests/test_memory_tools_integration.py \
  src/zeta/tools/read/tests/test_read_images.py \
  tests/test_read_images_integration.py \
  src/zeta/tools/read/tests/test_read.py src/zeta/tools/bash/tests/test_bash.py \
  src/zeta/tools/exec/tests/test_exec.py src/zeta/tools/edit/tests/test_edit.py \
  src/zeta/tools/write/tests/test_write.py tests/test_tools_integration.py \
  src/zeta/tools/todo/tests/test_todo.py tests/test_todo_persistence.py \
  tests/test_todo_tui.py src/zeta/tools/fetch/tests/test_fetch.py \
  src/zeta/tools/websearch/tests/test_websearch.py \
  src/zeta/tools/skill/tests/test_skill_tool.py tests/test_skills.py \
  src/zeta/tools/route/tests/test_route_evals.py tests/test_evals.py \
  src/zeta/tools/bash/tests/test_bash_safety.py \
  src/zeta/tools/exec/tests/test_exec_safety.py tests/test_safety.py \
  src/zeta/tools/exec/tests/test_exec_commands.py tests/test_commands.py \
  src/zeta/tools/zeta_background/tests/test_background.py \
  src/zeta/tools/_shared/tests/test_process.py tests/test_background_integration.py
```

Run the targeted post-move tests and wheel build:

```sh
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

The chosen sequence is `execution.py`, then the CLI import rewrite,
then `headless.py`, then `loop.py`.
Before moving `headless.py`, rewrite the current `zeta/cli.py` lazy import
from `.headless` to `.runtime.headless`. This keeps the current CLI importer
valid after `headless.py` moves. When `cli.py` moves in 4b, change that import
to `..runtime.headless` for `cli/main.py`.

Update `tools/registry.py`, `tools/memory/__init__.py`,
`tools/route/__init__.py`, and
`src/zeta/tools/agent/tests/test_agent.py` as listed in the
stage-4 importer inventory below.
Move `headless.py` after `execution.py`, updating its runtime driver import.
Move `loop.py` last among runtime children, updating
`runtime.composition` and the loop importers listed in the stage-4 inventory.
Keep `zeta.execution` and
`zeta.loop` as re-export shims until their importer counts reach zero.

Before each child move, run a fresh process. Immediately after `execution.py`
moves, run:

```sh
cd harness && PYTHONPATH=src python -c 'import zeta.runtime.execution'
cd harness && PYTHONPATH=src python -c 'import zeta.tools.registry'
```

Immediately after `headless.py` moves, run:

```sh
cd harness && PYTHONPATH=src python -c 'import zeta.cli; import zeta.runtime.headless'
```

Immediately after `loop.py` moves, run:

```sh
cd harness && PYTHONPATH=src python -c 'import zeta.runtime.loop'
cd harness && PYTHONPATH=src python -c 'import zeta.runtime; from zeta.runtime import compose_runtime'
```

#### 4b. Move the remaining loose modules

Create each target package initializer before its first child move. Use empty
initializers for `zeta.agent`, `zeta.config`, `zeta.media`, `zeta.models`,
`zeta.protocol`, and `zeta.submission`. Use the explicit lazy initializers
listed below for `zeta.cli`, `zeta.runtime`, and `zeta.tui`. Move the remaining
modules in this order:

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
- `submission.py` -> `zeta/submission/model.py`;
- `submission_pipeline.py` -> `zeta/submission/pipeline.py`;
- `persistence.py` -> `zeta/tui/persistence.py`.
- `types.py` -> `zeta/protocol/types.py`, last.

Update the exact source and test importers listed below. Keep the current root
`zeta` re-exports. Preserve the console entry point `zeta.cli:main`. Keep
explicit re-export shims for `zeta.loop`, `zeta.settings`, and `zeta.types`
until repository searches show zero importers. Remove each shim only after its
replacement passes the boundary suite and its importer count is zero. Do not
maintain two implementations.

The `zeta.cli` package uses this lazy export map in `cli/__init__.py`:

```python
_EXPORTS = {
    "build_parser": (".main", "build_parser"),
    "main": (".main", "main"),
    "create_app": (".main", "create_app"),
    "_cleanup_ephemeral": (".main", "_cleanup_ephemeral"),
    "_print_exit_hint": (".main", "_print_exit_hint"),
    "_run_login": (".main", "_run_login"),
}
```

Its `__getattr__` imports the selected relative module with
`importlib.import_module(module_name, __name__)`, then returns the named
attribute. It raises `AttributeError` for every name outside this map. The
`__all__` list remains `['build_parser', 'main']`.

Before moving `persistence.py`, replace `zeta.tui.__init__` with this lazy
export map:

```python
_EXPORTS = {
    "TUIApp": (".app", "TUIApp"),
    "main": ("..cli", "main"),
}
```

Use the same `__getattr__` behavior and preserve `__all__ = ["TUIApp", "main"]`.
Verify `import zeta.tui` alone does not import `zeta.tui.app` or `zeta.cli`;
then verify `from zeta.tui import TUIApp, main` in a separate fresh process.

The complete stage-4 importer rewrite inventory is below. It was regenerated
from the current importer grep and the stage-3 destination table. Split source
files are represented by every destination that retains the importer.

- `agent_background` -> `agent.background`: `agent/receipt.py` and
  `agent/runner.py`, `from .agent_background` -> `from .background`;
  `runtime/loop.py`, `from .agent_background` -> `from ..agent.background`;
  `src/zeta/tools/agent/tests/test_agent.py`,
  `src/zeta/tools/agent_send/tests/test_agent_send.py`, and
  `tests/test_agent_integration.py`, plus
  `src/zeta/tools/agent/tests/test_agent_output.py`,
  `tests/test_agent_output_tui.py`, `tests/test_session_shutdown.py`, and
  `src/zeta/tools/agent_send/tests/test_agent_send.py`, all
  `from zeta.agent_background` -> `from zeta.agent.background`.
- `agent_budget` -> `agent.budget`: `agent/runner.py`, `from .agent_budget` ->
  `from .budget`; `runtime/loop.py`, `from .agent_budget` ->
  `from ..agent.budget`; `src/zeta/tools/agent/tests/test_agent.py`,
  `src/zeta/tools/agent_send/tests/test_agent_send.py`, and
  `tests/test_agent_integration.py`, `from zeta.agent_budget` ->
  `from zeta.agent.budget`.
- `agent_receipt` -> `agent.receipt`: `agent/background.py` and
  `agent/runner.py`, `from .agent_receipt` -> `from .receipt`;
  `runtime/loop.py`, `from .agent_receipt` -> `from ..agent.receipt`;
  `core/store.py`, `tui/agent_card.py`, and `tui/render.py`, `from
  ..agent_receipt` -> `from ..agent.receipt`; `tools/agent/__init__.py`,
  `from ...agent_receipt` -> `from ...agent.receipt`;
  `src/zeta/tools/agent/tests/test_agent_output.py`,
  `tests/test_agent_output_tui.py`, and `tests/test_tool_ergonomics.py`,
  `from zeta.agent_receipt` -> `from zeta.agent.receipt`.
- `agent_runner` -> `agent.runner`: `runtime/loop.py`, `from .agent_runner` ->
  `from ..agent.runner`; `src/zeta/tools/agent/tests/test_agent.py`,
  `src/zeta/tools/agent_send/tests/test_agent_send.py`,
  `tests/test_agent_integration.py`, and `tests/test_agents.py`, all
  `zeta.agent_runner` imports and monkeypatch strings -> `zeta.agent.runner`.
  Their `from zeta import agent_runner` imports become `from zeta.agent import
  runner as agent_runner`.
- `execution` -> `runtime.execution`: `tools/memory/__init__.py` and
  `tools/route/__init__.py`, `from ...execution` ->
  `from ...runtime.execution`; `tools/registry.py`, `from ..execution` ->
  `from ..runtime.execution`; `src/zeta/tools/agent/tests/test_agent.py`,
  `src/zeta/tools/agent_send/tests/test_agent_send.py`, and
  `tests/test_agent_integration.py`, `import zeta.execution as execution_module`
  -> `import zeta.runtime.execution as execution_module`.
- `headless` -> `runtime.headless`: `cli/main.py`, `from .headless` ->
  `from ..runtime.headless`; tests `tests/test_headless.py`,
  `tests/test_session_shutdown.py`, and `tests/test_stream_watchdog.py`, all
  `zeta.headless` imports and monkeypatch strings -> `zeta.runtime.headless`.
- `images` -> `media.images`: `protocol/types.py`, `from .images` ->
  `from ..media.images`; `tools/read/__init__.py`, `from ...images` ->
  `from ...media.images`; `providers/anthropic_payload.py`,
  `providers/codex_payload.py`, `server/ergonomics.py`, and `tui/composer.py`,
  `from ..images` -> `from ..media.images`;
  `src/zeta/tools/read/tests/test_read_images.py`,
  `tests/test_read_images_integration.py`, and `tests/test_anthropic.py`,
  `from zeta.images` -> `from zeta.media.images`.
- `model_catalog` -> `models.catalog`: `agent/runner.py`, `from
  .model_catalog` -> `from ..models.catalog`; `providers/__init__.py`,
  `providers/factory.py`, `server/ergonomics.py`, `server/model_selection.py`,
  `skills/agent_catalog.py`, and `tui/models.py`, `from ..model_catalog` ->
  `from ..models.catalog`; `tools/agent/__init__.py`, `from ...model_catalog` ->
  `from ...models.catalog`; `src/zeta/tools/agent/tests/test_agent.py`,
  `src/zeta/tools/agent_send/tests/test_agent_send.py`,
  `tests/test_agent_integration.py`, `tests/test_model_picker.py`, and
  `tests/test_server.py`, `from zeta.model_catalog` ->
  `from zeta.models.catalog`.
- `persistence` -> `tui.persistence`: `tui/app.py`, `from ..persistence` ->
  `from .persistence`; `src/zeta/tools/agent/tests/test_session_safety.py`,
  `src/zeta/tools/exec/tests/test_session_safety.py`,
  `tests/test_session_safety_integration.py`, and `tests/test_tui.py`,
  `from zeta.persistence` -> `from zeta.tui.persistence`.
- `session_cli` -> `cli.session`: `cli/main.py`, both `from .session_cli` forms
  -> `from .session`; `src/zeta/tools/agent/tests/test_session_resilience.py`
  and `tests/test_session_resilience_integration.py`, `from zeta import
  session_cli` -> `from zeta.cli import session as session_cli`.
- `settings` -> `config.settings`: `automations/authoring.py`,
  `runtime/composition.py`, `runtime/unattended.py`, `server/runtime.py`,
  `tui/app.py`, and `tui/bootstrap.py`, each `from ..settings` -> `from
  ..config.settings`; tests `tests/test_jev_compaction.py`,
  `src/zeta/tools/memory/tests/test_memory_tools.py`,
  `tests/test_memory_tools_integration.py`, `tests/test_router_mode.py`,
  `tests/test_safety.py`, `tests/test_settings.py`, and
  `tests/test_stream_watchdog.py`,
  `from zeta.settings` -> `from zeta.config.settings`.
- `submission` and `submission_pipeline`: `submission/pipeline.py`, `from
  .submission` -> `from .model` and `from .types` -> `from
  ..protocol.types`; `tui/app.py`, `from ..submission_pipeline` -> `from
  ..submission.pipeline`; `tui/slash_handlers/command_runtime.py`, `from
  ...submission` -> `from ...submission.model`.
- moved implementation files also need these depth rewrites: `cli/main.py`,
  `.core.*` -> `..core.*`, `.providers.login` -> `..providers.login`, and
  `.tui.app` -> `..tui.app`; `cli/session.py`, `.core.session` ->
  `..core.session`; `runtime/execution.py`, `.core.*` -> `..core.*`;
  `runtime/headless.py`, `.core.*` -> `..core.*` and `.runtime.driver` ->
  `.driver`; `media/images.py`, `.types` -> `..protocol.types`;
  `agent/background.py`, `agent/runner.py`, and `agent/receipt.py`, their
  `.core.*` imports -> `..core.*`; `agent/runner.py`, `.providers.factory`,
  `.skills.agent_catalog`, and `.tools.*` -> `..providers.factory`,
  `..skills.agent_catalog`, and `..tools.*`; `tui/persistence.py`, `.core.*`
  -> `..core.*`; `submission/pipeline.py`, `.core.*` and `.tools.*` ->
  `..core.*` and `..tools.*`; and `runtime/loop.py`, every former root-level
  sibling import changes from `.name` to `..name`, while its `tools/*`,
  `mcp/*`, `prompts`, `providers`, and `skills` imports use `..` before the
  package name, and its current `.core.*` imports become `..core.*`.
- `loop` -> `runtime.loop`: `zeta/__init__.py`, `from .loop` ->
  `from .runtime.loop`; `agent/runner.py`, `from .loop` ->
  `from ..runtime.loop`; `core/loop.py`, `from ..loop` ->
  `from ..runtime.loop`; `runtime/cleanup.py`, `runtime/composition.py`,
  `runtime/driver.py`, and `runtime/unattended.py`, each `from ..loop` ->
  `from .loop`; `server/runtime.py` and `tui/app.py`, `from ..loop` ->
  `from ..runtime.loop`; `tests/zeta_test_plugin.py`,
  `tests/test_agent_status.py`, and all of `tests/test_agents.py`,
  `tests/test_anthropic.py`, `tests/test_checkpoint.py`,
  `tests/test_codex.py`, `tests/test_command_menu.py`,
  `tests/test_headless.py`, `tests/test_jev_compaction.py`, `tests/test_loop.py`,
  `tests/test_mcp.py`, `tests/test_mcp_oauth.py`, `tests/test_model_picker.py`,
  `tests/test_plan_mode.py`, `tests/test_read_images.py`,
  `tests/test_router_auto.py`, `tests/test_router_mode.py`,
  `tests/test_selection.py`, `tests/test_server.py`, `tests/test_slash.py`,
  `tests/test_stream_watchdog.py`, `tests/test_theme_and_keys.py`,
  `tests/test_tree.py`, `tests/test_user_tool_discovery.py`, and
  `tests/test_workspace_snapshots.py`, all `zeta.loop` imports and monkeypatch
  strings -> `zeta.runtime.loop`; split sources map as follows:
  `test_agent.py` -> `src/zeta/tools/agent/tests/test_agent.py`,
  `src/zeta/tools/agent_send/tests/test_agent_send.py`, and
  `tests/test_agent_integration.py`; `test_agent_output.py` ->
  `src/zeta/tools/agent/tests/test_agent_output.py` and
  `tests/test_agent_output_tui.py`; `test_commands.py` ->
  `src/zeta/tools/exec/tests/test_exec_commands.py` and `tests/test_commands.py`;
  `test_evals.py` -> `src/zeta/tools/route/tests/test_route_evals.py` and
  `tests/test_evals.py`; `test_session_shutdown.py` ->
  `src/zeta/tools/agent_send/tests/test_agent_send.py` and
  `tests/test_session_shutdown.py`; `test_todo.py` ->
  `src/zeta/tools/todo/tests/test_todo.py`, `tests/test_todo_persistence.py`,
  and `tests/test_todo_tui.py`; `test_read_images.py` ->
  `src/zeta/tools/read/tests/test_read_images.py` and
  `tests/test_read_images_integration.py`.

For `types`, update the following 62 source importers from their current
relative `types` path to `protocol.types`; the new relative form is shown by
source directory:

- `zeta/__init__.py`: `.types` -> `.protocol.types`;
  `agent/background.py`, `agent/budget.py`, `agent/receipt.py`, and
  `agent/runner.py`: `.types` -> `..protocol.types`;
  `automations/delivery.py`, `automations/runner.py`, `core/agent_state.py`,
  `core/approval.py`, `core/context.py`, `core/fake.py`, `core/slash.py`,
  `core/store.py`, `core/tool_dispatch.py`, `mcp/client.py`,
  `mcp/prompt_actor.py`, `mcp/server_actor.py`, `providers/anthropic.py`,
  `providers/anthropic_payload.py`, `providers/codex.py`,
  `providers/codex_payload.py`, `providers/factory.py`,
  `providers/transport.py`, `runtime/composition.py`, `runtime/driver.py`,
  `runtime/unattended.py`, `server/ergonomics.py`, `server/fake_backend.py`,
  `server/model_selection.py`, `server/runtime.py`, `server/server.py`,
  `tui/agent_card.py`, `tui/app.py`, `tui/checkpoints.py`, `tui/composer.py`,
  `tui/fake_backend.py`, `tui/render.py`,
  `tui/slash_handlers/command_runtime.py`, `tui/transcript.py`, and
  `tui/transcript_presenter.py`: `from ..types` ->
  `from ..protocol.types`;
  `core/checkpoints/__init__.py`, `from ...types` ->
  `from ...protocol.types`; `runtime/driver.py` and all other listed runtime
  files keep their directory-relative depth while changing only the module
  name; `runtime/execution.py`, `media/images.py`, `runtime/loop.py`,
  `submission/pipeline.py`, `agent/presets.py`, `runtime/tool_setup.py`,
  `tools/agent/__init__.py`, `tools/automation/__init__.py`,
  `tools/bash/__init__.py`, `tools/calendar/__init__.py`,
  `tools/edit/__init__.py`, `tools/exec/__init__.py`, `tools/fetch/__init__.py`,
  `tools/memory/__init__.py`, `tools/read/__init__.py`,
  `tools/registry.py`, `tools/route/__init__.py`, `tools/todo/__init__.py`,
  `tools/websearch/__init__.py`, `tools/write/__init__.py`, and
  `tools/zeta_background/__init__.py`: their current root-relative `types`
  import changes to the same depth under `protocol.types`.

The post-stage-3 test importer destinations are these. In each case,
`from zeta.types` becomes `from zeta.protocol.types`. Split source files list
every destination that retains the importer:

```text
src/zeta/tools/agent/tests/test_agent.py,
src/zeta/tools/agent_send/tests/test_agent_send.py,
tests/test_agent_integration.py,
src/zeta/tools/agent/tests/test_agent_output.py,
tests/test_agent_output_tui.py,
src/zeta/tools/agent/tests/test_agent_status.py,
src/zeta/tools/agent/tests/test_agents.py,
tests/test_anthropic.py, tests/test_approval.py, tests/test_attachments.py,
src/zeta/tools/agent/tests/test_automations.py,
src/zeta/tools/automation/tests/test_automation.py,
tests/test_automations_integration.py,
src/zeta/tools/zeta_background/tests/test_background.py,
src/zeta/tools/_shared/tests/test_process.py,
tests/test_background_integration.py,
src/zeta/tools/calendar/tests/test_calendar_tools.py, tests/test_checkpoint.py,
tests/test_codex.py, tests/test_context.py,
src/zeta/tools/exec/tests/test_exec_commands.py, tests/test_commands.py,
tests/test_evals.py, src/zeta/tools/route/tests/test_route_evals.py,
tests/test_headless.py, tests/test_hooks.py,
tests/test_jev_compaction.py, tests/test_loop.py, tests/test_mcp.py,
tests/test_mcp_oauth.py, src/zeta/tools/memory/tests/test_memory_tools.py,
tests/test_memory_tools_integration.py,
tests/test_plan_mode.py, tests/test_project_context.py,
src/zeta/tools/read/tests/test_read_images.py,
tests/test_read_images_integration.py,
tests/test_router_auto.py, tests/test_router_mode.py,
src/zeta/tools/bash/tests/test_bash_safety.py,
src/zeta/tools/exec/tests/test_exec_safety.py,
tests/test_safety.py,
src/zeta/tools/bash/tests/test_bash.py, src/zeta/tools/exec/tests/test_exec.py,
src/zeta/tools/edit/tests/test_edit.py, src/zeta/tools/write/tests/test_write.py,
src/zeta/tools/_shared/tests/test_sandbox.py, tests/test_selection.py,
tests/test_server.py, tests/test_server_login.py,
tests/test_session_lifecycle.py,
src/zeta/tools/agent/tests/test_session.py, tests/test_session_integration.py,
src/zeta/tools/agent/tests/test_session_resilience.py,
tests/test_session_resilience_integration.py,
src/zeta/tools/agent/tests/test_session_safety.py,
src/zeta/tools/exec/tests/test_session_safety.py,
tests/test_session_safety_integration.py,
tests/test_session_shutdown.py,
tests/test_skills.py,
src/zeta/tools/skill/tests/test_skill_tool.py,
tests/test_slash.py, tests/test_steering.py, tests/test_store.py,
tests/test_stream_watchdog.py, src/zeta/tools/todo/tests/test_todo.py,
tests/test_todo_persistence.py, tests/test_todo_tui.py,
src/zeta/tools/fetch/tests/test_fetch.py,
src/zeta/tools/websearch/tests/test_websearch.py,
src/zeta/tools/read/tests/test_read.py,
tests/test_tool_discovery.py, tests/test_tool_ergonomics.py,
tests/test_tool_result_shape.py, tests/test_tools_integration.py,
tests/test_transcript_paint.py, tests/test_tree.py, tests/test_tui.py,
tests/test_types.py, tests/test_user_tool_discovery.py,
tests/test_workspace_snapshots.py
```

For `cli.py`, the package-level `from zeta.cli import ...` imports in
`test_automations.py`, `test_cli.py`, `test_headless.py`, `test_hooks.py`,
`test_jev_compaction.py`, `test_login.py`, `test_plan_mode.py`,
`test_project_context.py`, `test_router_mode.py`, `test_safety.py`,
`test_server.py`, `test_session.py`, `test_session_lifecycle.py`,
`test_session_safety.py`, `test_session_shutdown.py`, `test_settings.py`, and
`test_tui.py` remain unchanged because `zeta.cli` is the new compatibility
package. The `tui/__init__.py` and `tui/app.py` `from ..cli import main`
imports also remain package imports. Only `cli/main.py` changes its internal
relative imports as listed above.

The `types.py` move is shim-first and last. Create empty
`zeta/protocol/__init__.py`, move the implementation to
`zeta/protocol/types.py`, and immediately replace the old `zeta/types.py` with
a one-way shim before changing any importer. The shim re-exports the single
implementation with `from .protocol.types import *`. The current module has no
`__all__`, so do not add one as part of the move. Then apply every `types` rewrite below. Verify in
separate fresh processes that both `import zeta.types` and
`import zeta.protocol.types` expose the same named objects, that
`from zeta import TextContent` still works, and that the boundary suite passes.
Run this full old-path grep after the rewrite, excluding the shim itself:

```sh
rg -n --glob '*.py' --glob '!src/zeta/types.py' \
  'zeta\.types|from \.{1,3}types\b' harness/src harness/tests
```

Remove the shim only when this command finds no importer and the new-path
import checks pass.

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
  `zeta.models/catalog`, `zeta.submission/{model,pipeline}`;
- gate: each grouping initializer is empty or lazy before its first child;
  `settings` and `loop` use one-way shims during migration;
- verdict: cycle-free after importer rewrites and shim checks.
- module moved: `zeta.tui/persistence`;
- gate: `zeta.tui.__init__` is lazy before the move because the old initializer
  eagerly imported `.app`;
- verdict: cycle-free.
- runtime modules moved: `execution`, `headless`, and `loop`;
  - gate: `zeta.runtime.__init__` remains lazy throughout, and each fresh
  process import passes before the next runtime child move;
  - verdict: cycle-free with runtime composition and registry.
- module moved last: `zeta.protocol/types`;
  - gate: the old `zeta.types` shim exists before importer rewrites, both old
  and new imports expose the same objects, and the old-path grep is empty
  before shim removal;
  - verdict: cycle-free with the root re-exports and all 116 importer files.

The targeted parity gate runs before and after each sub-task:

```sh
cd harness && uv run --frozen pytest -q \
  tests/test_import_boundaries.py tests/test_cli.py tests/test_headless.py \
  src/zeta/tools/agent/tests/test_agent.py \
  src/zeta/tools/agent/tests/test_agent_output.py \
  src/zeta/tools/agent/tests/test_agent_status.py \
  src/zeta/tools/agent/tests/test_agents.py \
  src/zeta/tools/agent_send/tests/test_agent_send.py \
  src/zeta/tools/agent/tests/test_session.py \
  src/zeta/tools/agent/tests/test_session_resilience.py \
  src/zeta/tools/agent/tests/test_session_safety.py \
  src/zeta/tools/exec/tests/test_session_safety.py \
  tests/test_agent_integration.py tests/test_session_integration.py \
  tests/test_session_resilience_integration.py \
  tests/test_session_safety_integration.py tests/test_session_shutdown.py \
  tests/test_agent_output_tui.py tests/test_settings.py tests/test_types.py \
  tests/test_loop.py tests/test_checkpoint.py \
  src/zeta/tools/exec/tests/test_exec_commands.py tests/test_commands.py \
  tests/test_mcp.py tests/test_mcp_oauth.py tests/test_server.py tests/test_tui.py \
  tests/test_anthropic.py tests/test_codex.py tests/test_model_picker.py \
  src/zeta/tools/automation/tests src/zeta/tools/browser/tests \
  src/zeta/tools/calendar/tests src/zeta/tools/memory/tests \
  src/zeta/tools/read/tests src/zeta/tools/websearch/tests \
  src/zeta/tools/fetch/tests src/zeta/tools/skill/tests \
  src/zeta/tools/route/tests src/zeta/tools/bash/tests \
  src/zeta/tools/exec/tests src/zeta/tools/edit/tests \
  src/zeta/tools/write/tests src/zeta/tools/zeta_background/tests \
  tests/test_automations_integration.py tests/test_memory_tools_integration.py \
  tests/test_read_images_integration.py tests/test_tools_integration.py \
  tests/test_evals.py tests/test_safety.py tests/test_skills.py \
  src/zeta/tools/todo/tests/test_todo.py tests/test_todo_persistence.py \
  tests/test_todo_tui.py tests/test_background_integration.py \
  src/zeta/tools/_shared/tests/test_sandbox.py \
  src/zeta/tools/_shared/tests/test_process.py
```

Also run the stage-3 post-move directories in targeted batches:
`src/zeta/tools/automation/tests`, `browser/tests`, `calendar/tests`,
`memory/tests`, `read/tests`, `websearch/tests`, `fetch/tests`, `skill/tests`,
`route/tests`, `bash/tests`, `exec/tests`, `edit/tests`, `write/tests`, and
`zeta_background/tests`, plus `tests/test_automations_integration.py`,
`tests/test_memory_tools_integration.py`, `tests/test_read_images_integration.py`,
`tests/test_tools_integration.py`, `tests/test_evals.py`,
`tests/test_safety.py`, and `tests/test_skills.py`. These are the exact
post-stage-3 replacements for the central paths moved by stage 3.

Run package import smoke checks for every target package in a new Python
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
