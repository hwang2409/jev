# jev-zeta: harness source reorg design

Date: 2026-09-22  
Status: proposed design for review  
Scope: `harness/src/zeta/` source layout and tool-test placement

## 1. goal and non-goals

The goal is a behavior-preserving reorganization of the Python harness.
Tools get one directory each. Loose top-level modules move into cohesive
subpackages. Tool tests move beside the tool that owns them.

The reorg must preserve:

- tool names, schemas, handler behavior, approval behavior, and result shapes;
- the `zeta.tools.<name>` import path for each tool package;
- built-in tool discovery through `pkgutil.iter_modules`;
- the synchronous `register(registry)` contract;
- the existing `harness/tests` suite during the migration;
- the console entry point `zeta.cli:main` until the final CLI package move.

This design does not change tool logic, provider behavior, storage formats,
public tool APIs, test assertions, or the browser feature scope. It does not
write the implementation plan. It does not move files in this change.

An import path may change when its module moves. During migration, a short
compatibility shim may keep the old path working. The final layout removes
shims only after all in-repo importers use the new path.

## 2. current layout and findings

### 2.1 package shape

The project uses a `src` layout:

```text
harness/
  pyproject.toml
  src/zeta/
  tests/
```

`harness/pyproject.toml` sets Hatchling's wheel package to `src/zeta`.
Pytest has `testpaths = ["tests"]`, `asyncio_mode = "auto"`, and no explicit
`pythonpath`. The editable project install makes `zeta` importable in the
current test command, `uv run pytest -q`.

The loose top-level modules are 16 files:

```text
agent_background.py  agent_budget.py      agent_receipt.py
agent_runner.py      cli.py               execution.py
headless.py          images.py            loop.py
model_catalog.py     persistence.py       session_cli.py
settings.py          submission.py        submission_pipeline.py
types.py
```

The existing non-tool subpackages are:

- `core/`: session state, safety, commands, context, and tool dispatch;
- `providers/`: provider clients, payloads, auth, and transport;
- `runtime/`: composition, drivers, cleanup, and unattended execution;
- `server/`: the local server protocol and runtime;
- `tui/`: the terminal UI and transcript rendering;
- `mcp/`: MCP clients, mounts, resources, and server actors;
- `automations/`: scheduled jobs and delivery;
- `prompts/`: packaged identity prompt;
- `skills/`: skill discovery, loading, and agent catalog.

### 2.2 current tools

There are 13 registered single-file tool modules:

```text
agent.py       bash.py       calendar.py   edit.py
exec.py        fetch.py      memory.py     read.py
route.py       skill.py      todo.py      websearch.py
write.py
```

There are four existing directories:

```text
agent_send/__init__.py       automation/__init__.py
plan_mode/__init__.py        zeta_background/__init__.py
```

`automation` and `zeta_background` already expose `register()`. `agent_send`
is a support package with `register_send()`; `agent.py` calls it from its own
`register()`. `plan_mode` is configuration, not a discovered tool. These two
exceptions matter when the layout is normalized.

The remaining files in `tools/` are:

```text
_process.py       _sandbox.py       _user_discovery.py
agent_presets.py  loop_setup.py     registry.py
browser_adapter.py                 browser_catalog.py
```

The first three are shared helpers. `registry.py`, `agent_presets.py`, and
`loop_setup.py` are framework support. The browser files are the merged
arc-3 foundation, not registered tools yet.

### 2.3 discovery contract

`tools/registry.py` imports the `zeta.tools` package and runs:

```python
modules = (
    f"{package.__name__}.{module_info.name}"
    for module_info in pkgutil.iter_modules(package.__path__)
    if not module_info.name.startswith("_")
)
```

It sorts the resulting names with:

```python
sorted(modules, key=lambda name: (name.endswith(".agent"), name))
```

It imports every result. A module without a callable `register` is skipped.
A callable `register(registry)` must be synchronous. A package directory is
already a valid `pkgutil` result. Its `__init__.py` is the imported module,
so the import path stays `zeta.tools.<name>`.

The four current directories show that package discovery works. Only
`automation` and `zeta_background` currently register directly. The target
layout makes every actual tool package use the same direct contract.

The leading-underscore rule is also a contract. A new `tools/_shared/`
package remains invisible to the top-level discovery scan. Its nested modules
are imported only by explicit relative imports.

### 2.4 coupling map for tools

The following inventory counts distinct importer files, not import statements.
Paths are relative to `harness/`. `src` importers use relative imports within
the package. Test importers use absolute `zeta.tools...` imports. A tool with
no listed importer is reached through dynamic built-in discovery.

```text
module                         src importers (count)                         test importers (count)
zeta.tools._process            mcp/stdio.py, tools/bash.py, tools/exec.py,   test_background.py, test_session_safety.py,
                               tools/memory.py, tools/registry.py (5)        test_tools.py (3)
zeta.tools._sandbox            tools/bash.py, edit.py, read.py, registry.py,  test_sandbox.py, test_tools.py (2)
                               write.py, zeta_background/__init__.py (6)
zeta.tools._user_discovery     runtime/composition.py, tui/app.py,          test_user_tool_discovery.py (1)
                               tui/slash_handlers/__init__.py (3)
zeta.tools.agent               agent_runner.py, loop.py, tui/agent_card.py  test_agent.py, test_agent_output.py,
                               (3)                                           test_agent_status.py, test_automations.py,
                                                                            test_session.py, test_session_resilience.py,
                                                                            test_session_safety.py (7)
zeta.tools.agent_presets       agent_runner.py, loop.py, skills/agent_catalog.py,
                               tools/agent.py, tools/plan_mode/__init__.py,
                               tui/agent_card.py (6)                        test_agent.py (1)
zeta.tools.bash                dynamic discovery only (0)                  none (0)
zeta.tools.browser_adapter     tools/browser_catalog.py (1)                test_browser_adapter.py, test_browser_catalog.py (2)
zeta.tools.browser_catalog     none (0)                                     test_browser_catalog.py, test_browser_prefilter.py (2)
zeta.tools.calendar            tools/route.py (1)                           test_calendar_tools.py (1)
zeta.tools.edit                dynamic discovery only (0)                  none (0)
zeta.tools.exec                submission_pipeline.py, tui/app.py,         test_commands.py, test_safety.py,
                               tui/render.py, tui/slash_handlers/           test_session_safety.py, test_tools.py (4)
                               command_runtime.py (4)
zeta.tools.fetch               tools/websearch.py (1)                      test_webtools.py (1)
zeta.tools.loop_setup          loop.py (1)                                  none (0)
zeta.tools.memory              loop.py, tools/route.py (2)                  test_memory_tools.py, test_router_auto.py (2)
zeta.tools.read                none (0)                                     test_read_images.py, test_sandbox.py, test_tools.py (3)
zeta.tools.registry            27 source files; see the full list below    test_calendar_tools.py, test_evals.py,
                                                                            test_router_auto.py, test_router_mode.py,
                                                                            test_tool_ergonomics.py,
                                                                            test_tool_result_shape.py (6)
zeta.tools.route                loop.py (1)                                  test_evals.py, test_memory_tools.py,
                                                                            test_router_auto.py, test_router_mode.py (4)
zeta.tools.skill               dynamic discovery only (0)                  none (0)
zeta.tools.todo                 dynamic discovery only (0)                  test_todo.py (1)
zeta.tools.websearch            dynamic discovery only (0)                  test_webtools.py (1)
zeta.tools.write               none (0)                                     test_tools.py (1)
zeta.tools.agent_send          tools/agent.py (1)                           test_agent.py, test_session_shutdown.py (2)
zeta.tools.automation          dynamic discovery only (0)                  none (0)
zeta.tools.plan_mode           loop.py (1)                                  test_plan_mode.py (1)
zeta.tools.zeta_background     dynamic discovery only (0)                  none (0)
```

Rebuild this inventory at implementation time with these exact searches:

```sh
rg -n --glob '*.py' 'from zeta\.|import zeta\.|from zeta\.tools\.' harness/src harness/tests
rg -n --glob '*.py' '^\s*(from|import) ' harness/src harness/tests
rg -n --glob '*.py' 'from zeta import ' harness/src harness/tests
```

Count each distinct importer file after resolving relative imports and
`from zeta import <module>` aliases to the module they import. The first
search covers dotted absolute importers. The second search covers relative
source imports and verifies the complete import surface. The third search
covers root-package aliases, including `agent_runner` and `session_cli`.
Update this table from the output, not from memory or a prior table.

The registry source importer list is:

```text
agent_receipt.py, agent_runner.py, loop.py,
mcp/client.py, mcp/mount.py, mcp/server_actor.py,
runtime/composition.py, tools/__init__.py, tools/_sandbox.py,
tools/_user_discovery.py, tools/agent.py, tools/agent_send/__init__.py,
tools/automation/__init__.py, tools/bash.py, tools/calendar.py,
tools/edit.py, tools/exec.py, tools/fetch.py, tools/loop_setup.py,
tools/memory.py, tools/read.py, tools/route.py, tools/skill.py,
tools/todo.py, tools/websearch.py, tools/write.py,
tools/zeta_background/__init__.py
```

The important cross-tool edges are:

- `route.py` imports catalog helpers from `calendar.py` and `memory.py`;
- `websearch.py` imports response helpers from `fetch.py`;
- `browser_catalog.py` imports value types from `browser_adapter.py`;
- `agent.py` imports `agent_send` and `agent_presets`;
- `read.py`, `write.py`, `edit.py`, `bash.py`, and `registry.py` use sandbox
  helpers;
- `bash.py`, `exec.py`, `memory.py`, and `registry.py` use process helpers;
- `registry.py` owns dynamic discovery, so registered modules do not need
  static importers.

### 2.5 coupling map for loose top-level modules

The proposed target paths and importer counts are below. Counts include source
and test files. The exact importer names are listed so the later move can be
checked against this baseline. Source paths use the current relative import
form. Test paths use `from zeta.<module> import ...` or `import zeta.<module>`.

```text
current module       target path                    source importers (count)                         test importers (count)
agent_background     zeta.agent.background          agent_receipt.py, agent_runner.py, loop.py (3)    test_agent.py, test_agent_output.py, test_session_shutdown.py (3)
agent_budget         zeta.agent.budget             agent_runner.py, loop.py (2)                      test_agent.py (1)
agent_receipt        zeta.agent.receipt            agent_background.py, agent_runner.py,             test_agent_output.py, test_tool_ergonomics.py (2)
                                                     core/store.py, loop.py, tools/agent.py,
                                                     tui/agent_card.py, tui/render.py (7)
agent_runner         zeta.agent.runner             loop.py (1)                                        test_agent.py, test_agents.py (2)
cli                  zeta.cli.main                tui/__init__.py, tui/app.py (2)                   test_automations.py, test_cli.py,
                                                                                                      test_headless.py, test_hooks.py,
                                                                                                      test_jev_compaction.py, test_login.py,
                                                                                                      test_plan_mode.py, test_project_context.py,
                                                                                                      test_router_mode.py, test_safety.py,
                                                                                                      test_server.py, test_session.py,
                                                                                                      test_session_lifecycle.py, test_session_safety.py,
                                                                                                      test_session_shutdown.py, test_settings.py,
                                                                                                      test_tui.py (17)
execution            zeta.runtime.execution       tools/memory.py, tools/registry.py,               test_agent.py (1)
                                                     tools/route.py (3)
headless             zeta.runtime.headless        cli.py (1)                                         test_headless.py, test_session_shutdown.py,
                                                                                                      test_stream_watchdog.py (3)
images               zeta.media.images            providers/anthropic_payload.py,                   test_anthropic.py, test_read_images.py (2)
                                                     providers/codex_payload.py, server/ergonomics.py,
                                                     tools/read.py, tui/composer.py, types.py (6)
loop                 zeta.runtime.loop            zeta/__init__.py, agent_runner.py, core/loop.py,
                                                     runtime/cleanup.py, runtime/composition.py,
                                                     runtime/driver.py, runtime/unattended.py,
                                                     server/runtime.py, tui/app.py (9)                conftest.py, test_agent.py,
                                                                                                      test_agent_output.py, test_agent_status.py,
                                                                                                      test_agents.py, test_anthropic.py,
                                                                                                      test_checkpoint.py, test_codex.py,
                                                                                                      test_command_menu.py, test_commands.py,
                                                                                                      test_evals.py, test_headless.py,
                                                                                                      test_jev_compaction.py, test_loop.py,
                                                                                                      test_mcp.py, test_mcp_oauth.py,
                                                                                                      test_model_picker.py, test_plan_mode.py,
                                                                                                      test_read_images.py, test_router_auto.py,
                                                                                                      test_router_mode.py, test_selection.py,
                                                                                                      test_server.py, test_session_shutdown.py,
                                                                                                      test_slash.py, test_stream_watchdog.py,
                                                                                                      test_theme_and_keys.py, test_todo.py,
                                                                                                      test_tree.py, test_user_tool_discovery.py,
                                                                                                      test_workspace_snapshots.py (31)
model_catalog        zeta.models.catalog          agent_runner.py, providers/__init__.py,             test_agent.py, test_model_picker.py,
                                                     providers/factory.py, server/ergonomics.py,       test_server.py (3)
                                                     server/model_selection.py, skills/agent_catalog.py,
                                                     tools/agent.py, tui/models.py (8)
persistence          zeta.tui.persistence          tui/app.py (1)                                     test_session_safety.py, test_tui.py (2)
session_cli          zeta.cli.session             cli.py (1)                                          test_session_resilience.py (1)
settings             zeta.config.settings         automations/authoring.py, runtime/composition.py,   test_jev_compaction.py, test_memory_tools.py,
                                                     runtime/unattended.py, server/runtime.py,         test_router_mode.py, test_safety.py,
                                                     tui/app.py, tui/bootstrap.py (6)                  test_settings.py, test_stream_watchdog.py (6)
submission           zeta.submission.model        submission_pipeline.py,                           none (0)
                                                     tui/slash_handlers/command_runtime.py (2)
submission_pipeline  zeta.submission.pipeline     tui/app.py (1)                                     none (0)
types                zeta.protocol.types          62 source files; see the full list below          54 test files; see the full list below
```

The largest risks are `types` (116 importer files), `loop` (40), `cli` (19),
and `settings` (12). The counts for `types` and `loop` include the package
root and integration tests. They require compatibility shims during the move.

The `types` source importers are:

```text
zeta/__init__.py, agent_background.py, agent_budget.py, agent_receipt.py,
agent_runner.py, automations/delivery.py, automations/runner.py,
core/agent_state.py, core/approval.py, core/checkpoints/__init__.py,
core/context.py, core/fake.py, core/slash.py, core/store.py,
core/tool_dispatch.py, execution.py, images.py, loop.py, mcp/client.py,
mcp/prompt_actor.py, mcp/server_actor.py, providers/anthropic.py,
providers/anthropic_payload.py, providers/codex.py, providers/codex_payload.py,
providers/factory.py, providers/transport.py, runtime/composition.py,
runtime/driver.py, runtime/unattended.py, server/ergonomics.py,
server/fake_backend.py, server/model_selection.py, server/runtime.py,
server/server.py, submission_pipeline.py, tools/agent.py,
tools/agent_presets.py, tools/automation/__init__.py, tools/bash.py,
tools/calendar.py, tools/edit.py, tools/exec.py, tools/fetch.py,
tools/loop_setup.py, tools/memory.py, tools/read.py, tools/registry.py,
tools/route.py, tools/todo.py, tools/websearch.py, tools/write.py,
tools/zeta_background/__init__.py, tui/agent_card.py, tui/app.py,
tui/checkpoints.py, tui/composer.py, tui/fake_backend.py, tui/render.py,
tui/slash_handlers/command_runtime.py, tui/transcript.py,
tui/transcript_presenter.py
```

The 54 `types` test importers are:

```text
test_agent.py, test_agent_output.py, test_agent_status.py, test_agents.py,
test_anthropic.py, test_approval.py, test_attachments.py, test_automations.py,
test_background.py, test_calendar_tools.py, test_checkpoint.py, test_codex.py,
test_commands.py, test_context.py, test_evals.py, test_headless.py,
test_hooks.py, test_jev_compaction.py, test_loop.py, test_mcp.py,
test_mcp_oauth.py, test_memory_tools.py, test_plan_mode.py,
test_project_context.py, test_read_images.py, test_router_auto.py,
test_router_mode.py, test_safety.py, test_sandbox.py, test_selection.py,
test_server.py, test_server_login.py, test_session.py,
test_session_lifecycle.py, test_session_resilience.py, test_session_safety.py,
test_session_shutdown.py, test_skills.py, test_slash.py, test_steering.py,
test_store.py, test_stream_watchdog.py, test_todo.py, test_tool_discovery.py,
test_tool_ergonomics.py, test_tool_result_shape.py, test_tools.py,
test_transcript_paint.py, test_tree.py, test_tui.py, test_types.py,
test_user_tool_discovery.py, test_webtools.py, test_workspace_snapshots.py
```

No loose top-level module uses dynamic import discovery. The dynamic import
contract is limited to built-in tools and external project tools.

## 3. target tools layout

### 3.1 one package per tool

Every discovered tool gets one of these shapes:

Simple tool:
```text
tools/<name>/
  __init__.py       # implementation, stable API, and register(registry)
  tests/
    test_<name>.py  # tool-private tests after stage 3
```

Multi-module tool:
```text
tools/<name>/
  __init__.py       # stable package API and register(registry)
  <internal>.py     # only when the tool has multiple real modules
  tests/
    test_<name>.py  # tool-private tests after stage 3
```

Simple tools move their current implementation directly into `__init__.py`.
The reorganization does not add a thin wrapper or a mandatory `impl.py`.
This preserves existing module-global patch seams, including the seams used
by calendar, memory, read, write, and exec tests. An internal module appears
only when the tool genuinely has multiple modules.

For a multi-module tool, `__init__.py` re-exports names that existing callers
use. For example:

```python
from .adapter import BrowserTimeoutError
from .catalog import SnapshotCatalogBuilder

__all__ = ["BrowserTimeoutError", "SnapshotCatalogBuilder", "register"]
```

The exact public names come from the current modules. The example is
illustrative. It does not add a new API.

The module-level path `zeta.tools.<name>` remains the package path. Existing
imports such as `from zeta.tools.exec import run_exec_macro` continue to work
because simple tool implementations remain in `__init__.py`.

The only planned tool-internal split is the browser foundation. Its tests
change `zeta.tools.browser_adapter` imports to
`zeta.tools.browser.adapter`, and `zeta.tools.browser_catalog` imports to
`zeta.tools.browser.catalog`. Any monkeypatch target for those old modules
changes to the matching submodule. The shared `process` and `sandbox` helper
moves also update every test import: `test_tools.py`, `test_background.py`,
`test_sandbox.py`, and `test_session_safety.py` use
`zeta.tools._shared.process` or `zeta.tools._shared.sandbox`.

No patch-seam changes are needed for the simple tools. In particular,
`test_calendar_tools.py` keeps patching `zeta.tools.calendar`;
`test_memory_tools.py` keeps patching `zeta.tools.memory`, and
`test_router_auto.py` keeps importing it; the read, write, and exec imports
and patch targets in `test_tools.py`, `test_read_images.py`, `test_commands.py`,
`test_safety.py`, and `test_session_safety.py` remain on their package paths.

### 3.2 representative before and after layouts

Simple tool, `read`:

```text
before:
tools/read.py

after:
tools/read/
  __init__.py
  tests/
    test_read.py
```

Helper-heavy shell tool, `bash`:

```text
before:
tools/bash.py
tools/_process.py
tools/_sandbox.py

after:
tools/bash/
  __init__.py
  tests/
    test_bash.py
tools/_shared/
  __init__.py
  process.py
  sandbox.py
  user_discovery.py
```

Stage 1 runs before `_shared/` exists. The moved package initializers therefore
use interim parent imports: `bash/__init__.py` and `exec/__init__.py` use
`from .._process` and `from .._sandbox`; `memory/__init__.py` uses
`from .._process`; and `read`, `write`, and `edit` use `from .._sandbox`.
The package initializers for `agent`, `calendar`, `fetch`, `route`, `skill`,
`todo`, and `websearch` use `from ..registry` for the registry and `route`
uses `from ..calendar` and `from ..memory`. `websearch` uses `from ..fetch`.
The `agent` package uses `from ..agent_presets` and `from ..agent_send` for
the support packages that remain at the tools root during stage 1.
The browser catalog uses `from .adapter` after the browser move.

Stage 2 creates `_shared/` and rewrites every interim helper import to
`from .._shared.process`, `from .._shared.sandbox`, or the matching
`_shared.user_discovery` path. The root `registry.py` changes its helper
imports from `._process` and `._sandbox` to `._shared.process` and
`._shared.sandbox`. No stage may
leave a moved package importing a helper at the wrong relative depth.
The helpers are not copied into each tool directory.

The existing browser foundation:

```text
before:
tools/browser_adapter.py
tools/browser_catalog.py
tests/test_browser_adapter.py
tests/test_browser_catalog.py
tests/test_browser_prefilter.py

after:
tools/browser/
  __init__.py
  adapter.py
  catalog.py
  tests/
    test_adapter.py
    test_catalog.py
    test_prefilter.py
```

`tools/browser/__init__.py` will own the future browser `register()` function
and re-export its internal modules. Until browser handlers exist, it may
expose only the foundation types and no registration. Arc-3 tasks 5-12 add
the handlers under this package. Task 12, the real Playwright adapter, builds
in `tools/browser/` with co-located browser tests. The adapter and catalog
remain separate modules because they are different seams, not one large
handler implementation.

The other registered tools use the simple package shape:

```text
agent/         calendar/       edit/       exec/
fetch/         memory/         read/       route/
skill/         todo/           websearch/ write/
automation/    zeta_background/
```

`agent_send/` remains a separate support package, but it is not a discovered
tool package. It keeps its implementation and `register_send()` in
`__init__.py`. `agent.register()` continues to call `register_send()` after
registering `agent`, `agent_status`, and `agent_output`. This preserves the
current registration order without a discovery wrapper.

`plan_mode/` is not a tool. It moves with framework support in stage 2 rather
than becoming a fake discovered tool.

## 4. shared helpers and non-tool support

### 4.1 shared helpers

The target shared helper package is:

```text
tools/_shared/
  __init__.py
  process.py          # current _process.py
  sandbox.py          # current _sandbox.py
  user_discovery.py   # current _user_discovery.py
```

The leading underscore on `_shared` keeps it out of
`pkgutil.iter_modules(zeta.tools.__path__)`. The nested names are not scanned
by the built-in discovery function. Explicit imports use paths such as:

```python
from .._shared.process import tool_subprocess_env
from .._shared.sandbox import open_target
from zeta.tools._shared.user_discovery import ExternalToolDiscovery
```

The last form is a private framework import. It is updated in the runtime and
TUI callers during stage 2. The helper modules keep their current behavior.

### 4.2 framework support

The support placement is intentionally mixed:

- keep `tools/registry.py` at the `zeta.tools.registry` path;
- move `tools/agent_presets.py` to `zeta.agent.presets`;
- move `tools/loop_setup.py` to `zeta.runtime.tool_setup`;
- move `tools/plan_mode/` to `zeta.agent.plan_mode`.

`registry.py` is framework code, but it is also the discovery anchor. Keeping
it at the tools package root preserves the central import path and avoids
adding a second package scan target. Its direct fan-out is 33 importer files,
so a separate registry package would add risk without improving the target
tree.

Agent presets and plan mode belong with agent policy. Tool setup belongs with
runtime composition. Their moves remove non-tools from the discovered tools
namespace. Their new paths are internal framework paths, so all in-repo
importers will update in stage 2.

## 5. top-level grouping proposal

The target groups are:

```text
zeta/agent/
  __init__.py          # empty or lazy before any child move
  background.py       # agent_background.py
  budget.py           # agent_budget.py
  receipt.py          # agent_receipt.py
  runner.py           # agent_runner.py
  presets.py          # tools/agent_presets.py
  plan_mode.py        # tools/plan_mode/

zeta/cli/
  __init__.py         # lazy compatibility exports and console entry point
  main.py             # cli.py
  session.py          # session_cli.py

zeta/config/
  __init__.py         # empty or lazy before settings.py moves
  settings.py          # settings.py

zeta/media/
  __init__.py         # empty or lazy before images.py moves
  images.py            # images.py

zeta/models/
  __init__.py         # empty or lazy before catalog.py moves
  catalog.py            # model_catalog.py

zeta/protocol/
  __init__.py         # empty or lazy before types.py moves
  types.py              # types.py

zeta/runtime/
  __init__.py            # lazy exports; no eager composition or registry imports
  execution.py          # execution.py
  headless.py           # headless.py
  loop.py               # loop.py
  tool_setup.py         # tools/loop_setup.py

zeta/submission/
  __init__.py         # empty or lazy before child moves
  model.py              # submission.py
  pipeline.py           # submission_pipeline.py

zeta/tui/
  persistence.py        # persistence.py
```

The grouping follows the import graph:

- `agent/` contains agent lifecycle, budgets, receipts, runners, presets,
  and plan policy. These modules already form a dense agent-only cluster.
- `cli/` contains user-facing parser and session subcommands. `headless.py`
  is runtime execution, so it joins `runtime/`.
- `config/` contains settings resolution and approval configuration.
- `media/` owns image validation and byte-level image helpers.
- `models/` owns provider model metadata and lookup.
- `protocol/` owns shared message, content, stream, and tool types.
- `runtime/` contains the loop, tool execution context, headless driver, and
  tool-registry setup. It aligns with the existing runtime package. Its
  `__init__.py` must stay lazy: it cannot eagerly import `composition`, which
  imports `tools.registry`, or any child that imports the registry. Preserve
  the current public names with module-level lazy attribute loading if needed.
- `submission/` contains the immutable submission value and its pipeline.
- `tui/persistence.py` is used only by the TUI application and its tests.

The old `zeta.cli` path becomes a package. `zeta/cli/__init__.py` re-exports
`main`, `build_parser`, and other current public names. The project script can
keep `zeta.cli:main`, while implementation imports use `zeta.cli.main`.

The old top-level names for the other modules are moved to their target paths.
During the stage 4 transition, each old module may be a re-export shim. The
shim is removed after the importer count reaches zero. The package root keeps
the current `zeta` re-exports for names that it currently exposes.

The package-initializer invariant for every move is: no module moves under a
package until that package's `__init__.py` is lazy, or the package is proven
cycle-free with the newcomer. This prevents a package initializer from
eagerly loading a module that imports the newcomer back through the package.
The audit is:

```text
stage 1  tools/agent, bash, browser/{adapter,catalog}, calendar, edit, exec,
         fetch, memory, read, route, skill, todo, websearch, write
         gate: zeta.tools.__init__ imports only registry. Registry does not
         import discovered children during package import. Each new tool
         package has an import-free __init__ until its implementation is
         moved into it. Result: cycle-free with the newcomer.

stage 2  tools/_shared/{process,sandbox,user_discovery}
         gate: _shared/__init__.py is empty. Result: cycle-free.

stage 2  zeta.agent/{presets,plan_mode}
         gate: agent/__init__.py is empty or lazy before either move.
         Result: cycle-free.

stage 2  zeta.runtime/tool_setup
         gate: runtime/__init__.py becomes lazy before this move. Result:
         cycle-free; this is the runtime child that exposed the old cycle.

stage 4  zeta.agent/{background,budget,receipt,runner};
         zeta.cli/{main,session}; zeta.config/settings; zeta.media/images;
         zeta.models/catalog; zeta.protocol/types;
         zeta.runtime/{execution,headless,loop};
         zeta.submission/{model,pipeline}; zeta.tui/persistence
         gate: each new grouping initializer is empty or lazy before its
         first child moves. runtime remains lazy. Before moving persistence,
         make zeta.tui.__init__ lazy: the current initializer eagerly imports
         .app, which imports persistence. The lazy initializer is the gate.
         Result: every listed move passes the invariant.
```

`zeta.runtime.tool_setup` is a stage 2 move. Stage 4 does not move it again;
stage 4 only verifies it before moving the other runtime children.

### 5.1 move table and risk decisions

```text
move                         import path change                         affected files   decision
agent_background             zeta.agent_background ->                   6                move in stage 4
                              zeta.agent.background
agent_budget                 zeta.agent_budget -> zeta.agent.budget     3                move in stage 4
agent_receipt                zeta.agent_receipt -> zeta.agent.receipt   9                move in stage 4
agent_runner                 zeta.agent_runner -> zeta.agent.runner     3                move in stage 4
cli                          zeta.cli module -> zeta.cli.main           19               move as package; keep __init__ export
execution                    zeta.execution -> zeta.runtime.execution   4                move in stage 4
headless                     zeta.headless -> zeta.runtime.headless     4                move in stage 4
images                       zeta.images -> zeta.media.images            8                move in stage 4
loop                         zeta.loop -> zeta.runtime.loop             40               high risk; shim first, move last
model_catalog                zeta.model_catalog -> zeta.models.catalog   11               move in stage 4
persistence                  zeta.persistence -> zeta.tui.persistence    3                move with TUI imports
session_cli                  zeta.session_cli -> zeta.cli.session        2                move with CLI package
settings                     zeta.settings -> zeta.config.settings      12               high risk; shim first
submission                   zeta.submission -> zeta.submission.model    2                move with pipeline
submission_pipeline          zeta.submission_pipeline ->                1                move with submission package
                              zeta.submission.pipeline
types                        zeta.types -> zeta.protocol.types          116              highest risk; shim first, move last
```

`loop.py`, `settings.py`, and `types.py` are the riskiest moves. Keep their
old modules as explicit re-export shims while source and tests migrate. Run
the import-boundary tests after each update. Do not create two independent
implementations. The shim must import the one new implementation.

The runtime package requires a dependency-safe move sequence. First replace
`zeta/runtime/__init__.py` with a lazy package initializer. It may define
`__all__` and a module-level `__getattr__`, but it must not import
`composition`, `tools.registry`, `execution`, or `loop` during package import.
Then move `execution.py` to `zeta.runtime.execution`, update the registry and
its other importers, and keep `zeta.execution` as a re-export shim. Move
`headless.py` next. `tool_setup.py` was already moved in stage 2 after the
lazy initializer change. Move `loop.py` last among the runtime children,
update `composition` and its other importers, and keep `zeta.loop` as a
re-export shim until the importer count is zero. Remove the shims only after
a repository search confirms that no importer remains.

After each runtime-child move, run the relevant import-boundary checks in a
fresh process. At minimum, run each command from a new Python process:

```sh
PYTHONPATH=src python -c 'import zeta.runtime.execution'
PYTHONPATH=src python -c 'import zeta.runtime.loop'
PYTHONPATH=src python -c 'import zeta.runtime; from zeta.runtime import compose_runtime'
PYTHONPATH=src python -c 'import zeta.tools.registry'
```

The first two checks must run immediately after their moves, before any test
session imports `zeta.runtime`. A warm pytest process can hide this cycle by
leaving the package partially initialized in `sys.modules`.

## 6. test co-location mechanics

### 6.1 discovery

During migration, keep central tests in `harness/tests` and add the tool
source tree to pytest's search paths:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "src/zeta/tools"]
pythonpath = ["tests", "src"]
addopts = ["--import-mode=importlib", "-p", "zeta_test_plugin"]
asyncio_mode = "auto"
```

Pytest still uses `harness/` as `rootdir` because `pyproject.toml` remains at
that level. The existing `tests/` tree remains discoverable. The `pythonpath`
entry makes the shared plugin importable in every collection tree. Move the
body of `harness/tests/conftest.py` to
`harness/tests/zeta_test_plugin.py`; this plugin owns the HOME isolation,
network block, terminal defaults, live-home guard, and `stock_router_mode`
fixture. The old `harness/tests/conftest.py` becomes a one-line compatibility
loader containing `pytest_plugins = ["zeta_test_plugin"]`. The global `-p`
load is the authoritative path, so tests under both `harness/tests/` and
`harness/src/zeta/tools/**/tests/` receive the same fixtures. Do not add a
second source-tree conftest or copy the fixture code.

Tests import the installed package with absolute imports, for example
`from zeta.tools.read import IMAGE_MAX_BYTES`.

Do not add `__init__.py` files to test directories. The explicit importlib
mode avoids test module-name collisions when several tool directories contain
same-named files during an intermediate stage. The final names should be
tool-specific, such as `test_bash.py` and `test_exec.py`.

Tool-private tests use the public package path when possible:

```python
from zeta.tools.read import read_file
```

Tests use an internal module path only for a private seam that the test owns,
such as the browser foundation:

```python
from zeta.tools.browser.adapter import FakeBrowserAdapter
```

Relative imports from co-located tests are not required. This keeps test
imports stable when pytest changes collection order.

### 6.2 ownership and migration of existing tests

The tool-only portions of the current tests move as follows:

```text
tools/agent/tests/         agent portions of test_agent.py, test_agent_output.py,
                           test_agent_status.py, test_agents.py,
                           test_automations.py, test_session.py,
                           test_session_resilience.py, test_session_safety.py
tools/agent_send/tests/    direct agent_send/send_to_run tests from test_agent.py
                           and test_session_shutdown.py
tools/automation/tests/    test_automations.py tool portions
tools/browser/tests/       test_browser_adapter.py, test_browser_catalog.py,
                           test_browser_prefilter.py
tools/calendar/tests/      test_calendar_tools.py
tools/memory/tests/        memory portions of test_memory_tools.py
tools/read/tests/          read portions of test_read_images.py and test_tools.py
tools/todo/tests/          tool-handler portions of test_todo.py
tools/websearch/tests/     websearch portions of test_webtools.py
tools/fetch/tests/         fetch portions of test_webtools.py
tools/skill/tests/         skill assertions from test_skills.py:77-105
tools/route/tests/         route-only portions of test_evals.py and router tests
tools/bash/tests/          bash portions of test_tools.py and test_safety.py
tools/exec/tests/          exec portions of test_tools.py, test_commands.py,
                           test_safety.py, and test_session_safety.py
tools/edit/tests/          edit portions of test_tools.py
tools/write/tests/         write portions of test_tools.py
tools/zeta_background/tests/  test_background.py tool portions
tools/_shared/tests/       sandbox and process helper portions of test_sandbox.py
                           and test_background.py
```

Mixed integration tests stay in `harness/tests`. Examples are loop/router,
MCP, session lifecycle, TUI, import boundaries, and provider tests. Move
single-owner files whole only when every test follows the same owner. Keep
`test_agent_status.py` as an agent-owned file. Split mixed files by the
behavior under test:

- `test_agent_output.py` sends all agent lifecycle and `agent_output` handler
  tests to `tools/agent/tests/test_agent_output.py`. The test
  `test_foreground_receipt_shows_lifecycle_stats` calls `tui.render` and
  moves to `harness/tests/test_agent_output_tui.py`.
- `test_todo.py` sends the tool-handler, schema, validation, mutation, and
  transcript tests through `test_todo_empty_list_clears_state_and_does_not_pollute_transcript`
  to `tools/todo/tests/test_todo.py`. The store-only test
  `test_todo_items_persist_across_store_resume` moves to
  `harness/tests/test_todo_persistence.py`. All `TodoWidget` and `TUIApp`
  rendering, layout, status, truncation, pinning, and terminal-size tests
  move to `harness/tests/test_todo_tui.py`.
- `test_import_boundaries.py` stays whole at
  `harness/tests/test_import_boundaries.py`. It scans production Python files
  under the complete `src/zeta` tree, starts fresh subprocess imports, and
  checks forbidden dependencies; it is a cross-cutting boundary suite, not a
  tool test. Its production-module scans exclude every `**/tests/**` path
  after stage 3 co-location.

Use this ownership rule for future mixed files: place a test with the
smallest behavior owner when it tests one tool's public contract; keep it in
`harness/tests` when it tests TUI behavior, persistence shared by multiple
layers, import boundaries, or integration across tools. Split a file when
different tests have different owners. A test's imports do not decide its
owner; the behavior and contract under test do.

Rebuild direct send ownership with this content search:

```sh
rg -n --glob 'test_*.py' 'agent_send|send_to_run' harness/tests
```

The direct `agent_send` or `send_to_run` tests in `test_agent.py` move to
`tools/agent_send/tests/`: `test_agent_send_waits_for_blocked_append_before_cancellation`,
`test_tool_registry_reports_agent_send_result_after_cleanup_cancellation`,
`test_agent_send_aborts_before_append_when_store_lock_is_held`,
`test_agent_send_reports_when_the_run_just_closed`,
`test_a_follow_up_reaches_the_run_at_its_next_turn`,
`test_send_to_run_rejects_unknown_and_finished_runs`, and
`test_send_to_run_rejects_non_run_children`. The direct send assertion in
`test_recovery_and_send_release_borrowed_child_stores` moves to the same
owner, while its session shutdown and lease-recovery assertions stay central.

The later run-lifecycle tests `test_a_queued_prompt_stays_out_of_the_run_context`
and `test_a_run_with_an_empty_queue_finishes_normally` stay agent-owned because
they test `AgentLoop` lifecycle, not the direct `agent_send` contract. The
slash-command integration test `test_runs_and_send_commands_drive_a_live_run`
stays central in `harness/tests` because it crosses agent lifecycle and TUI
boundaries. This behavior-owner rule also applies to later tests that mention
a run without calling `send_to_run` directly.

The session and CLI lifecycle portions of `test_session_resilience.py` stay in
`harness/tests`; its `from zeta import session_cli` alias is an importer count,
not a tool-ownership signal.

The remaining mixed files split by the same owner rule. `test_agent.py` sends
its other agent assertions to `tools/agent/tests/`.
`test_skills.py` is also split: only lines 77-105 move to
`tools/skill/tests/`. The other mixed files listed above split by the same
owner rule. Splitting changes file location and imports, not assertions.

### 6.3 packaging

Because Hatchling packages `src/zeta`, test files under `src/zeta/tools/`
would otherwise be candidates for the wheel. Add an explicit test exclusion:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/zeta"]
exclude = ["src/zeta/**/tests/**"]
```

Keep the existing `force-include` entries unchanged. Verify the pattern with
`uv build` and inspect the wheel file list. The verification must show runtime
modules and packaged markdown, but no `zeta/tools/**/tests/` files.

The source checkout still contains the co-located tests. The exclusion affects
the wheel only, not pytest collection.

## 7. discovery-contract preservation

The target preserves discovery in these ways:

1. `tools` remains the scanned package. The registry still calls
   `pkgutil.iter_modules(package.__path__)` on `zeta.tools`.
2. Each real tool directory has `tools/<name>/__init__.py`. `pkgutil` reports
   it as the name `<name>`, and `importlib.import_module` loads
   `zeta.tools.<name>`.
3. Each discovered tool package `__init__.py` exposes a synchronous
   `register(registry)`. The registry calls the same contract as before.
   `agent_send` is the explicit exception: it is a support package without
   `register()` and is registered by `agent.register()` through
   `register_send()`.
4. The `.agent` sort special-case remains unchanged. The discovered name is
   still exactly `zeta.tools.agent`, so the sort key still places it after
   other names.
5. `_shared` begins with an underscore, so the current leading-underscore
   filter skips it. Its nested `process.py`, `sandbox.py`, and
   `user_discovery.py` are never top-level discovery candidates.
6. Modules without `register()` remain valid support modules only when they
   stay outside the actual tool package list. `plan_mode` moves out of
   `tools/`; the browser foundation gains registration only with browser
   handlers.
7. `test_tool_discovery.py` gains a package fixture. It must prove that a
   directory containing `__init__.py` and `register()` is discovered, that
   `_shared` is ignored, and that the sorted `.agent` special-case remains.
8. The registration-order invariant is unchanged: `agent.register()` adds
   `agent`, `agent_status`, `agent_output`, then `agent_send`. Because
   `ToolRegistry.schemas` preserves insertion order, the parity gate records
   ordered tool names and ordered schemas. It compares both lists before and
   after each stage and fails if this sequence or any later entry changes.

The external user-tool loader is separate. It still loads a user file and
requires its own callable `register(registry)`. Moving the internal helper to
`tools/_shared/user_discovery.py` does not change that contract.

## 8. staged migration order

Each stage is independently reviewable. Each stage keeps the behavior-parity
gate: run the relevant suite before and after the stage, and run the full
targeted harness suite before declaring the stage green. No test assertion is
changed. Only file locations, imports, package exports, and pytest packaging
configuration change.

### stage 1: convert registered tool modules to packages

Convert the 13 single-file registered tools to directories, moving each
implementation directly into `__init__.py`. Use internal modules only for
tools with multiple real modules. Move the browser foundation to
`tools/browser/`. Keep tests in `harness/tests` for this stage, updating only
imports that must change for browser internals and the interim relative depth.

The four existing package directories are covered explicitly. `automation/`
and `zeta_background/` already conform: their implementation and
`register()` are in `__init__.py`, so stage 1 verifies and keeps that shape.
`agent_send/` keeps its implementation and `register_send()` in `__init__.py`
but remains the non-discovered registration exception; stage 2 verifies its
order-preserving call from `agent.register()`. `plan_mode/` is framework
support, not a registered tool; stage 2 moves it to `zeta.agent.plan_mode`
without applying the discovered-tool shape.

Risk: a missing re-export or a package whose `register()` is not visible will
change tool discovery or break private test imports.

Importability: after each file becomes a package, rewrite its relative imports
to the stage-1 parent paths from section 3.2. Run a fresh process import of
each changed `zeta.tools.<name>` package before the next move. No stage-1
import may reference `_shared`; that package does not exist yet.

Verification: `test_tool_discovery.py`, all current tool-specific files,
browser adapter/catalog/prefilter tests, and then `uv run --frozen pytest -q`.
Run a registry smoke check that compares discovered tool names and schemas
before and after.

### stage 2: move helpers and framework support

First replace `zeta/runtime/__init__.py` with the lazy initializer required by
section 5.1. Verify it in a fresh process before moving any child under
`zeta.runtime`. Then create `tools/_shared/`, move the three shared helpers,
and rewrite all source and test imports. Create the empty or lazy
`zeta.agent/__init__.py`, then move agent presets and plan mode to
`zeta.agent`. Move tool setup to `zeta.runtime.tool_setup` only after the
runtime initializer change, and update `loop.py` to import
`zeta.runtime.tool_setup`. Keep `registry.py` at `zeta.tools.registry`.
Keep `agent_send` out of discovery and retain the explicit `agent.register()`
call to `register_send()` after the other agent tools.

Risk: relative import depth, circular imports, and accidental discovery of
support files.

Importability: run fresh processes after the lazy initializer, helper rewrite,
agent moves, and tool-setup move. At minimum, verify `import zeta.runtime`,
`import zeta.runtime.tool_setup`, `import zeta.loop`, and
`import zeta.tools.registry`. A warm process is not valid evidence.

Verification: background, sandbox, user-tool-discovery, agent, plan-mode,
route, registry, and import-boundary tests. Assert `_shared` does not appear
in `_discover_tool_modules()`.

### stage 3: co-locate tool tests

Move tool-private tests under their owning `tools/<name>/tests/` directory.
Split mixed test files by ownership without changing assertions. Keep
cross-tool integration tests in `harness/tests`. Add `src/zeta/tools` to
`testpaths`, use importlib test loading if needed, and add the Hatch wheel
test exclusion.

Risk: pytest collection gaps, duplicate test module names, and accidental test
inclusion in the wheel.

Importability: this stage moves tests only. It must not change source import
depth or runtime package initialization. Run collection in importlib mode
before and after each ownership split.

Verification: `uv run --frozen pytest --collect-only -q` and compare collected
node counts with the pre-move baseline. Run every moved test by its new path,
then the complete suite. Update both production-module scans in
`test_import_boundaries.py` to exclude paths matching `**/tests/**`: the
fresh-process module list and the forbidden-import scan. Run the boundary
suite after that update. Run `uv build` and inspect the wheel contents.

### stage 4: group loose top-level modules

The lazy `zeta.runtime.__init__` and `zeta.runtime.tool_setup` move are
complete in stage 2. First verify those paths in a fresh process. Then move
`execution.py`, updating `tools.registry` and every other importer for its
new relative depth. Run a fresh-process import of
`zeta.runtime.execution` and `zeta.tools.registry`. Move `headless.py` next,
updating its runtime driver import. Move `loop.py` last among runtime
children, update `runtime.composition` and every other importer, and run a
fresh-process import of `zeta.runtime.loop`. Only after these checks pass may
the other loose modules move to the proposed `agent`, `cli`, `config`,
`media`, `models`, `protocol`, `submission`, and `tui` locations. Keep
temporary re-export shims for `loop`, `settings`, and `types` until their
importer counts reach zero. Preserve the `zeta.cli` package exports and
`zeta.cli:main` entry point.

Before moving `persistence.py` to `zeta.tui.persistence`, replace the current
eager `zeta.tui.__init__` import of `.app` with a lazy initializer. Verify
`import zeta.tui` and `import zeta.tui.persistence` in separate fresh Python
processes before and after the move.

Risk: import cycles and high fan-out breakage, especially around the lazy
runtime initializer, `types`, `loop`, and `settings`.

Importability: create each target package initializer before moving its first
child. Update relative imports at the target depth before each move. Run a
fresh Python process for every moved child, then run the boundary suite.

Verification: import-boundary tests, CLI parser and entry-point tests, all
agent/runtime/provider/server/TUI suites, package import smoke tests, and the
full `uv run --frozen pytest -q` suite. Run the runtime import checks from
separate Python processes, not only from the warm pytest process. Remove a
shim only after a repository search shows no remaining importer.

Arc-3 tasks 5-12 are paused pending this reorg. Tasks that add browser
handlers, including task 12's real Playwright adapter, must use
`tools/browser/`, its package `register()` contract, and co-located browser
tests. Do not resume those tasks against the old flat
`browser_adapter.py` and `browser_catalog.py` paths.

## 9. risks, rollback, and parity gate

Primary risks are:

- a tool package imports the wrong internal module and fails during discovery;
- a support package is discovered as a tool or a real tool lacks `register()`;
- re-export omissions break private consumers or monkeypatch string targets;
- `types`, `loop`, or `settings` create a cycle after grouping;
- pytest finds fewer tests after co-location;
- wheel exclusion patterns fail and ship test code;
- external users import private old paths not found by the repository search.

Rollback is per stage. Revert the stage commit or restore the previous path
and keep the old compatibility shim. No database, session, or user data
format changes are part of this reorg. A rollback must leave the old
`zeta.tools.<name>` modules and central test paths usable.

The explicit behavior-parity gate is:

```text
before stage: targeted suite is green; record collected test count, ordered tool names, and ordered schemas
after stage: the same targeted suite is green; compare counts, names, and schemas in order
stage exit: full targeted harness suite is green; no test assertion changed
```

The reorg changes locations, imports, package exports, and test discovery
configuration only. Any changed result, schema, registration order, approval
decision, or assertion is a stage failure.

## 10. self-review record

This design was reviewed against the ticket requirements before handoff:

- current tree, tools, support files, and other subpackages are enumerated;
- source and test import coupling is mapped with distinct importer counts;
- the inventory is reproducible with the recorded `rg` commands and includes
  aliases such as `from zeta import agent_runner`;
- seven stale count cells were corrected: five tool test counts and the
  `agent_runner` and `session_cli` move-row counts;
- existing package-directory behavior and the `register()` exceptions are
  recorded;
- browser foundation placement and the arc-3 task 5-12 pause are explicit;
- the real Playwright adapter is assigned to `tools/browser/` with co-located
  tests;
- simple tools keep implementation in `__init__.py`, with browser and shared
  helper patch-seam changes enumerated;
- `agent_send` registration remains manual and its ordering invariant is
  checked by the parity gate;
- direct `agent_send` and `send_to_run` tests are found by content, while
  broader lifecycle tests stay agent-owned and the slash-command integration
  test stays central by the behavior-owner rule;
- `agent_send`, `skill`, agent-output, todo, and boundary-test ownership,
  including mixed-file splits, is explicit;
- pytest discovery, the shared plugin and its fixtures, src layout, and wheel
  exclusion mechanics are specified;
- the runtime initializer is lazy before `tool_setup.py`, `execution.py`, or
  `loop.py` moves; the stage order and fresh-process import checks are
  explicit;
- every moved module has a package-initializer cycle audit, and each stage
  states the relative import depth that is valid at that point;
- `pkgutil`, the leading-underscore skip, `.agent` sorting, and synchronous
  registration are preserved;
- migration stages include risks and targeted verification;
- no implementation code or file moves are included in this spec.

The placeholder scan found no unresolved `TBD`, `FIXME`, or decision marker.
The only open decision is whether Henry wants the temporary old-path shims
removed in the same final reorg PR or in a follow-up cleanup PR. The design
assumes removal in the final stage after the importer count reaches zero.
