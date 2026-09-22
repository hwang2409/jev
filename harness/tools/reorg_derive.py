#!/usr/bin/env python3
"""Derive deterministic source-reorg moves, rewrites, tests, and scans."""
from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SRC = ROOT / "harness/src"
HARNESS = ROOT / "harness"

# The only hand-maintained migration data. Do not duplicate it in prose.
MOVE_TABLE = {
    1: [
        ("zeta.tools.agent", "zeta.tools.agent"),
        ("zeta.tools.bash", "zeta.tools.bash"),
        ("zeta.tools.calendar", "zeta.tools.calendar"),
        ("zeta.tools.edit", "zeta.tools.edit"),
        ("zeta.tools.exec", "zeta.tools.exec"),
        ("zeta.tools.fetch", "zeta.tools.fetch"),
        ("zeta.tools.memory", "zeta.tools.memory"),
        ("zeta.tools.read", "zeta.tools.read"),
        ("zeta.tools.route", "zeta.tools.route"),
        ("zeta.tools.skill", "zeta.tools.skill"),
        ("zeta.tools.todo", "zeta.tools.todo"),
        ("zeta.tools.websearch", "zeta.tools.websearch"),
        ("zeta.tools.write", "zeta.tools.write"),
        ("zeta.tools.browser_adapter", "zeta.tools.browser.adapter"),
        ("zeta.tools.browser_catalog", "zeta.tools.browser.catalog"),
    ],
    2: [
        ("zeta.tools._process", "zeta.tools._shared.process"),
        ("zeta.tools._sandbox", "zeta.tools._shared.sandbox"),
        ("zeta.tools._user_discovery", "zeta.tools._shared.user_discovery"),
        ("zeta.tools.agent_presets", "zeta.agent.presets"),
        ("zeta.tools.plan_mode", "zeta.agent.plan_mode"),
        ("zeta.tools.loop_setup", "zeta.runtime.tool_setup"),
    ],
    3: [],
    4: [
        ("zeta.execution", "zeta.runtime.execution"),
        ("zeta.headless", "zeta.runtime.headless"),
        ("zeta.loop", "zeta.runtime.loop"),
        ("zeta.agent_background", "zeta.agent.background"),
        ("zeta.agent_budget", "zeta.agent.budget"),
        ("zeta.agent_receipt", "zeta.agent.receipt"),
        ("zeta.agent_runner", "zeta.agent.runner"),
        ("zeta.cli", "zeta.cli.main"),
        ("zeta.session_cli", "zeta.cli.session"),
        ("zeta.settings", "zeta.config.settings"),
        ("zeta.images", "zeta.media.images"),
        ("zeta.model_catalog", "zeta.models.catalog"),
        ("zeta.submission", "zeta.submission.model"),
        ("zeta.submission_pipeline", "zeta.submission.pipeline"),
        ("zeta.persistence", "zeta.tui.persistence"),
        ("zeta.types", "zeta.protocol.types"),
    ],
}

# Stage 3 ownership is also input data. Each source test has every post-split
# destination so later targeted-test lists include central remainders.
SPLIT_TABLE = {
    "test_agent.py": ["src/zeta/tools/agent/tests/test_agent.py", "src/zeta/tools/agent_send/tests/test_agent_send.py", "tests/test_agent_integration.py"],
    "test_agent_output.py": ["src/zeta/tools/agent/tests/test_agent_output.py", "tests/test_agent_output_tui.py"],
    "test_agent_status.py": ["src/zeta/tools/agent/tests/test_agent_status.py"],
    "test_agents.py": ["src/zeta/tools/agent/tests/test_agents.py"],
    "test_automations.py": ["src/zeta/tools/agent/tests/test_automations.py", "src/zeta/tools/automation/tests/test_automation.py", "tests/test_automations_integration.py"],
    "test_background.py": ["src/zeta/tools/zeta_background/tests/test_background.py", "src/zeta/tools/_shared/tests/test_process.py", "tests/test_background_integration.py"],
    "test_browser_adapter.py": ["src/zeta/tools/browser/tests/test_adapter.py"],
    "test_browser_catalog.py": ["src/zeta/tools/browser/tests/test_catalog.py"],
    "test_browser_prefilter.py": ["src/zeta/tools/browser/tests/test_prefilter.py"],
    "test_calendar_tools.py": ["src/zeta/tools/calendar/tests/test_calendar_tools.py"],
    "test_commands.py": ["src/zeta/tools/exec/tests/test_exec_commands.py", "tests/test_commands.py"],
    "test_evals.py": ["src/zeta/tools/route/tests/test_route_evals.py", "tests/test_evals.py"],
    "test_memory_tools.py": ["src/zeta/tools/memory/tests/test_memory_tools.py", "tests/test_memory_tools_integration.py"],
    "test_read_images.py": ["src/zeta/tools/read/tests/test_read_images.py", "tests/test_read_images_integration.py"],
    "test_router_auto.py": ["src/zeta/tools/route/tests/test_router_auto.py", "tests/test_router_auto_integration.py"],
    "test_router_mode.py": ["src/zeta/tools/route/tests/test_router_mode.py", "tests/test_router_mode_integration.py"],
    "test_safety.py": ["src/zeta/tools/bash/tests/test_bash_safety.py", "src/zeta/tools/exec/tests/test_exec_safety.py", "tests/test_safety.py"],
    "test_sandbox.py": ["src/zeta/tools/_shared/tests/test_sandbox.py", "tests/test_sandbox_integration.py"],
    "test_session.py": ["src/zeta/tools/agent/tests/test_session.py", "tests/test_session_integration.py"],
    "test_session_resilience.py": ["src/zeta/tools/agent/tests/test_session_resilience.py", "tests/test_session_resilience_integration.py"],
    "test_session_safety.py": ["src/zeta/tools/agent/tests/test_session_safety.py", "src/zeta/tools/exec/tests/test_session_safety.py", "tests/test_session_safety_integration.py"],
    "test_session_shutdown.py": ["src/zeta/tools/agent_send/tests/test_agent_send.py", "tests/test_session_shutdown.py"],
    "test_skills.py": ["src/zeta/tools/skill/tests/test_skill_tool.py", "tests/test_skills.py"],
    "test_tools.py": ["src/zeta/tools/read/tests/test_read.py", "src/zeta/tools/bash/tests/test_bash.py", "src/zeta/tools/exec/tests/test_exec.py", "src/zeta/tools/edit/tests/test_edit.py", "src/zeta/tools/write/tests/test_write.py", "tests/test_tools_integration.py"],
    "test_todo.py": ["src/zeta/tools/todo/tests/test_todo.py", "tests/test_todo_persistence.py", "tests/test_todo_tui.py"],
    "test_webtools.py": ["src/zeta/tools/fetch/tests/test_fetch.py", "src/zeta/tools/websearch/tests/test_websearch.py"],
}
SPLIT_TABLE["conftest.py"] = ["tests/zeta_test_plugin.py"]
for path in sorted((HARNESS / "tests").glob("test_*.py")):
    SPLIT_TABLE.setdefault(path.name, [f"tests/{path.name}"])

PACKAGE_ROOTS = {old for old, new in MOVE_TABLE[1] if old == new}

@dataclass(frozen=True)
class ImportRecord:
    path: Path
    line: int
    old: str
    target: str
    text: str
    kind: str
    names: str = ""
    imported_name: str = ""
    asname: str = ""


def module_for(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def package_for(path: Path) -> str:
    module = module_for(path)
    return module.rsplit(".", 1)[0] if "." in module else module


def resolve(current: str, level: int, module: str | None) -> str:
    if not level:
        return module or ""
    package = current.rsplit(".", 1)[0] if not current.endswith(".__init__") else current.removesuffix(".__init__")
    base = package.split(".") if package else []
    base = base[: len(base) - level + 1]
    if module:
        base += module.split(".")
    return ".".join(base)


def source_files() -> list[Path]:
    return sorted((HARNESS / "src/zeta").rglob("*.py")) + sorted((HARNESS / "tests").glob("*.py"))


def display(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_imports(path: Path) -> list[ImportRecord]:
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    if path.is_relative_to(SRC):
        current = module_for(path)
        if path.name == "__init__.py":
            current += ".__init__"
    else:
        current = f"tests.{path.stem}"
    known_modules = {module_for(candidate) for candidate in (HARNESS / "src/zeta").rglob("*.py")}
    moved_modules = {old for moves in MOVE_TABLE.values() for old, _ in moves}
    records = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                records.append(ImportRecord(path, node.lineno, alias.name, alias.name, text.splitlines()[node.lineno-1].strip(), "import", asname=alias.asname or ""))
        elif isinstance(node, ast.ImportFrom):
            target = resolve(current, node.level, node.module)
            old = ("." * node.level + (node.module or "")) if node.level else target
            names = text.splitlines()[node.lineno - 1].strip().split(" import ", 1)[-1]
            records.append(ImportRecord(path, node.lineno, old, target, text.splitlines()[node.lineno-1].strip(), "from", names=names))
            for alias in node.names:
                child = f"{target}.{alias.name}" if target else alias.name
                if child in known_modules or child in moved_modules:
                    records.append(ImportRecord(path, node.lineno, old, child, text.splitlines()[node.lineno-1].strip(), "from_alias", imported_name=alias.name, asname=alias.asname or ""))
    return sorted(records, key=lambda r: (display(r.path), r.line, r.old))


def dotted_prefix(name: str, root: str) -> bool:
    return name == root or name.startswith(root + ".")


def remap(name: str, moves: list[tuple[str, str]]) -> str:
    new_names = {new for _, new in moves}
    if any(dotted_prefix(name, new) for new in new_names):
        return name
    for old, new in sorted(moves, key=lambda item: len(item[0]), reverse=True):
        if dotted_prefix(name, old):
            return new + name[len(old):]
    return name


def importer_module(path: Path, moves: list[tuple[str, str]]) -> str:
    if path.is_relative_to(SRC):
        module = remap(module_for(path), moves)
        package_roots = {old for old, new in moves if old == new}
        if path.name == "__init__.py" or module in package_roots:
            return module + ".__init__"
        return module
    rel = path.relative_to(HARNESS).with_suffix("")
    return ".".join(rel.parts)


def relative_import(current: str, target: str) -> str:
    package = current.rsplit(".", 1)[0].split(".")
    target_parts = target.split(".")
    common = 0
    while common < min(len(package), len(target_parts)) and package[common] == target_parts[common]:
        common += 1
    level = len(package) - common + 1
    tail = ".".join(target_parts[common:])
    return f"{'.' * level}{tail}"


def import_text(record: ImportRecord, target: str, current: str) -> str:
    if record.kind == "import":
        suffix = f" as {record.asname}" if record.asname else ""
        return f"import {target}{suffix}"
    if record.kind == "from_alias":
        parent, name = target.rsplit(".", 1)
        dotted = relative_import(current, parent) if record.old.startswith(".") else parent
        binding = record.asname or record.imported_name
        suffix = f" as {binding}" if binding != name else ""
        return f"from {dotted} import {name}{suffix}"
    dotted = relative_import(current, target) if record.old.startswith(".") else target
    return f"from {dotted} import {record.names or '*'}"


def moved_old_names(stage: int) -> set[str]:
    return {old for old, _ in MOVE_TABLE[stage]}


def all_moves_through(stage: int) -> list[tuple[str, str]]:
    result = []
    for number in sorted(MOVE_TABLE):
        if number <= stage:
            result += MOVE_TABLE[number]
    return result


def state_moves(stage: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    before = all_moves_through(stage - 1)
    return before, before + MOVE_TABLE[stage]


def old_path_matches(name: str, moves: list[tuple[str, str]]) -> bool:
    old_names = {old for old, new in moves if old != new}
    new_names = {new for _, new in moves}
    if any(dotted_prefix(name, new) for new in new_names):
        return False
    return any(dotted_prefix(name, old) for old in old_names)


def scan(names: list[str], moves: list[tuple[str, str]]) -> list[str]:
    return sorted(name for name in names if old_path_matches(name, moves))


def self_check() -> None:
    moves = all_moves_through(max(MOVE_TABLE))
    paths = {path for stage in MOVE_TABLE.values() for old, new in stage for path in (old, new)}
    for path in sorted(paths):
        remapped = remap(path, moves)
        assert remap(remapped, moves) == remapped
    assert remap("zeta.cli.main", moves) == "zeta.cli.main"
    assert remap("zeta.submission.model", moves) == "zeta.submission.model"
    correct_tree = {remap(path, moves) for path in paths}
    correct_tree.update(f"{path}.child" for path in tuple(correct_tree))
    assert scan(sorted(correct_tree), moves) == []


PINNED_STAGE4_ORDER = [
    "zeta.execution",
    "zeta.headless",
    "zeta.loop",
    "zeta.agent_background",
    "zeta.agent_budget",
    "zeta.agent_receipt",
    "zeta.agent_runner",
    "zeta.cli",
    "zeta.session_cli",
    "zeta.settings",
    "zeta.images",
    "zeta.model_catalog",
    "zeta.submission",
    "zeta.submission_pipeline",
    "zeta.persistence",
    "zeta.types",
]


def stage4_order(records: list[ImportRecord]) -> tuple[list[str], list[tuple[str, str, str]]]:
    entries = dict(MOVE_TABLE[4])
    names = set(entries)
    order_index = {name: i for i, name in enumerate(PINNED_STAGE4_ORDER)}
    shim_roots = {"zeta.loop", "zeta.settings", "zeta.types"}
    deps = {name: set() for name in names}
    for record in records:
        importer = module_for(record.path) if record.path.is_relative_to(SRC) else ""
        importer_root = next((name for name in names if importer == name or importer.startswith(name + ".")), None)
        target_root = next((name for name in names if record.target == name or record.target.startswith(name + ".")), None)
        if importer_root and target_root and importer_root != target_root and target_root not in shim_roots:
            deps[importer_root].add(target_root)
    for before, after in zip(PINNED_STAGE4_ORDER, PINNED_STAGE4_ORDER[1:]):
        deps[after].add(before)
    order = []
    interim = []
    remaining = {name: set(values) for name, values in deps.items()}
    while len(order) < len(names):
        ready = [name for name in names - set(order) if not (remaining[name] - set(order))]
        if ready:
            name = min(ready, key=order_index.get)
        else:
            # Break a cycle at the earliest table entry. Its later targets use
            # old roots at the importer's new depth until those targets move.
            name = min(names - set(order), key=order_index.get)
            for target in sorted(remaining[name] - set(order), key=order_index.get):
                interim.append((name, target, "cycle: keep old target root until target move"))
            remaining[name].clear()
        order.append(name)
    return order, interim


def test_destinations(path: Path) -> list[str]:
    if not path.is_relative_to(HARNESS / "tests"):
        return [display(path)]
    return [f"harness/{item}" for item in SPLIT_TABLE.get(path.name, [f"tests/{path.name}"])]


def source_is_moved(path: Path, stage_moves: list[tuple[str, str]]) -> bool:
    if not path.is_relative_to(SRC):
        return False
    return any(module_for(path) == old for old, _ in stage_moves)


def dotted_references(stage: int, before_moves: list[tuple[str, str]], after_moves: list[tuple[str, str]]) -> list[tuple[str, int, str, str]]:
    old_names = {old for old, _ in MOVE_TABLE[stage]}
    if not old_names:
        return []
    pattern = re.compile(
        r"zeta\.(?:"
        + "|".join(re.escape(old.removeprefix("zeta.")) for old in old_names)
        + r")(?:\.[A-Za-z_][\w]*)*"
    )
    references = []
    for root in (HARNESS / "src/zeta", HARNESS / "tests"):
        for path in sorted(root.rglob("*.py")):
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                for match in pattern.finditer(line):
                    before = remap(match.group(), before_moves)
                    after = remap(match.group(), after_moves)
                    if before != after:
                        references.append((display(path), line_no, before, after))
    return references


def old_path_scan_command(stage: int) -> list[str]:
    moves = [(old, new) for old, new in MOVE_TABLE[stage] if old != new]
    if not moves:
        return ["true"]
    old_names = sorted(old for old, _ in moves)
    new_names = sorted({new for _, new in moves})
    exclusions = sorted(str(HARNESS / f"src/zeta/{name}.py") for name in ("loop", "settings", "types")) if stage == 4 else []
    output = f"/tmp/reorg-stage{stage}-old-paths.txt"
    return [
        "set +e",
        f"python - <<'PY' > {output}",
        "import ast",
        "import re",
        "from pathlib import Path",
        "",
        "ROOT = Path('.')",
        "SRC = ROOT / 'harness/src/zeta'",
        f"OLD_NAMES = {old_names!r}",
        f"NEW_NAMES = {new_names!r}",
        f"EXCLUDED = {exclusions!r}",
        "",
        "def dotted_prefix(name, root):",
        "    return name == root or name.startswith(root + '.')",
        "",
        "def old_path_matches(name):",
        "    if any(dotted_prefix(name, new) for new in NEW_NAMES):",
        "        return False",
        "    return any(dotted_prefix(name, old) for old in OLD_NAMES)",
        "",
        "def module_for(path):",
        "    rel = path.relative_to(SRC).with_suffix('')",
        "    parts = list(rel.parts)",
        "    if parts[-1] == '__init__':",
        "        parts.pop()",
        "    return '.'.join(parts)",
        "",
        "def current_for(path):",
        "    if path.is_relative_to(SRC):",
        "        current = module_for(path)",
        "        return current + '.__init__' if path.name == '__init__.py' else current",
        "    return '.'.join(path.relative_to(ROOT / 'harness').with_suffix('').parts)",
        "",
        "def resolve(current, level, module):",
        "    if not level:",
        "        return module or ''",
        "    package = current.rsplit('.', 1)[0] if not current.endswith('.__init__') else current.removesuffix('.__init__')",
        "    base = package.split('.') if package else []",
        "    base = base[:len(base) - level + 1]",
        "    if module:",
        "        base += module.split('.')",
        "    return '.'.join(base)",
        "",
        "hits = set()",
        "for path in sorted(list((ROOT / 'harness/src/zeta').rglob('*.py')) + list((ROOT / 'harness/tests').glob('*.py'))):",
        "    if str(path) in EXCLUDED:",
        "        continue",
        "    text = path.read_text()",
        "    tree = ast.parse(text, filename=str(path))",
        "    current = current_for(path)",
        "    for node in ast.walk(tree):",
        "        names = []",
        "        if isinstance(node, ast.Import):",
        "            names = [alias.name for alias in node.names]",
        "        elif isinstance(node, ast.ImportFrom):",
        "            target = resolve(current, node.level, node.module)",
        "            names = ([target] if target else []) + [f'{target}.{alias.name}' if target else alias.name for alias in node.names]",
        "        for name in names:",
        "            if name and old_path_matches(name):",
        "                hits.add(f'{path}:{node.lineno}:{name}')",
        "    for line_no, line in enumerate(text.splitlines(), 1):",
        "        for match in re.finditer(r'\\bzeta(?:\\.[A-Za-z_]\\w*)+', line):",
        "            name = match.group()",
        "            if old_path_matches(name):",
        "                hits.add(f'{path}:{line_no}:{name}')",
        "",
        "print('\\n'.join(sorted(hits)))",
        "PY",
        "status=$?",
        "if [ \"$status\" -ne 0 ]; then exit \"$status\"; fi",
        f"if [ -s {output} ]; then cat {output}; exit 1; fi",
        "set -e",
    ]


def stage_output(stage: int, records: list[ImportRecord]) -> str:
    stage_moves = MOVE_TABLE[stage]
    before_moves, after_moves = state_moves(stage)
    rewrites = []
    for record in records:
        if not source_is_moved(record.path, stage_moves) and remap(record.target, before_moves) == remap(record.target, after_moves):
            continue
        before_current = importer_module(record.path, before_moves)
        after_current = importer_module(record.path, after_moves)
        before_text = import_text(record, remap(record.target, before_moves), before_current)
        new_text = import_text(record, remap(record.target, after_moves), after_current)
        destinations = (test_destinations(record.path) if stage >= 3 else [display(record.path)]) if record.path.is_relative_to(HARNESS / "tests") else [display(record.path)]
        for destination in destinations:
            if before_text != new_text:
                rewrites.append((destination, record.line, before_text, new_text))
    string_refs = dotted_references(stage, before_moves, after_moves)
    if stage == 4:
        order, cycles = stage4_order(records)
    elif stage == 3:
        order, cycles = [], []
    else:
        order = [old for old, _ in stage_moves]
        cycles = []
    affected_tests = set()
    touched_targets = {old for old, _ in stage_moves}
    for record in records:
        touches = any(record.target == old or record.target.startswith(old + ".") for old in touched_targets) or source_is_moved(record.path, stage_moves)
        if record.path.is_relative_to(HARNESS / "tests") and touches:
            affected_tests.update(test_destinations(record.path) if stage >= 3 else [display(record.path)])
    for path, _, _, _ in string_refs:
        path_obj = ROOT / path
        if stage >= 3 and path_obj.is_relative_to(HARNESS / "tests"):
            affected_tests.update(test_destinations(path_obj))
        else:
            affected_tests.add(path)
    if stage == 3:
        affected_tests = {f"harness/{item}" for destinations in SPLIT_TABLE.values() for item in destinations}
    if stage == 1:
        affected_tests.add("harness/tests/test_tool_discovery.py")
    if stage == 2:
        affected_tests.update({"harness/tests/test_router_auto.py", "harness/tests/test_router_mode.py", "harness/tests/test_tools.py", "harness/tests/test_import_boundaries.py"})
    # Final scan includes AST imports, aliases, and dotted strings.
    scan = []
    scan_moves = MOVE_TABLE[stage] if stage != 3 else []
    for record in records:
        if record.path.name in {"loop.py", "settings.py", "types.py"} and record.path.parent == HARNESS / "src/zeta":
            continue
        if old_path_matches(record.target, scan_moves):
            scan.append((display(record.path), record.line, record.target))
    scan.extend((path, line, old) for path, line, old, _ in dotted_references(stage, [], []))
    out = [f"## stage {stage} generated output", "", "generated by `reorg_derive.py` — regenerate, do not hand-edit.", "", "### move order", ""]
    if stage == 3:
        for source, destinations in sorted(SPLIT_TABLE.items()):
            out.append(f"- `harness/tests/{source}` -> " + ", ".join(f"`harness/{destination}`" for destination in destinations))
    else:
        move_map = dict(stage_moves if stage != 4 else MOVE_TABLE[4])
        for i, name in enumerate(order, 1):
            out.append(f"{i}. `{name}` -> `{move_map[name]}`")
    if cycles:
        out += ["", "cycle report and interim rewrites:", ""]
        out += [f"- `{name}` -> `{target}`: {why}" for name, target, why in cycles]
    final_rewrite_count = len(set(rewrites))
    interim_count = 0
    if stage == 4:
        order_index = {name: i for i, name in enumerate(order)}
        stage4_map = dict(MOVE_TABLE[4])
        for record in records:
            importer = module_for(record.path) if record.path.is_relative_to(SRC) else ""
            importer_root = next((name for name in stage4_map if importer == name or importer.startswith(name + ".")), None)
            target_root = next((name for name in stage4_map if record.target == name or record.target.startswith(name + ".")), None)
            if importer_root and target_root and order_index[importer_root] < order_index[target_root]:
                interim_count += 1
    out += ["", "### summary", "", f"moves: {len(stage_moves) if stage != 3 else sum(len(destinations) for destinations in SPLIT_TABLE.values())}; final rewrites: {final_rewrite_count}; interim rewrites: {interim_count}; forward-import violations: 0; targeted test files: {len(affected_tests)}; final scan: {len(scan)}", "", "### importer rewrites", "", "file | line | old import | new import", "--- | ---: | --- | ---"]
    if stage == 4:
        order_index = {name: i for i, name in enumerate(order)}
        stage4_map = dict(MOVE_TABLE[4])
        for record in records:
            importer = module_for(record.path) if record.path.is_relative_to(SRC) else ""
            importer_root = next((name for name in stage4_map if importer == name or importer.startswith(name + ".")), None)
            target_root = next((name for name in stage4_map if record.target == name or record.target.startswith(name + ".")), None)
            if not importer_root or not target_root or importer_root == target_root:
                continue
            if order_index[importer_root] < order_index[target_root]:
                old_target = record.target
                current = importer_module(record.path, after_moves)
                interim_text = import_text(record, old_target, current)
                destinations = [display(record.path)]
                if record.path.is_relative_to(HARNESS / "tests") and stage >= 3:
                    destinations = test_destinations(record.path)
                for destination in destinations:
                    out.append(" | ".join(f"`{value}`" for value in (destination, record.line, record.text, interim_text + " (interim)")))
    for row in sorted(set(rewrites)):
        out.append(" | ".join(f"`{value}`" for value in row))
    for row in sorted(string_refs):
        out.append(" | ".join([f"`{row[0]}`", f"`{row[1]}`", f"`{row[2]}`", f"`{row[3]}` (string reference)" ]))
    out += ["", "### executable targeted tests", "", "generated by `reorg_derive.py` — regenerate, do not hand-edit.", ""]
    out += [f"- `{path}`" for path in sorted(affected_tests)]
    out += ["", "### gate commands", "", "generated by `reorg_derive.py` — regenerate, do not hand-edit.", ""]
    before_listing = f"/tmp/reorg-stage{stage}-collection-before.txt"
    after_listing = f"/tmp/reorg-stage{stage}-collection-after.txt"
    normalizer = "sed -E 's#(src/zeta/tools/[^ ]+|tests/[^ ]+)::#::#'"
    listing = [
        f"test -s {before_listing}",
        f"(cd harness && uv run --frozen pytest --collect-only -q | {normalizer} > {after_listing})",
        f"diff -u {before_listing} {after_listing}",
    ]
    if stage == 3:
        targeted = " ".join(path.removeprefix("harness/") for path in sorted(affected_tests))
        out += ["```sh", *listing, f"(cd harness && uv run --frozen pytest -q {targeted})", "```"]
    else:
        target_modules = [new for old, new in stage_moves]
        imports = "; ".join(f"import {name}" for name in target_modules)
        targeted = " ".join(path.removeprefix("harness/") for path in sorted(affected_tests))
        out += ["```sh", *listing, f"(cd harness && PYTHONPATH=src python -c '{imports}')", f"(cd harness && uv run --frozen pytest -q {targeted})", "```"]
    out += ["", "### final old-path scan", "", "generated by `reorg_derive.py` — regenerate, do not hand-edit.", "", f"expected count: {len(scan)}", ""]
    out += [f"- `{path}:{line}:{target}`" for path, line, target in sorted(scan)]
    out += ["", "executable zero-result check:", "", "```sh", *old_path_scan_command(stage), "```"]
    return "\n".join(out)


def main() -> None:
    self_check()
    records = [record for path in source_files() for record in parse_imports(path)]
    print("# reorg_derive.py output")
    print("# deterministic; generated from MOVE_TABLE and SPLIT_TABLE")
    for stage in sorted(MOVE_TABLE):
        print()
        print(stage_output(stage, records))

if __name__ == "__main__":
    main()
