import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).parents[1] / "src" / "zeta"
LEGACY_RUNTIME_LOOP_EXPORTS = (
    "AgentCatalog",
    "AgentLoop",
    "AgentTree",
    "Any",
    "ApprovalPolicy",
    "AsyncIterator",
    "BackgroundAgentOwner",
    "Callable",
    "CompletionBackend",
    "ContentBlock",
    "ContextAssembler",
    "ConversationStore",
    "Coroutine",
    "ErrorInfo",
    "FAILED_TURN_ERROR",
    "FAILED_TURN_MARKER",
    "HookManager",
    "MAX_AGENT_DEPTH",
    "MAX_AGENT_RESULT_BYTES",
    "MAX_ERROR_MESSAGE",
    "MCPCommandError",
    "MCPConfigError",
    "MCPMount",
    "MCP_USAGE",
    "MEMORY_INJECTION_EXCERPT_CHARS",
    "MEMORY_INJECTION_PREFIX",
    "MEMORY_INJECTION_TOP_K",
    "MEMORY_INJECTION_TOTAL_CHARS",
    "MEMORY_RELEVANCE_GATE",
    "Mapping",
    "MemoryInjectionSkipReason",
    "Message",
    "MessageRole",
    "NEEDS_TOOL_GATE",
    "PLAN_MODE_PREAMBLE",
    "PLAN_MODE_TOOLS",
    "Path",
    "ROUTE_TOPK_CONFIDENCE",
    "RoutingSchemaContent",
    "Sequence",
    "SkillCatalog",
    "SlashModelInput",
    "StrEnum",
    "StreamEvent",
    "StreamEventType",
    "TYPE_CHECKING",
    "TaskResult",
    "TerminalState",
    "TextContent",
    "ThinkingContent",
    "ToolAbortSignal",
    "ToolCall",
    "ToolExecutionContext",
    "ToolHandler",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "ToolStreamPublisher",
    "ToolUseContent",
    "TypeVar",
    "UTC",
    "add_and_mount",
    "agent_result",
    "annotations",
    "asdict",
    "asyncio",
    "auto_route",
    "build_catalog",
    "compose_system_prompt",
    "consume_turn",
    "datetime",
    "deque",
    "dispatch_tool_calls",
    "finalize_agent_results",
    "flatten_tool_content",
    "hashlib",
    "home_config_path",
    "httpx",
    "json",
    "load_identity",
    "load_mcp_config_overlay",
    "logging",
    "memory_relevance",
    "mount_mcp_servers",
    "os",
    "parse_add_command",
    "project_config_path",
    "recover_agent_children",
    "remove_and_unshadow",
    "render_mcp_status",
    "replace",
    "run_agent_tool",
    "run_mcp_auth",
    "run_mcp_resource_attach",
    "run_mcp_resources_list",
    "select_tool_registry",
    "shlex",
    "terminal_state",
    "tool_prefix",
    "validate_tool_result",
    "warnings",
)
FORBIDDEN = {
    "automations": {"tui", "cli"},
    "runtime": {"tui", "cli"},
    "core": {"providers", "tools", "tui", "cli"},
    "providers": {"tools", "tui", "cli"},
    "tools": {"providers", "tui", "cli"},
    "skills": {"providers", "tui", "cli"},
    "server": {"tui"},
    "tui": {"server"},
}


def _module_name(file_path: Path, root: Path = ROOT) -> str:
    relative = file_path.relative_to(root).with_suffix("")
    parts = ("zeta", *relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_from_target(
    file_path: Path,
    node: ast.ImportFrom,
    root: Path = ROOT,
) -> tuple[str, ...]:
    relative = file_path.relative_to(root).with_suffix("")
    package = ("zeta", *relative.parts[:-1])
    if node.level == 0:
        return tuple((node.module or "").split("."))
    package = package[: len(package) - node.level + 1]
    if node.module:
        return (*package, *node.module.split("."))
    return package


def _forbidden_imports(file_path: Path, root: Path = ROOT) -> list[str]:
    relative = file_path.relative_to(root)
    bucket = relative.parts[0] if len(relative.parts) > 1 else relative.stem
    forbidden = FORBIDDEN.get(bucket, set())
    if not forbidden:
        return []

    violations: list[str] = []
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    for node in ast.walk(tree):
        targets: list[tuple[str, ...]] = []
        if isinstance(node, ast.Import):
            targets = [tuple(alias.name.split(".")) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = _import_from_target(file_path, node, root)
            targets.append(target)
            if node.module is None:
                targets.extend((*target, alias.name) for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            targets.append(tuple(node.args[0].value.split(".")))

        for target in targets:
            if len(target) > 1 and target[0] == "zeta" and target[1] in forbidden:
                violations.append(f"{relative}:{node.lineno}: {'.'.join(target)}")
    return violations


def test_layer_modules_import_in_fresh_processes() -> None:
    source_root = str(ROOT.parent.parent)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, environment.get("PYTHONPATH")) if part
    )
    modules = sorted(
        _module_name(path)
        for path in ROOT.rglob("*.py")
        if path.relative_to(ROOT).parts[0] in FORBIDDEN
        and "tests" not in path.relative_to(ROOT).parts
    )

    failures: list[str] = []
    for module in modules:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; importlib.import_module(sys.argv[1])",
                module,
            ],
            cwd=ROOT.parent.parent,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            failures.append(f"{module}:\n{result.stderr}")
    assert not failures, "\n".join(failures)


def test_absolute_import_boundary_has_teeth(tmp_path: Path) -> None:
    mutated_root = tmp_path / "zeta"
    shutil.copytree(ROOT, mutated_root)
    loop_path = mutated_root / "core" / "loop.py"
    loop_path.write_text(
        "from zeta.tools import ToolRegistry\n" + loop_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    violations = [
        violation
        for file_path in mutated_root.rglob("*.py")
        if "tests" not in file_path.relative_to(mutated_root).parts
        for violation in _forbidden_imports(file_path, mutated_root)
    ]

    assert any(
        "core/loop.py" in violation and "zeta.tools" in violation
        for violation in violations
    ), "\n".join(violations)


def test_import_boundaries() -> None:
    violations = [
        violation
        for file_path in ROOT.rglob("*.py")
        if "tests" not in file_path.relative_to(ROOT).parts
        for violation in _forbidden_imports(file_path)
    ]
    assert not violations, "\n".join(violations)


def test_runtime_loop_preserves_legacy_exports() -> None:
    """Keep the complete legacy surface covered.

    Regenerate ``LEGACY_RUNTIME_LOOP_EXPORTS`` with:

        git show "$(git merge-base HEAD origin/master):harness/src/zeta/runtime/loop.py" | python -c 'import ast,json,sys; tree=ast.parse(sys.stdin.read()); names=set(); [names.add(node.name) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]; [names.add(target.id) for node in tree.body if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)) for target in (node.targets if isinstance(node,ast.Assign) else [node.target]) if isinstance(target,ast.Name)]; [names.add(alias.asname or alias.name.split(".")[0]) for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in node.names if alias.name != "*"]; print(json.dumps(sorted(name for name in names if not name.startswith("_")), indent=2))'

    ``annotations`` is included because the legacy module imported it from
    ``__future__``.
    """
    import importlib

    runtime_loop = importlib.import_module("zeta.runtime.loop")
    expected = set(LEGACY_RUNTIME_LOOP_EXPORTS)

    assert len(expected) == 101
    assert set(runtime_loop.__all__) == expected
    assert all(hasattr(runtime_loop, name) for name in expected)

    wildcard_namespace: dict[str, object] = {}
    exec("from zeta.loop import *", wildcard_namespace)  # noqa: S102
    assert {
        name for name in wildcard_namespace if not name.startswith("_")
    } == expected
