"""The Jev-backed tool router."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypedDict

from ..execution import ToolExecutionContext
from ..providers.jev import route_step
from ..types import StructuredToolResult
from .registry import ToolRegistry, _error_result, _success_result, text_block


class RouteArguments(TypedDict):
    step: str


ROUTE_TOPK_CONFIDENCE = 0.8
"""Choice cutoff; thresholds do not transfer (Jev jaggedness section 8)."""
# TODO: calibrate this Choice threshold with route confidence data.

LOW_CLARITY_NUDGE = 0.3
"""Noul cutoff; thresholds do not transfer (Jev jaggedness section 8)."""
# TODO: calibrate this Noul threshold with step-clarity data.

_BOUNDARIES: dict[str, tuple[str, list[str]]] = {
    "read": (
        "Searching file contents by pattern; use grep or the matching search tool.",
        ["Read src/zeta/loop.py to inspect the routing logic."],
    ),
    "write": (
        "Changing part of an existing file; use edit.",
        ["Write a new notes.md file with the requested content."],
    ),
    "edit": (
        "Creating or replacing a whole file; use write.",
        ["Replace one old threshold in src/zeta/loop.py with a new constant."],
    ),
    "fetch": (
        "Searching the web for relevant pages; use websearch.",
        ["Fetch https://docs.typesafe.ai/primitives/advanced.md."],
    ),
    "websearch": (
        "Opening a known URL; use fetch.",
        ["Search the web for the latest TypeSafe routing documentation."],
    ),
    "bash": (
        "editing a file in place; use edit",
        ["Run pytest tests/test_router_auto.py."],
    ),
    "exec": (
        "Running a session shell command with persistent shell state; use bash.",
        ["Run pytest with a 60 second timeout and bounded output."],
    ),
}


def build_catalog(
    schemas: Iterable[Mapping[str, object]],
    *,
    excluded_names: set[str] | frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for schema in schemas:
        name = schema.get("name")
        if not isinstance(name, str) or name in excluded_names:
            continue
        description = schema.get("description", "")
        if not isinstance(description, str):
            description = ""
        lines = description.splitlines()
        what = (lines[0] if lines else "")[:150]
        not_for, examples = _BOUNDARIES.get(
            name,
            (
                "Actions outside this tool's description; use the matching tool.",
                [f"Use {name} for the action described by its tool description."],
            ),
        )
        catalog[name] = {
            "what": what,
            "not_for": not_for,
            "examples": examples,
        }
    return catalog


def _catalog(registry: ToolRegistry) -> dict[str, dict[str, Any]]:
    return build_catalog(registry.schemas, excluded_names={"route"})


def _route_error(message: str) -> StructuredToolResult:
    return _error_result(message, kind="error")


def _route_text(tool: str, probabilities: dict[str, float], confidence: float) -> str:
    if confidence >= ROUTE_TOPK_CONFIDENCE:
        return f"routed to {tool} (confidence {confidence:.2f})"
    top_tools = sorted(
        probabilities.items(), key=lambda item: item[1], reverse=True
    )[:3]
    choices = ", ".join(f"{name} ({probability:.2f})" for name, probability in top_tools)
    return f"routed tools: {choices}"


async def _route(
    registry: ToolRegistry,
    arguments: RouteArguments,
    *,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult:
    recent_steps = (
        execution_context.router_recent_steps
        if execution_context is not None
        else None
    )
    try:
        result = await route_step(
            arguments["step"],
            _catalog(registry),
            list(recent_steps or []),
        )
        routed_tools = (
            [result.tool]
            if result.confidence >= ROUTE_TOPK_CONFIDENCE
            else [
                name
                for name, _probability in sorted(
                    result.probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
            ]
        )
        available = registry.registered_names - {"route"}
        routed_tools = [name for name in routed_tools if name in available]
        if not routed_tools:
            raise ValueError(f"Jev selected no registered tool for {result.tool!r}")
        if execution_context is not None:
            sink = execution_context.router_tools_sink
            if sink is not None:
                sink(routed_tools)
        message = _route_text(result.tool, result.probabilities, result.confidence)
        if result.step_clarity < LOW_CLARITY_NUDGE:
            message += "; restate the step more concretely"
        telemetry: dict[str, Any] = {
            "service": "jev",
            "usage": dict(result.usage),
            "confidence": result.confidence,
            "needs_tool": result.needs_tool,
            "step_clarity": result.step_clarity,
        }
        if result.call_confidence is not None:
            telemetry["call_confidence"] = result.call_confidence
        return _success_result(text_block(message), structured_content=telemetry)
    except Exception as exc:  # noqa: BLE001 - routing must fail open
        if execution_context is not None:
            sink = execution_context.router_tools_sink
            if sink is not None:
                sink(None)
        return _route_error(f"router failed: {exc}")
    finally:
        if recent_steps is not None:
            recent_steps.append(arguments["step"])
            del recent_steps[:-5]


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "route",
        _route,
        description="Describe the next concrete action so Jev can choose a tool.",
        parameters={
            "type": "object",
            "properties": {"step": {"type": "string"}},
            "required": ["step"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
