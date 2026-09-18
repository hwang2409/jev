"""The Jev-backed tool router."""

from __future__ import annotations

from typing import TypedDict

from ..execution import ToolExecutionContext
from ..providers.jev import route_step
from ..types import StructuredToolResult
from .registry import ToolRegistry, _error_result, _success_result, text_block


class RouteArguments(TypedDict):
    step: str


def _catalog(registry: ToolRegistry) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for schema in registry.schemas:
        name = schema.get("name")
        if not isinstance(name, str) or name == "route":
            continue
        description = schema.get("description", "")
        if not isinstance(description, str):
            description = ""
        lines = description.splitlines()
        catalog[name] = (lines[0] if lines else "")[:150]
    return catalog


def _route_error(message: str) -> StructuredToolResult:
    return _error_result(message, kind="error")


def _route_text(tool: str, probabilities: dict[str, float], confidence: float) -> str:
    if confidence >= 0.8:
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
            if result.confidence >= 0.8
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
        if result.step_clarity < 0.3:
            message += "; restate the step more concretely"
        return _success_result(text_block(message))
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
