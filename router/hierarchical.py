"""Two-stage category-then-tool router for large catalogs."""

from jm.client import JevClient

from catalogs import category_catalogs
from router import (
    RouteResult,
    _answer_field,
    _response_field,
    build_choice_request,
)

CATEGORY_INSTRUCTIONS = (
    "Which single category contains the tool that best accomplishes the "
    "agent's current step?"
)
TOOL_INSTRUCTIONS = (
    "Which single tool in this category should the agent call to accomplish "
    "the current step?"
)


def _answer(response: object, question: str) -> object:
    return _response_field(response, "answers")[question]


def _usage(response: object) -> dict[str, int]:
    return dict(_response_field(response, "usage") or {})


def _latency(response: object) -> int:
    return int(getattr(response, "latency_ms", 0) or 0)


def _add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    keys = set(left) | set(right)
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in keys}


def route_hierarchical(
    task: str,
    step: str,
    history: list[str] | None = None,
    catalog: dict[str, str] | None = None,
    client: JevClient | None = None,
) -> RouteResult:
    """Route through one category Choice and one within-category Choice."""
    selected_catalog = catalog or {}
    groups = category_catalogs(selected_catalog)
    if not groups:
        raise ValueError("hierarchical routing requires a non-empty catalog")

    history = history or []
    category_criteria = {
        name: f"Tools for {name.replace('_', ' ')} operations."
        for name in groups
    }
    category_request = build_choice_request(
        task,
        step,
        history,
        "category",
        CATEGORY_INSTRUCTIONS,
        category_criteria,
    )
    jev_client = client if client is not None else JevClient()
    category_response = jev_client.evaluate(
        category_request["state"], category_request["questions"]
    )
    category_answer = _answer(category_response, "category")
    category = _answer_field(category_answer, "choice")
    if category not in groups:
        raise ValueError(f"unknown category returned by Jev: {category}")

    tool_request = build_choice_request(
        task,
        step,
        history,
        "tool",
        TOOL_INSTRUCTIONS,
        groups[category],
    )
    tool_request["state"]["selected_category"] = category
    tool_response = jev_client.evaluate(
        tool_request["state"], tool_request["questions"]
    )
    tool_answer = _answer(tool_response, "tool")
    category_confidence = float(_answer_field(category_answer, "confidence") or 0.0)
    tool_confidence = float(_answer_field(tool_answer, "confidence") or 0.0)
    return RouteResult(
        tool=_answer_field(tool_answer, "choice"),
        probabilities=dict(_answer_field(tool_answer, "probabilities")),
        confidence=min(category_confidence, tool_confidence),
        needs_tool=1.0,
        step_clarity=1.0,
        usage=_add_usage(_usage(category_response), _usage(tool_response)),
        calls=2,
        latency_ms=_latency(category_response) + _latency(tool_response),
        category=category,
        category_confidence=category_confidence,
    )


__all__ = ["route_hierarchical"]
