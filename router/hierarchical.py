"""Two-stage category-then-tool router for large catalogs."""

from jm.client import JevClient

from catalogs import CATEGORY_DESCRIPTIONS, category_catalogs, primary_category_for_tool
from router import (
    GATE_QUESTIONS,
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
CATEGORY_RESCUE_THRESHOLD = 0.8
CATEGORY_RESCUE_COUNT = 3


def _answer(response: object, question: str) -> object:
    return _response_field(response, "answers")[question]


def _usage(response: object) -> dict[str, int]:
    return dict(_response_field(response, "usage") or {})


def _latency(response: object) -> int:
    return int(getattr(response, "latency_ms", 0) or 0)


def _add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    keys = set(left) | set(right)
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in keys}


def _top_categories(probabilities: dict[str, float]) -> list[str]:
    return sorted(probabilities, key=probabilities.get, reverse=True)[
        :CATEGORY_RESCUE_COUNT
    ]


def _end_to_end_probabilities(
    catalog: dict[str, str],
    category_probabilities: dict[str, float],
    tool_probabilities: dict[str, float],
) -> dict[str, float]:
    return {
        tool: round(
            category_probabilities.get(primary_category_for_tool(tool), 0.0)
            * tool_probabilities.get(tool, 0.0),
            8,
        )
        for tool in catalog
    }


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
        name: CATEGORY_DESCRIPTIONS[name] for name in groups
    }
    category_request = build_choice_request(
        task,
        step,
        history,
        "category",
        CATEGORY_INSTRUCTIONS,
        category_criteria,
    )
    category_request["questions"].update(GATE_QUESTIONS)
    jev_client = client if client is not None else JevClient()
    category_response = jev_client.evaluate(
        category_request["state"], category_request["questions"]
    )
    category_answer = _answer(category_response, "category")
    category = _answer_field(category_answer, "choice")
    if category not in groups:
        raise ValueError(f"unknown category returned by Jev: {category}")

    category_probabilities = dict(_answer_field(category_answer, "probabilities"))
    category_confidence = float(_answer_field(category_answer, "confidence") or 0.0)
    if category_confidence < CATEGORY_RESCUE_THRESHOLD:
        candidate_categories = _top_categories(category_probabilities)
    else:
        candidate_categories = [category]
    candidate_categories = [
        candidate for candidate in candidate_categories if candidate in groups
    ]
    if category not in candidate_categories:
        candidate_categories.insert(0, category)

    tool_criteria = {}
    for candidate in candidate_categories:
        tool_criteria.update(groups[candidate])
    tool_request = build_choice_request(
        task,
        step,
        history,
        "tool",
        TOOL_INSTRUCTIONS,
        tool_criteria,
    )
    tool_request["state"]["selected_category"] = category
    tool_request["state"]["candidate_categories"] = candidate_categories
    tool_response = jev_client.evaluate(
        tool_request["state"], tool_request["questions"]
    )
    tool_answer = _answer(tool_response, "tool")
    tool_confidence = float(_answer_field(tool_answer, "confidence") or 0.0)
    return RouteResult(
        tool=_answer_field(tool_answer, "choice"),
        probabilities=_end_to_end_probabilities(
            selected_catalog,
            category_probabilities,
            dict(_answer_field(tool_answer, "probabilities")),
        ),
        confidence=min(category_confidence, tool_confidence),
        needs_tool=float(
            _answer_field(_answer(category_response, "needs_tool"), "noul")
        ),
        step_clarity=float(
            _answer_field(_answer(category_response, "step_clarity"), "noul")
        ),
        usage=_add_usage(_usage(category_response), _usage(tool_response)),
        calls=2,
        latency_ms=_latency(category_response) + _latency(tool_response),
        category=category,
        category_confidence=category_confidence,
        category_probabilities=category_probabilities,
    )


__all__ = ["route_hierarchical"]
