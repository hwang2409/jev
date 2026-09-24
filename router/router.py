"""Jev-backed tool router: one client call routes an agent step to a tool."""

from collections.abc import Mapping
from dataclasses import dataclass

from jm.client import JevClient, JevResponse

from catalog import CATALOG


@dataclass
class RouteResult:
    tool: str
    probabilities: dict[str, float]
    confidence: float
    needs_tool: float
    step_clarity: float
    usage: dict[str, int]
    calls: int = 1
    latency_ms: int | None = None
    category: str | None = None
    category_confidence: float | None = None
    category_probabilities: dict[str, float] | None = None


GATE_QUESTIONS = {
    "needs_tool": {
        "type": "noul",
        "instructions": (
            "Does the current step require calling a tool, rather than "
            "the agent answering or reasoning directly from what it "
            "already knows?"
        ),
    },
    "step_clarity": {
        "type": "noul",
        "instructions": (
            "Is the current step description specific enough to route "
            "to a single tool with confidence?"
        ),
    },
}


def build_request(
    task: str, step: str, history: list[str], catalog: dict[str, str]
) -> dict:
    body = build_choice_request(
        task,
        step,
        history,
        "tool",
        "An agent is working on the task and describes its current step. Which "
        "single tool should it call to accomplish this step?",
        catalog,
    )
    body["questions"].update(GATE_QUESTIONS)
    return body


def build_choice_request(
    task: str,
    step: str,
    history: list[str],
    choice_name: str,
    instructions: str,
    criteria: dict[str, str],
) -> dict:
    return {
        "state": {
            "task": task,
            "current_step": step,
            "recent_steps": list(history[-5:]),
        },
        "questions": {
            choice_name: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            },
        },
    }


def parse_response(response: JevResponse | Mapping[str, object]) -> RouteResult:
    answers = _response_field(response, "answers")
    tool = answers["tool"]
    return RouteResult(
        tool=_answer_field(tool, "choice"),
        probabilities=_answer_field(tool, "probabilities"),
        confidence=_answer_field(tool, "confidence"),
        needs_tool=_answer_field(answers["needs_tool"], "noul"),
        step_clarity=_answer_field(answers["step_clarity"], "noul"),
        usage=dict(_response_field(response, "usage") or {}),
        latency_ms=getattr(response, "latency_ms", None),
    )


def route(
    task: str,
    step: str,
    history: list[str] | None = None,
    catalog: dict[str, str] | None = None,
    client: JevClient | None = None,
) -> RouteResult:
    body = build_request(task, step, history or [], catalog or CATALOG)
    jev_client = client if client is not None else JevClient()
    response = jev_client.evaluate(body["state"], body["questions"])
    return parse_response(response)


def _response_field(response: JevResponse | Mapping[str, object], name: str):
    if isinstance(response, Mapping):
        return response[name]
    return getattr(response, name)


def _answer_field(answer: object, name: str):
    if isinstance(answer, Mapping):
        return answer[name]
    return getattr(answer, name)
