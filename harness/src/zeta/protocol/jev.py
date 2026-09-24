"""Provider-neutral contracts for Jev-backed decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class JevRouterError(RuntimeError):
    """Raised when the configured Jev provider cannot classify a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        gate: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.gate = gate


@dataclass(frozen=True, slots=True)
class RouteResult:
    tool: str
    probabilities: dict[str, float]
    confidence: float
    needs_tool: float
    step_clarity: float
    usage: dict[str, int]
    call_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class AutoRouteResult:
    tool: str
    probabilities: dict[str, float]
    confidence: float
    needs_tool: float
    usage: dict[str, int]
    call_confidence: float | None = None
    memory_relevance: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class MemoryRelevanceResult:
    scores: dict[str, float]
    usage: dict[str, int]


@dataclass(frozen=True, slots=True)
class TriageResult:
    keep_probabilities: dict[str, float]
    usage: dict[str, int]
    call_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SafetyScoreResult:
    score: int
    probabilities: dict[str, float]
    confidence: float
    touches_outside_cwd: float
    plausibly_irreversible: float
    usage: dict[str, int]
    call_confidence: float


@dataclass(frozen=True, slots=True)
class BrowserElementChoiceResult:
    element_id: str | None
    affordance: str | None
    candidate_ids: tuple[str, ...]
    probabilities: dict[str, float]
    confidence: float
    goal_element_present: float
    page_loaded_and_stable: float
    action_is_the_next_step: float
    usage: dict[str, int]
    call_confidence: float


@dataclass(frozen=True, slots=True)
class SearchResultScoreResult:
    scores: dict[str, float]
    confidence: float
    usage: dict[str, int]
    call_confidence: float


@dataclass(frozen=True, slots=True)
class BrowserPageStateResult:
    page_loaded_and_stable: float
    goal_element_present: float
    action_is_the_next_step: float
    action_succeeded: float | None
    dead_end: float
    needs_different_approach: float
    usage: dict[str, int]
    call_confidence: float


_provider: object | None = None


def register_provider(provider: object) -> None:
    """Register the provider implementation at the composition boundary."""

    global _provider
    _provider = provider


def _implementation(name: str) -> Any:
    if _provider is None:
        raise JevRouterError("no Jev provider has been configured")
    implementation = getattr(_provider, name, None)
    if not callable(implementation):
        raise JevRouterError(f"configured Jev provider has no {name} implementation")
    return implementation


async def route_step(
    step: str,
    catalog: dict[str, dict[str, Any]],
    history: list[str] | None = None,
) -> RouteResult:
    return await _implementation("route_step")(step, catalog, history)


async def triage(
    task: str,
    items: list[dict[str, str]],
    *,
    latest_assistant_text: str = "",
    recent_tool_actions: list[str] | None = None,
) -> TriageResult:
    return await _implementation("triage")(
        task,
        items,
        latest_assistant_text=latest_assistant_text,
        recent_tool_actions=recent_tool_actions,
    )


async def safety_score(
    command: str,
    cwd: str,
    task_excerpt: str = "",
) -> SafetyScoreResult:
    return await _implementation("safety_score")(command, cwd, task_excerpt)


async def choose_browser_element(
    goal: str,
    action: str,
    page_state: dict[str, object],
    candidates: list[dict[str, object]],
    recent_actions: list[str] | None = None,
) -> BrowserElementChoiceResult:
    return await _implementation("choose_browser_element")(
        goal, action, page_state, candidates, recent_actions
    )


async def score_search_results(
    goal: str, items: list[dict[str, str]]
) -> SearchResultScoreResult:
    return await _implementation("score_search_results")(goal, items)


async def judge_browser_page_state(
    goal: str,
    action: str,
    page_state: dict[str, object],
    candidates: list[dict[str, object]],
    recent_actions: list[str] | None = None,
    action_result: dict[str, object] | None = None,
    gate: str | None = None,
) -> BrowserPageStateResult:
    return await _implementation("judge_browser_page_state")(
        goal,
        action,
        page_state,
        candidates,
        recent_actions,
        action_result,
        gate,
    )


__all__ = [
    "AutoRouteResult",
    "BrowserElementChoiceResult",
    "BrowserPageStateResult",
    "JevRouterError",
    "MemoryRelevanceResult",
    "RouteResult",
    "SafetyScoreResult",
    "SearchResultScoreResult",
    "TriageResult",
    "choose_browser_element",
    "judge_browser_page_state",
    "register_provider",
    "route_step",
    "safety_score",
    "score_search_results",
    "triage",
]
