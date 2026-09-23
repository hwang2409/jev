"""Conservative page-state gates for browser actions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from zeta.providers import jev

PAGE_LOADED_AND_STABLE_THRESHOLD = 0.5
GOAL_ELEMENT_PRESENT_THRESHOLD = 0.5
ACTION_IS_NEXT_STEP_THRESHOLD = 0.5
DEAD_END_THRESHOLD = 0.5
DIFFERENT_APPROACH_THRESHOLD = 0.5
ACTION_SUCCEEDED_LOW_CONFIDENCE_RADIUS = 0.2

PageStateProvider = Callable[..., Awaitable[jev.BrowserPageStateResult]]


@dataclass(frozen=True, slots=True)
class PageStateDecision:
    allow_action: bool
    recovery: str | None
    error_kind: str | None


_PROVIDER_ERROR_DECISIONS = {
    "page_loaded_and_stable": PageStateDecision(
        False, "observe", "page_load_failed"
    ),
    "goal_element_present": PageStateDecision(
        False, "state", "goal_element_absent"
    ),
    "action_is_the_next_step": PageStateDecision(
        False, "state", "action_not_next_step"
    ),
    "action_succeeded": PageStateDecision(True, "state", None),
    "dead_end": PageStateDecision(False, "stop", "dead_end"),
    "needs_different_approach": PageStateDecision(
        False, "reroute", "different_approach"
    ),
}


def conservative_provider_error_decision(
    gate: str, _error: jev.JevRouterError
) -> PageStateDecision:
    """Map any Jev failure to the safe recovery direction for one gate."""

    try:
        return _PROVIDER_ERROR_DECISIONS[gate]
    except KeyError as exc:
        raise ValueError(f"unknown page-state gate: {gate}") from exc


def evaluate_page_state(
    *,
    page_loaded_and_stable: float,
    goal_element_present: float,
    action_is_the_next_step: float,
    action_succeeded: float | None,
    dead_end: float | None,
    needs_different_approach: float | None,
    deterministic_loaded: bool,
    deterministic_attached: bool,
) -> PageStateDecision:
    """Apply deterministic checks before Jev's conservative page-state gates."""

    if (
        not deterministic_loaded
        or page_loaded_and_stable < PAGE_LOADED_AND_STABLE_THRESHOLD
    ):
        return PageStateDecision(False, "observe", "page_load_failed")
    if (
        not deterministic_attached
        or goal_element_present < GOAL_ELEMENT_PRESENT_THRESHOLD
    ):
        error_kind = (
            "element_unavailable"
            if not deterministic_attached
            else "goal_element_absent"
        )
        return PageStateDecision(False, "state", error_kind)
    if dead_end is not None and dead_end >= DEAD_END_THRESHOLD:
        return PageStateDecision(False, "stop", "dead_end")
    if (
        needs_different_approach is not None
        and needs_different_approach >= DIFFERENT_APPROACH_THRESHOLD
    ):
        return PageStateDecision(False, "reroute", "different_approach")
    if action_is_the_next_step < ACTION_IS_NEXT_STEP_THRESHOLD:
        return PageStateDecision(False, "state", "action_not_next_step")
    if (
        action_succeeded is not None
        and (
            action_succeeded < 0.5
            or abs(action_succeeded - 0.5)
            <= ACTION_SUCCEEDED_LOW_CONFIDENCE_RADIUS
        )
    ):
        return PageStateDecision(True, "state", None)
    return PageStateDecision(True, None, None)


async def evaluate_page_state_with_provider(
    *,
    goal: str,
    action: str,
    page_state: dict[str, object],
    candidates: list[dict[str, object]],
    deterministic_loaded: bool,
    deterministic_attached: bool,
    recent_actions: list[str] | None = None,
    action_result: dict[str, object] | None = None,
    provider: PageStateProvider | None = None,
) -> PageStateDecision:
    """Judge page state through Jev, then apply the conservative control rule."""

    judge = provider or jev.judge_browser_page_state
    try:
        result = await judge(
            goal=goal,
            action=action,
            page_state=page_state,
            candidates=candidates,
            recent_actions=recent_actions,
            action_result=action_result,
        )
    except jev.JevRouterError as exc:
        return conservative_provider_error_decision("page_loaded_and_stable", exc)
    return evaluate_page_state(
        page_loaded_and_stable=result.page_loaded_and_stable,
        goal_element_present=result.goal_element_present,
        action_is_the_next_step=result.action_is_the_next_step,
        action_succeeded=result.action_succeeded,
        dead_end=result.dead_end,
        needs_different_approach=result.needs_different_approach,
        deterministic_loaded=deterministic_loaded,
        deterministic_attached=deterministic_attached,
    )
