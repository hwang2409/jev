"""Conservative page-state gates for browser actions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ...protocol import jev

PAGE_LOADED_AND_STABLE_THRESHOLD = 0.5
GOAL_ELEMENT_PRESENT_THRESHOLD = 0.5
ACTION_IS_NEXT_STEP_THRESHOLD = 0.5
DEAD_END_THRESHOLD = 0.5
DIFFERENT_APPROACH_THRESHOLD = 0.5
ACTION_SUCCEEDED_LOW_CONFIDENCE_RADIUS = 0.2
# The arc-3 browsing spec allows one bounded recovery observation, then stops.
PAGE_STATE_RECOVERY_ATTEMPT_CAP = 1
ACTION_OUTCOME_UNKNOWN = "action_outcome_unknown"
LAST_RESORT_PROVIDER_ERROR = "provider_error"

PageStateProvider = Callable[..., Awaitable[jev.BrowserPageStateResult]]


@dataclass(frozen=True, slots=True)
class PageStateDecision:
    allow_action: bool
    recovery: str | None
    error_kind: str | None
    recovery_attempts: int = 0

    @property
    def attempt(self) -> int:
        """Expose the recovery counter for callers that track one attempt."""

        return self.recovery_attempts


_PROVIDER_ERROR_DECISIONS = {
    "page_loaded_and_stable": PageStateDecision(False, "observe", "page_load_failed"),
    "goal_element_present": PageStateDecision(False, "state", "goal_element_absent"),
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
    gate: str | None, _error: jev.JevRouterError
) -> PageStateDecision:
    """Map any Jev failure to the safe recovery direction for one gate."""

    if gate is None:
        return PageStateDecision(False, "observe", LAST_RESORT_PROVIDER_ERROR)
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
    page_state: dict[str, object] | None = None,
    previous_page_state: dict[str, object] | None = None,
    recovery_attempts: int = 0,
) -> PageStateDecision:
    """Apply deterministic checks before Jev's conservative page-state gates."""

    if recovery_attempts >= PAGE_STATE_RECOVERY_ATTEMPT_CAP:
        return PageStateDecision(
            False,
            None,
            ACTION_OUTCOME_UNKNOWN,
            recovery_attempts,
        )
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
    if _action_succeeded_is_uncertain(action_succeeded):
        return _bound_action_recovery(
            page_state=page_state,
            previous_page_state=previous_page_state,
            recovery_attempts=recovery_attempts,
        )
    return PageStateDecision(True, None, None)


def _catalog_identity(
    page_state: dict[str, object] | None,
) -> tuple[int | None, int | None] | None:
    if page_state is None:
        return None
    snapshot_id = page_state.get("snapshot_id")
    generation = page_state.get("generation")
    if type(snapshot_id) is not int:
        snapshot_id = None
    if type(generation) is not int:
        generation = None
    if snapshot_id is None and generation is None:
        return None
    return snapshot_id, generation


def _action_succeeded_is_uncertain(action_succeeded: float | None) -> bool:
    return action_succeeded is not None and (
        action_succeeded < 0.5
        or abs(action_succeeded - 0.5) <= ACTION_SUCCEEDED_LOW_CONFIDENCE_RADIUS
    )


def _bound_action_recovery(
    *,
    page_state: dict[str, object] | None,
    previous_page_state: dict[str, object] | None,
    recovery_attempts: int,
) -> PageStateDecision:
    next_attempt = min(recovery_attempts + 1, PAGE_STATE_RECOVERY_ATTEMPT_CAP)
    unchanged = _catalog_identity(
        previous_page_state
    ) is not None and _catalog_identity(page_state) == _catalog_identity(
        previous_page_state
    )
    if recovery_attempts >= PAGE_STATE_RECOVERY_ATTEMPT_CAP or unchanged:
        return PageStateDecision(False, None, ACTION_OUTCOME_UNKNOWN, next_attempt)
    return PageStateDecision(True, "state", None, next_attempt)


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
    gate: str | None = None,
    recovery_attempts: int = 0,
    previous_page_state: dict[str, object] | None = None,
) -> PageStateDecision:
    """Judge page state through Jev, then apply the conservative control rule."""

    if recovery_attempts >= PAGE_STATE_RECOVERY_ATTEMPT_CAP:
        return PageStateDecision(
            False,
            None,
            ACTION_OUTCOME_UNKNOWN,
            recovery_attempts,
        )
    judge = provider or jev.judge_browser_page_state
    judge_kwargs: dict[str, object] = {
        "goal": goal,
        "action": action,
        "page_state": page_state,
        "candidates": candidates,
        "recent_actions": recent_actions,
        "action_result": action_result,
    }
    if gate is not None:
        judge_kwargs["gate"] = gate
    try:
        result = await judge(**judge_kwargs)
    except jev.JevRouterError as exc:
        return conservative_provider_error_decision(exc.gate, exc)
    return evaluate_page_state(
        page_loaded_and_stable=result.page_loaded_and_stable,
        goal_element_present=result.goal_element_present,
        action_is_the_next_step=result.action_is_the_next_step,
        action_succeeded=result.action_succeeded,
        dead_end=result.dead_end,
        needs_different_approach=result.needs_different_approach,
        deterministic_loaded=deterministic_loaded,
        deterministic_attached=deterministic_attached,
        page_state=page_state,
        previous_page_state=previous_page_state,
        recovery_attempts=recovery_attempts,
    )
