"""Conservative page-state gates for browser actions."""

from __future__ import annotations

from dataclasses import dataclass

PAGE_LOADED_AND_STABLE_THRESHOLD = 0.5
GOAL_ELEMENT_PRESENT_THRESHOLD = 0.5
ACTION_IS_NEXT_STEP_THRESHOLD = 0.5
DEAD_END_THRESHOLD = 0.5
DIFFERENT_APPROACH_THRESHOLD = 0.5
ACTION_SUCCEEDED_LOW_CONFIDENCE_RADIUS = 0.2


@dataclass(frozen=True, slots=True)
class PageStateDecision:
    allow_action: bool
    recovery: str | None
    error_kind: str | None


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
        and abs(action_succeeded - 0.5) < ACTION_SUCCEEDED_LOW_CONFIDENCE_RADIUS
    ):
        return PageStateDecision(True, "state", None)
    return PageStateDecision(True, None, None)
