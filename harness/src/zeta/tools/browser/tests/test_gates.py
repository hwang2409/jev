from __future__ import annotations

import pytest

from zeta.tools.browser.gates import PageStateDecision, evaluate_page_state


@pytest.mark.parametrize(
    (
        "loaded",
        "present",
        "succeeded",
        "deterministic_loaded",
        "deterministic_attached",
        "expected",
    ),
    [
        (
            0.1,
            0.9,
            None,
            True,
            True,
            PageStateDecision(False, "observe", "page_load_failed"),
        ),
        (
            0.9,
            0.1,
            None,
            True,
            True,
            PageStateDecision(False, "state", "goal_element_absent"),
        ),
        (0.9, 0.9, 0.55, True, True, PageStateDecision(True, "state", None)),
        (
            0.9,
            0.9,
            0.9,
            False,
            True,
            PageStateDecision(False, "observe", "page_load_failed"),
        ),
        (
            0.9,
            0.9,
            0.9,
            True,
            False,
            PageStateDecision(False, "state", "element_unavailable"),
        ),
    ],
)
def test_page_state_control_rule(
    loaded: float,
    present: float,
    succeeded: float | None,
    deterministic_loaded: bool,
    deterministic_attached: bool,
    expected: PageStateDecision,
) -> None:
    assert (
        evaluate_page_state(
            page_loaded_and_stable=loaded,
            goal_element_present=present,
            action_is_the_next_step=0.9,
            action_succeeded=succeeded,
            dead_end=0.1,
            needs_different_approach=0.1,
            deterministic_loaded=deterministic_loaded,
            deterministic_attached=deterministic_attached,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("dead_end", "needs_different_approach", "expected"),
    [
        (0.5, 0.1, PageStateDecision(False, "stop", "dead_end")),
        (0.1, 0.5, PageStateDecision(False, "reroute", "different_approach")),
    ],
)
def test_page_state_requests_recovery_for_terminal_or_mismatched_state(
    dead_end: float,
    needs_different_approach: float,
    expected: PageStateDecision,
) -> None:
    assert (
        evaluate_page_state(
            page_loaded_and_stable=0.9,
            goal_element_present=0.9,
            action_is_the_next_step=0.9,
            action_succeeded=0.9,
            dead_end=dead_end,
            needs_different_approach=needs_different_approach,
            deterministic_loaded=True,
            deterministic_attached=True,
        )
        == expected
    )


def test_page_state_blocks_an_action_that_is_not_the_next_step() -> None:
    assert evaluate_page_state(
        page_loaded_and_stable=0.9,
        goal_element_present=0.9,
        action_is_the_next_step=0.49,
        action_succeeded=None,
        dead_end=None,
        needs_different_approach=None,
        deterministic_loaded=True,
        deterministic_attached=True,
    ) == PageStateDecision(False, "state", "action_not_next_step")


def test_page_state_allows_a_clear_action_without_post_action_evidence() -> None:
    assert evaluate_page_state(
        page_loaded_and_stable=0.9,
        goal_element_present=0.9,
        action_is_the_next_step=0.9,
        action_succeeded=None,
        dead_end=None,
        needs_different_approach=None,
        deterministic_loaded=True,
        deterministic_attached=True,
    ) == PageStateDecision(True, None, None)
