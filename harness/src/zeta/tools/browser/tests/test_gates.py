from __future__ import annotations

from typing import Self

import pytest

from zeta.providers import jev
from zeta.tools.browser.gates import (
    PAGE_STATE_RECOVERY_ATTEMPT_CAP,
    PageStateDecision,
    conservative_provider_error_decision,
    evaluate_page_state,
    evaluate_page_state_with_provider,
)


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
        (0.9, 0.9, 0.55, True, True, PageStateDecision(True, "state", None, 1)),
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


@pytest.mark.parametrize(
    ("action_succeeded", "expected"),
    [
        (0.1, PageStateDecision(True, "state", None, 1)),
        (0.3, PageStateDecision(True, "state", None, 1)),
        (0.5, PageStateDecision(True, "state", None, 1)),
        (0.7, PageStateDecision(True, "state", None, 1)),
        (0.9, PageStateDecision(True, None, None)),
    ],
)
def test_action_failure_or_uncertainty_requests_a_fresh_state(
    action_succeeded: float, expected: PageStateDecision
) -> None:
    assert evaluate_page_state(
        page_loaded_and_stable=0.9,
        goal_element_present=0.9,
        action_is_the_next_step=0.9,
        action_succeeded=action_succeeded,
        dead_end=None,
        needs_different_approach=None,
        deterministic_loaded=True,
        deterministic_attached=True,
    ) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gate",
    [
        "page_loaded_and_stable",
        "goal_element_present",
        "action_is_the_next_step",
        "dead_end",
        "needs_different_approach",
        "action_succeeded",
    ],
)
@pytest.mark.parametrize("error_kind", ["timeout", "malformed", "api_error"])
async def test_real_provider_errors_follow_the_failed_gate(
    monkeypatch: pytest.MonkeyPatch, gate: str, error_kind: str
) -> None:
    class Response:
        status_code = 500 if error_kind == "api_error" else 200
        text = "backend error"

        def json(self) -> dict[str, object]:
            if error_kind == "malformed":
                answers = {
                    name: {"noul": 0.9}
                    for name in (
                        "page_loaded_and_stable",
                        "goal_element_present",
                        "action_is_the_next_step",
                        "action_succeeded",
                        "dead_end",
                        "needs_different_approach",
                    )
                    if name != gate
                }
                if gate == "action_succeeded":
                    answers[gate] = {"noul": "not-a-number"}
                return {"answers": answers}
            return {}

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(self, _url: str, **_kwargs: object) -> Response:
            if error_kind == "timeout":
                raise jev.httpx.ReadTimeout("timed out")
            return Response()

    monkeypatch.setattr(jev.httpx, "AsyncClient", Client)
    monkeypatch.setenv("JEV_API_KEY", "test-key")

    expected = {
        "page_loaded_and_stable": PageStateDecision(
            False, "observe", "page_load_failed"
        ),
        "goal_element_present": PageStateDecision(
            False, "state", "goal_element_absent"
        ),
        "action_is_the_next_step": PageStateDecision(
            False, "state", "action_not_next_step"
        ),
        "dead_end": PageStateDecision(False, "stop", "dead_end"),
        "needs_different_approach": PageStateDecision(
            False, "reroute", "different_approach"
        ),
        "action_succeeded": PageStateDecision(True, "state", None),
    }[gate]

    assert await evaluate_page_state_with_provider(
        goal="continue",
        action="click",
        page_state={},
        candidates=[],
        deterministic_loaded=True,
        deterministic_attached=True,
        action_result={"changed": True},
        gate=gate,
    ) == expected


@pytest.mark.asyncio
async def test_malformed_non_active_gate_uses_active_gate_for_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status_code = 200
        text = "backend error"

        def json(self) -> dict[str, object]:
            answers = {
                name: {"noul": 0.9}
                for name in (
                    "page_loaded_and_stable",
                    "goal_element_present",
                    "action_is_the_next_step",
                    "action_succeeded",
                    "dead_end",
                    "needs_different_approach",
                )
            }
            answers["goal_element_present"] = {"noul": "not-a-number"}
            return {"answers": answers}

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(self, _url: str, **_kwargs: object) -> Response:
            return Response()

    monkeypatch.setattr(jev.httpx, "AsyncClient", Client)
    monkeypatch.setenv("JEV_API_KEY", "test-key")

    assert await evaluate_page_state_with_provider(
        goal="continue",
        action="click",
        page_state={},
        candidates=[],
        deterministic_loaded=True,
        deterministic_attached=True,
        gate="page_loaded_and_stable",
    ) == PageStateDecision(False, "observe", "page_load_failed")


def test_untagged_provider_failure_uses_distinct_last_resort_decision() -> None:
    assert conservative_provider_error_decision(
        None, jev.JevRouterError("provider failure")
    ) == PageStateDecision(False, "observe", "provider_error")


@pytest.mark.asyncio
async def test_uncertain_action_with_changed_state_increments_recovery_attempt() -> None:
    async def judge(**_kwargs: object) -> jev.BrowserPageStateResult:
        return jev.BrowserPageStateResult(
            page_loaded_and_stable=0.9,
            goal_element_present=0.9,
            action_is_the_next_step=0.9,
            action_succeeded=0.5,
            dead_end=0.1,
            needs_different_approach=0.1,
            usage={},
            call_confidence=0.0,
        )

    decision = await evaluate_page_state_with_provider(
        goal="continue",
        action="click",
        page_state={"snapshot_id": 2, "generation": 1},
        candidates=[],
        deterministic_loaded=True,
        deterministic_attached=True,
        provider=judge,
        previous_page_state={"snapshot_id": 1, "generation": 1},
    )

    assert decision == PageStateDecision(True, "state", None, 1)


@pytest.mark.asyncio
async def test_uncertain_action_with_unchanged_state_reports_unknown_outcome() -> None:
    async def judge(**_kwargs: object) -> jev.BrowserPageStateResult:
        return jev.BrowserPageStateResult(
            page_loaded_and_stable=0.9,
            goal_element_present=0.9,
            action_is_the_next_step=0.9,
            action_succeeded=0.5,
            dead_end=0.1,
            needs_different_approach=0.1,
            usage={},
            call_confidence=0.0,
        )

    decision = await evaluate_page_state_with_provider(
        goal="continue",
        action="click",
        page_state={"snapshot_id": 1, "generation": 1},
        candidates=[],
        deterministic_loaded=True,
        deterministic_attached=True,
        provider=judge,
        previous_page_state={"snapshot_id": 1, "generation": 1},
    )

    assert decision == PageStateDecision(False, None, "action_outcome_unknown", 1)


@pytest.mark.asyncio
async def test_uncertain_action_at_recovery_cap_reports_unknown_outcome() -> None:
    called = False

    async def judge(**_kwargs: object) -> jev.BrowserPageStateResult:
        nonlocal called
        called = True
        return jev.BrowserPageStateResult(
            page_loaded_and_stable=0.9,
            goal_element_present=0.9,
            action_is_the_next_step=0.9,
            action_succeeded=0.5,
            dead_end=0.1,
            needs_different_approach=0.1,
            usage={},
            call_confidence=0.0,
        )

    decision = await evaluate_page_state_with_provider(
        goal="continue",
        action="click",
        page_state={"snapshot_id": 2, "generation": 1},
        candidates=[],
        deterministic_loaded=True,
        deterministic_attached=True,
        provider=judge,
        recovery_attempts=PAGE_STATE_RECOVERY_ATTEMPT_CAP,
        previous_page_state={"snapshot_id": 1, "generation": 1},
    )

    assert decision == PageStateDecision(
        False, None, "action_outcome_unknown", PAGE_STATE_RECOVERY_ATTEMPT_CAP
    )
    assert called is False


@pytest.mark.asyncio
async def test_provider_result_drives_all_page_state_gates() -> None:
    async def judge(**_kwargs: object) -> jev.BrowserPageStateResult:
        return jev.BrowserPageStateResult(
            page_loaded_and_stable=0.9,
            goal_element_present=0.9,
            action_is_the_next_step=0.9,
            action_succeeded=0.1,
            dead_end=0.1,
            needs_different_approach=0.1,
            usage={},
            call_confidence=0.8,
        )

    assert await evaluate_page_state_with_provider(
        goal="continue",
        action="click",
        page_state={},
        candidates=[],
        deterministic_loaded=True,
        deterministic_attached=True,
        provider=judge,
    ) == PageStateDecision(True, "state", None, 1)


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        ("page_loaded_and_stable", PageStateDecision(False, "observe", "page_load_failed")),
        ("goal_element_present", PageStateDecision(False, "state", "goal_element_absent")),
        ("action_is_the_next_step", PageStateDecision(False, "state", "action_not_next_step")),
        ("action_succeeded", PageStateDecision(True, "state", None)),
        ("dead_end", PageStateDecision(False, "stop", "dead_end")),
        ("needs_different_approach", PageStateDecision(False, "reroute", "different_approach")),
    ],
)
def test_provider_failure_maps_each_gate_to_its_safe_direction(
    gate: str, expected: PageStateDecision
) -> None:
    assert conservative_provider_error_decision(
        gate, jev.JevRouterError("provider failure")
    ) == expected
