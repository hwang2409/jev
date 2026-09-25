from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from zeta.tools.browser.adapter import (
    ActionObservation,
    BrowserTimeoutError,
    ElementRef,
    FakeBrowserAdapter,
    NavigationBlockedError,
    PageObservation,
)
from zeta.tools.browser.session import BrowserBudgetExhaustedError, BrowserSession


async def _allow_navigation(_destination: str, _current: str | None) -> None:
    return None


class _SlowAdapter(FakeBrowserAdapter):
    def __init__(self) -> None:
        super().__init__(
            [PageObservation(1, 1, "https://example.test", "", "", (), True, True)]
        )
        self.launches = 0

    async def launch(self) -> None:
        self.launches += 1
        await asyncio.sleep(0)


class _OutlivingRouteAdapter(FakeBrowserAdapter):
    def __init__(self) -> None:
        super().__init__(
            [PageObservation(1, 1, "https://example.test", "", "", (), True, True)]
        )
        self.route_started = asyncio.Event()
        self.release_route = asyncio.Event()
        self.route_task: asyncio.Task[None] | None = None
        self.late_continues = 0

    async def click(
        self, element_ref: ElementRef, timeout_ms: int
    ) -> ActionObservation:
        del element_ref, timeout_ms
        self.route_task = asyncio.create_task(self._late_route())
        await self.route_started.wait()
        raise BrowserTimeoutError("click")

    async def navigate(self, url: str, timeout_ms: int) -> PageObservation:
        del url, timeout_ms
        self.route_task = asyncio.create_task(self._late_route())
        await self.route_started.wait()
        raise BrowserTimeoutError("navigate")

    async def _late_route(self) -> None:
        self.route_started.set()
        try:
            await self._intercept_navigation("https://other.test/late")
        except asyncio.CancelledError:
            await self.release_route.wait()
            try:
                await self._intercept_navigation("https://other.test/late")
            except NavigationBlockedError:
                return
            self.late_continues += 1


class _FailingAdapter:
    def __init__(self) -> None:
        self.closed = 0

    def install_navigation_guard(self, _guard: object) -> None:
        return None

    async def launch(self) -> None:
        raise RuntimeError("launch failed")

    async def close(self) -> None:
        self.closed += 1


class _CancelledLaunchAdapter:
    def __init__(self) -> None:
        self.closed = 0

    def install_navigation_guard(self, _guard: object) -> None:
        return None

    async def launch(self) -> None:
        raise asyncio.CancelledError

    async def close(self) -> None:
        self.closed += 1


class _LaunchNavigationAdapter(_SlowAdapter):
    async def launch(self) -> None:
        await super().launch()
        await self.navigate("https://example.test/launch", 100)


@pytest.mark.asyncio
async def test_browser_session_single_flights_first_adapter_launch() -> None:
    adapter = _SlowAdapter()
    session = BrowserSession(lambda: adapter)

    first, second = await asyncio.gather(session.adapter(), session.adapter())

    assert first is adapter
    assert second is adapter
    assert adapter.launches == 1


@pytest.mark.asyncio
async def test_browser_session_closes_adapter_when_launch_fails() -> None:
    adapter = _FailingAdapter()
    session = BrowserSession(lambda: adapter)

    with pytest.raises(RuntimeError, match="launch failed"):
        await session.adapter()

    assert adapter.closed == 1
    assert session.adapter_instance is None


@pytest.mark.asyncio
async def test_browser_session_closes_adapter_when_launch_is_cancelled() -> None:
    adapter = _CancelledLaunchAdapter()
    session = BrowserSession(lambda: adapter)  # type: ignore[arg-type]

    with pytest.raises(asyncio.CancelledError):
        await session.adapter()

    assert adapter.closed == 1
    assert session.adapter_instance is None


@pytest.mark.asyncio
async def test_browser_session_installs_guard_before_launch_navigation() -> None:
    adapter = _LaunchNavigationAdapter()
    session = BrowserSession(lambda: adapter)

    with pytest.raises(NavigationBlockedError):
        await session.adapter()

    assert session.adapter_instance is None


@pytest.mark.asyncio
async def test_browser_session_charges_each_jev_call_and_usage_once() -> None:
    session = BrowserSession(
        lambda: _SlowAdapter(),
        page_jev_call_budget=1,
        page_jev_token_budget=20,
    )

    async def judge() -> SimpleNamespace:
        return SimpleNamespace(usage={"input_tokens": 3, "output_tokens": 2})

    await session.call_jev(judge)

    assert session.budget.page_jev_calls == 1
    assert session.budget.page_jev_tokens == 5
    with pytest.raises(BrowserBudgetExhaustedError):
        await session.call_jev(judge)


@pytest.mark.asyncio
async def test_browser_session_caps_tokens_and_actions_without_negative_counts() -> (
    None
):
    session = BrowserSession(
        lambda: _SlowAdapter(),
        page_jev_token_budget=5,
        task_action_budget=1,
    )

    async def judge() -> SimpleNamespace:
        return SimpleNamespace(usage={"input_tokens": 4, "output_tokens": 4})

    with pytest.raises(BrowserBudgetExhaustedError):
        await session.call_jev(judge)
    assert session.budget.page_jev_tokens == 5
    assert session.budget.page_jev_calls == 1

    session = BrowserSession(lambda: _SlowAdapter(), task_action_budget=1)
    session.consume_action()
    with pytest.raises(BrowserBudgetExhaustedError):
        session.consume_action()
    assert session.budget.task_actions == 1


def test_browser_session_uses_monotonic_wall_clock_budget() -> None:
    now = [0.0]
    session = BrowserSession(
        lambda: _SlowAdapter(),
        task_wall_clock_seconds=10,
        clock=lambda: now[0],
    )
    now[0] = 10.0
    session.ensure_available()
    now[0] = 20.0

    with pytest.raises(BrowserBudgetExhaustedError):
        session.ensure_available()


def test_browser_session_starts_task_clock_on_first_use() -> None:
    now = [0.0]
    session = BrowserSession(
        lambda: _SlowAdapter(),
        task_wall_clock_seconds=10,
        clock=lambda: now[0],
    )
    now[0] = 10.0

    session.ensure_available()
    now[0] = 19.0
    session.ensure_available()


def test_browser_threshold_version_tracks_resolved_settings() -> None:
    default = BrowserSession(lambda: _SlowAdapter())
    changed = BrowserSession(
        lambda: _SlowAdapter(),
        search_relevance_threshold=0.75,
    )

    assert default.threshold_version == "browser-thresholds-v1"
    assert changed.threshold_version != default.threshold_version


def test_browser_session_resets_task_budget_at_turn_boundary() -> None:
    session = BrowserSession(lambda: _SlowAdapter(), task_action_budget=1)
    session.consume_action()

    session.reset_turn_state()

    session.consume_action()
    assert session.budget.task_actions == 1
    assert session.budget.started_at is not None


@pytest.mark.asyncio
async def test_browser_session_resets_page_budget_after_navigation() -> None:
    adapter = _SlowAdapter()
    session = BrowserSession(lambda: adapter, page_jev_call_budget=1)

    async def judge() -> SimpleNamespace:
        return SimpleNamespace(usage={"input_tokens": 1, "output_tokens": 1})

    await session.call_jev(judge)
    assert session.budget.page_jev_calls == 1

    await session.navigate(
        "https://example.test/next", navigation_interceptor=_allow_navigation
    )

    assert session.budget.page_jev_calls == 0
    assert session.budget.page_jev_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["click", "submit"])
async def test_browser_session_resets_page_budget_after_action_navigation(
    action: str,
) -> None:
    first = PageObservation(1, 1, "https://example.test/one", "", "", (), True, True)
    second = PageObservation(2, 2, "https://example.test/two", "", "", (), True, True)
    adapter = FakeBrowserAdapter([first, second])
    session = BrowserSession(
        lambda: adapter, page_jev_call_budget=8, page_jev_token_budget=20
    )
    await session.observe(navigation_interceptor=_allow_navigation)
    session.budget.page_jev_calls = 3
    session.budget.page_jev_tokens = 7

    element = ElementRef(1, "e1", "button", action, "", "", None, None, False, True)
    await session.action(
        action, element, navigation_interceptor=_allow_navigation
    )

    assert session.budget.page_jev_calls == 0
    assert session.budget.page_jev_tokens == 0


@pytest.mark.asyncio
async def test_browser_session_keeps_page_budget_for_non_navigating_click() -> None:
    first = PageObservation(1, 1, "https://example.test/one", "", "", (), True, True)
    second = PageObservation(2, 2, first.url, "", "", (), True, True)
    adapter = FakeBrowserAdapter([first, second])
    session = BrowserSession(
        lambda: adapter, page_jev_call_budget=8, page_jev_token_budget=20
    )
    await session.observe(navigation_interceptor=_allow_navigation)
    session.budget.page_jev_calls = 3
    session.budget.page_jev_tokens = 7

    element = ElementRef(1, "e1", "button", "click", "", "", None, None, False, True)
    await session.action(
        "click", element, navigation_interceptor=_allow_navigation
    )

    assert session.budget.page_jev_calls == 3
    assert session.budget.page_jev_tokens == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["action", "navigate"])
async def test_slow_navigation_decision_cannot_continue_after_timeout(
    operation: str,
) -> None:
    adapter = _OutlivingRouteAdapter()
    session = BrowserSession(lambda: adapter)
    element = ElementRef(1, "e1", "button", "click", "", "", None, None, False, True)
    decision_started = asyncio.Event()
    release_decision = asyncio.Event()

    async def slow_decision(_destination: str, _current: str | None) -> None:
        decision_started.set()
        await release_decision.wait()

    if operation == "action":
        with pytest.raises(BrowserTimeoutError):
            await session.action("click", element, navigation_interceptor=slow_decision)
    else:
        with pytest.raises(BrowserTimeoutError):
            await session.navigate(
                "https://example.test/next",
                navigation_interceptor=slow_decision,
            )

    await asyncio.wait_for(decision_started.wait(), timeout=1)
    assert adapter.route_task is not None
    release_decision.set()
    adapter.release_route.set()
    await adapter.route_task

    assert adapter.late_continues == 0


@pytest.mark.asyncio
async def test_last_allowed_action_returns_bounded_observation_after_budget_expiry() -> (
    None
):
    now = [0.0]

    class _ExpiringAdapter(_SlowAdapter):
        async def click(
            self, element_ref: ElementRef, timeout_ms: int
        ) -> ActionObservation:
            result = await super().click(element_ref, timeout_ms)
            now[0] = 10.0
            return result

    session = BrowserSession(
        lambda: _ExpiringAdapter(),
        task_action_budget=1,
        task_wall_clock_seconds=10,
        clock=lambda: now[0],
    )
    element = ElementRef(
        1, "e1", "button", "click", "Continue", "Continue", None, "main", False, True
    )

    _action, state = await session.action(
        "click", element, navigation_interceptor=_allow_navigation
    )

    assert state.observation.url == "https://example.test"
    with pytest.raises(BrowserBudgetExhaustedError):
        session.ensure_available()
