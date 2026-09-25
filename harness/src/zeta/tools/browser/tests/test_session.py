from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from zeta.tools.browser.adapter import FakeBrowserAdapter, PageObservation
from zeta.tools.browser.session import BrowserBudgetExhaustedError, BrowserSession


class _SlowAdapter(FakeBrowserAdapter):
    def __init__(self) -> None:
        super().__init__(
            [PageObservation(1, 1, "https://example.test", "", "", (), True, True)]
        )
        self.launches = 0

    async def launch(self) -> None:
        self.launches += 1
        await asyncio.sleep(0)


class _FailingAdapter:
    def __init__(self) -> None:
        self.closed = 0

    async def launch(self) -> None:
        raise RuntimeError("launch failed")

    async def close(self) -> None:
        self.closed += 1


class _CancelledLaunchAdapter:
    def __init__(self) -> None:
        self.closed = 0

    async def launch(self) -> None:
        raise asyncio.CancelledError

    async def close(self) -> None:
        self.closed += 1


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

    with pytest.raises(BrowserBudgetExhaustedError):
        session.ensure_available()
