from __future__ import annotations

import asyncio

import pytest

from zeta.tools.browser.adapter import FakeBrowserAdapter, PageObservation
from zeta.tools.browser.session import BrowserSession


class _SlowAdapter(FakeBrowserAdapter):
    def __init__(self) -> None:
        super().__init__([PageObservation(1, 1, "https://example.test", "", "", (), True, True)])
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
