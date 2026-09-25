from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zeta.core.safety import BrowserRiskEvidence, SafetyOutcome, SafetyTier
from zeta.protocol import jev
from zeta.protocol.types import ToolCall
from zeta.tools.browser import PageStateDecision, _navigation_interceptor
from zeta.tools.browser.adapter import (
    FakeBrowserAdapter,
    NavigationBlockedError,
    PageObservation,
    SnapshotLimits,
)
from zeta.tools.browser.tests.test_tools import (
    _choice,
    _observation,
    _registry,
    _structured,
)


@pytest.mark.asyncio
async def test_navigation_guard_classifies_every_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    tier = SafetyTier(cwd=tmp_path, headless=True)
    registry = _registry(tmp_path, adapter, tier)
    seen: list[BrowserRiskEvidence] = []

    async def classify(evidence: BrowserRiskEvidence) -> SafetyOutcome:
        seen.append(evidence)
        return SafetyOutcome("deny", "layer0", reason="external_origin")

    monkeypatch.setattr(tier, "evaluate_browser_action", classify)
    interceptor = _navigation_interceptor(
        registry,
        {},
        abort_signal=None,
        execution_context=None,
        fallback_url="https://example.test/",
    )

    with pytest.raises(NavigationBlockedError):
        await interceptor("https://other.test/submit", "https://example.test/")

    assert seen and seen[0].target_url == "https://other.test/submit"


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [False, True], ids=["interactive", "headless"])
@pytest.mark.parametrize("navigation_kind", ["redirect", "javascript"])
async def test_actual_top_level_navigation_uses_shared_safety_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
    navigation_kind: str,
) -> None:
    first = _observation()
    second = PageObservation(
        2,
        2,
        "https://other.test/redirected",
        "Other",
        "Other page",
        (),
        True,
        True,
    )
    adapter = FakeBrowserAdapter([first, second])
    tier = SafetyTier(cwd=tmp_path, headless=headless)
    registry = _registry(tmp_path, adapter, tier)
    await registry.execute(ToolCall("state", "browser_state", {}))
    if navigation_kind == "javascript":
        adapter.queue_navigation("https://other.test/javascript")

    seen: list[BrowserRiskEvidence] = []

    async def classify(evidence: BrowserRiskEvidence) -> SafetyOutcome:
        seen.append(evidence)
        return SafetyOutcome(
            "deny" if headless else "ask",
            "layer0",
            reason="external_origin",
        )

    async def choose(
        *_args: object, **_kwargs: object
    ) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",))

    async def page_gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(True, None, None)

    monkeypatch.setattr(tier, "evaluate_browser_action", classify)
    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr(
        "zeta.tools.browser.evaluate_page_state_with_provider", page_gate
    )

    result = await registry.execute(
        ToolCall(
            "navigate by page action",
            "browser_click",
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "role": "button",
                "affordance": "click",
            },
        )
    )

    assert result["isError"] is True
    structured = _structured(result)
    assert structured["error"]["kind"] == "safety_denied"
    assert "external_origin" in structured["error"]["message"]
    assert seen and seen[0].target_url == (
        "https://other.test/javascript"
        if navigation_kind == "javascript"
        else "https://other.test/redirected"
    )
    assert (await adapter.observe(SnapshotLimits())).url == first.url


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [False, True], ids=["interactive", "headless"])
async def test_delayed_navigation_after_action_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    tier = SafetyTier(cwd=tmp_path, headless=headless)
    registry = _registry(tmp_path, adapter, tier)
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(
        *_args: object, **_kwargs: object
    ) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",))

    async def page_gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(True, None, None)

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr(
        "zeta.tools.browser.evaluate_page_state_with_provider", page_gate
    )

    result = await registry.execute(
        ToolCall(
            "click",
            "browser_click",
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "role": "button",
                "affordance": "click",
            },
        )
    )
    assert result["isError"] is False

    delayed = asyncio.create_task(
        adapter.trigger_navigation("https://other.test/timer")
    )
    await asyncio.sleep(0)
    await delayed

    observation = await registry.execute(ToolCall("state", "browser_state", {}))
    assert observation["isError"] is True
    assert _structured(observation)["error"]["kind"] == "safety_denied"
    assert adapter.navigations == []
