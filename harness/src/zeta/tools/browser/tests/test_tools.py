from __future__ import annotations

from pathlib import Path

import pytest

from zeta.protocol.types import ToolCall
from zeta.providers import jev
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.browser import PageStateDecision, register
from zeta.tools.browser.adapter import ElementRef, FakeBrowserAdapter, PageObservation


def _observation(snapshot_id: int = 1) -> PageObservation:
    return PageObservation(
        snapshot_id,
        snapshot_id,
        "https://example.test/",
        "Example",
        "Page text",
        (
            ElementRef(
                snapshot_id,
                "e1",
                "button",
                "click",
                "Continue",
                "Continue",
                None,
                "main",
                False,
                True,
            ),
        ),
        True,
        True,
    )


def _registry(tmp_path: Path, adapter: FakeBrowserAdapter) -> ToolRegistry:
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.browser_adapter_factory = lambda: adapter
    register(registry)
    return registry


def _structured(result: dict[str, object]) -> dict[str, object]:
    structured = result["structuredContent"]
    assert isinstance(structured, dict)
    return structured


def _choice(element_id: str | None, confidence: float, candidates: tuple[str, ...]) -> jev.BrowserElementChoiceResult:
    return jev.BrowserElementChoiceResult(
        element_id,
        "click" if element_id is not None else None,
        candidates,
        {candidate: 1 / len(candidates) for candidate in candidates},
        confidence,
        0.9,
        0.9,
        0.9,
        {},
        confidence,
    )


@pytest.mark.asyncio
async def test_browser_click_uses_jev_choice_and_pre_post_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    registry.browser_goal = "continue"
    await registry.execute(ToolCall("state", "browser_state", {}))
    gates: list[dict[str, object]] = []

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",))

    async def gate(**kwargs: object) -> PageStateDecision:
        gates.append(kwargs)
        return PageStateDecision(True, None, None)

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr("zeta.tools.browser.evaluate_page_state_with_provider", gate)

    result = await registry.execute(
        ToolCall(
            "click",
            "browser_click",
            {"snapshot_id": 1, "element_id": "e1", "role": "button", "affordance": "click"},
        )
    )

    assert result["isError"] is False
    assert len(gates) == 2
    assert [element.element_id for element in adapter.clicks] == ["e1"]


@pytest.mark.asyncio
async def test_browser_click_returns_top_three_without_acting_on_low_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _observation()
    elements = tuple(
        ElementRef(
            1,
            f"e{index}",
            "button",
            "click",
            f"Choice {index}",
            f"Choice {index}",
            None,
            "main",
            False,
            True,
        )
        for index in range(1, 4)
    )
    adapter = FakeBrowserAdapter(
        [
            PageObservation(
                observation.snapshot_id,
                observation.generation,
                observation.url,
                observation.title,
                observation.text,
                elements,
                observation.loaded,
                observation.stable,
            )
        ]
    )
    registry = _registry(tmp_path, adapter)
    registry.browser_goal = "choose a choice"
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice(None, 0.7, ("e1", "e2", "e3"))

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    result = await registry.execute(
        ToolCall(
            "click",
            "browser_click",
            {"snapshot_id": 1, "element_id": "e1", "role": "button", "affordance": "click"},
        )
    )

    assert result["isError"] is False
    assert _structured(result)["candidate_ids"] == ["e1", "e2", "e3"]
    assert adapter.clicks == []


@pytest.mark.asyncio
async def test_browser_click_does_not_act_when_page_state_gate_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    registry.browser_goal = "continue"
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",))

    async def gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(False, "state", "action_not_next_step")

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr("zeta.tools.browser.evaluate_page_state_with_provider", gate)
    result = await registry.execute(
        ToolCall(
            "click",
            "browser_click",
            {"snapshot_id": 1, "element_id": "e1", "role": "button", "affordance": "click"},
        )
    )

    assert result["isError"] is True
    assert _structured(result)["error"]["kind"] == "action_not_next_step"
    assert adapter.clicks == []


@pytest.mark.asyncio
async def test_browser_registers_stable_schemas_and_starts_lazily(tmp_path: Path) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)

    assert {
        "browser_navigate",
        "browser_state",
        "browser_click",
        "browser_type",
        "browser_select",
        "browser_extract",
        "browser_submit",
    } <= registry.registered_names
    assert adapter.navigations == []
    result = await registry.execute(ToolCall("state", "browser_state", {}))

    assert result["isError"] is False
    assert _structured(result)["snapshot_id"] == 1
    assert adapter.navigations == []


@pytest.mark.asyncio
async def test_browser_rejects_stale_and_mismatched_element_identity(
    tmp_path: Path,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    await registry.execute(ToolCall("state", "browser_state", {}))

    stale = await registry.execute(
        ToolCall(
            "stale",
            "browser_click",
            {
                "snapshot_id": 999,
                "element_id": "e1",
                "role": "button",
                "affordance": "click",
            },
        )
    )
    mismatch = await registry.execute(
        ToolCall(
            "mismatch",
            "browser_click",
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "role": "link",
                "affordance": "click",
            },
        )
    )

    assert _structured(stale)["error"]["kind"] == "stale_snapshot"
    assert _structured(mismatch)["error"]["kind"] == "element_unavailable"
    assert adapter.clicks == []


@pytest.mark.asyncio
async def test_browser_close_is_idempotent_and_closes_adapter(tmp_path: Path) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    await registry.execute(ToolCall("state", "browser_state", {}))

    await registry.close()
    await registry.close()


@pytest.mark.asyncio
async def test_browser_extract_clamps_limit_and_keeps_truncation_successful(
    tmp_path: Path,
) -> None:
    long_text = "x" * 10_000
    observation = PageObservation(
        1,
        1,
        "https://example.test/",
        "Example",
        long_text,
        (
            ElementRef(
                1,
                "e1",
                "article",
                "extract",
                long_text,
                long_text,
                None,
                "main",
                False,
                True,
            ),
        ),
        True,
        True,
    )
    adapter = FakeBrowserAdapter([observation])
    registry = _registry(tmp_path, adapter)
    await registry.execute(ToolCall("state", "browser_state", {}))

    result = await registry.execute(
        ToolCall(
            "extract",
            "browser_extract",
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "attributes": [],
                "limit": 100_000,
            },
        )
    )

    assert result["isError"] is False
    assert _structured(result)["truncated"] is True
    assert adapter.extractions[-1][2] == 8_000
