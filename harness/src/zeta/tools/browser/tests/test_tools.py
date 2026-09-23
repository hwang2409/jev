from __future__ import annotations

from pathlib import Path

import pytest

from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.browser import register
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
