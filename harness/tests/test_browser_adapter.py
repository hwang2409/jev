from __future__ import annotations

import pytest

from zeta.tools.browser.adapter import (
    ActionObservation,
    BrowserTimeoutError,
    ElementRef,
    ElementUnavailableError,
    ExtractedData,
    FakeBrowserAdapter,
    NavigationRaceError,
    PageObservation,
    SnapshotLimits,
)


def test_browser_value_types_construct_with_plain_values() -> None:
    limits = SnapshotLimits()
    element = ElementRef(
        snapshot_id=4,
        element_id="e17",
        role="button",
        affordance="click",
        text="Continue",
        name="continue",
        value_hint=None,
        landmark="main",
        disabled=False,
        visible=True,
    )
    page = PageObservation(
        4,
        9,
        "https://example.test",
        "Checkout",
        "body",
        (element,),
        True,
        True,
    )
    action = ActionObservation(4, 10, page.url, True, True, True)
    extracted = ExtractedData("body", False, 4)

    assert limits.catalog_bytes > 0
    assert page.elements == (element,)
    assert action.changed is True
    assert extracted.full_size == 4
    with pytest.raises(AttributeError):
        element.text = "changed"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_fake_records_actions_and_can_simulate_failures() -> None:
    first = PageObservation(1, 1, "https://example.test", "One", "", (), True, True)
    second = PageObservation(2, 2, first.url, "Two", "", (), True, True)
    element = ElementRef(1, "e1", "button", "click", "Next", "next", None, "main", False, True)
    adapter = FakeBrowserAdapter([first, second])

    observed = await adapter.observe(SnapshotLimits())
    assert observed.snapshot_id == 1
    adapter.detach("e1")
    with pytest.raises(ElementUnavailableError):
        await adapter.click(element, 100)
    adapter.race_next("click")
    with pytest.raises(NavigationRaceError):
        await adapter.click(element, 100)
    adapter.timeout_next("click")
    with pytest.raises(BrowserTimeoutError):
        await adapter.click(element, 100)
    await adapter.close()
    await adapter.close()
    assert adapter.clicks == []


@pytest.mark.asyncio
async def test_fake_returns_next_observation_and_records_values() -> None:
    first = PageObservation(1, 1, "https://example.test", "One", "", (), True, True)
    second = PageObservation(2, 2, "https://example.test/next", "Two", "", (), True, True)
    element = ElementRef(1, "e1", "textbox", "type", "Name", "name", None, "main", False, True)
    select = ElementRef(1, "e2", "combobox", "select", "Plan", "plan", "basic", "main", False, True)
    adapter = FakeBrowserAdapter([first, second])

    action = await adapter.type_text(element, "Ada", False, 100)
    selected = await adapter.select(select, "pro", 100)
    extracted = await adapter.extract(element, ["href"], 100)

    assert action.snapshot_id == 2
    assert action.url == second.url
    assert adapter.typed == [(element, "Ada", False)]
    assert adapter.selected == [(select, "pro")]
    assert adapter.extractions == [(element, ["href"], 100)]
    assert selected.changed is True
    assert extracted == ExtractedData({"href": None}, False, 0)


@pytest.mark.asyncio
async def test_fake_click_records_ref_and_advances_snapshot() -> None:
    first = PageObservation(1, 1, "https://example.test", "One", "", (), True, True)
    second = PageObservation(2, 2, "https://example.test/next", "Two", "", (), True, True)
    element = ElementRef(1, "e1", "button", "click", "Next", "next", None, "main", False, True)
    adapter = FakeBrowserAdapter([first, second])

    action = await adapter.click(element, 100)

    assert action.snapshot_id == 2
    assert action.url == second.url
    assert adapter.clicks == [element]


@pytest.mark.asyncio
async def test_fake_navigation_records_url_and_launch_is_idempotent() -> None:
    observation = PageObservation(1, 1, "https://example.test", "One", "", (), True, True)
    adapter = FakeBrowserAdapter([observation])

    await adapter.launch()
    await adapter.launch()
    result = await adapter.navigate("https://example.test/next", 100)

    assert result.url == "https://example.test/next"
    assert adapter.navigations == ["https://example.test/next"]


@pytest.mark.asyncio
async def test_fake_navigation_advances_scripted_snapshots() -> None:
    first = PageObservation(1, 1, "https://example.test", "One", "", (), True, True)
    second = PageObservation(2, 2, "https://example.test/two", "Two", "", (), True, True)
    third = PageObservation(3, 3, "https://example.test/three", "Three", "", (), True, True)
    adapter = FakeBrowserAdapter([first, second, third])

    first_result = await adapter.navigate(second.url, 100)
    second_result = await adapter.navigate(third.url, 100)

    assert second_result.snapshot_id > first_result.snapshot_id
