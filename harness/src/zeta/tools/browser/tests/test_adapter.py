from __future__ import annotations

from pathlib import Path

import pytest

from zeta.skills import SkillCatalog
from zeta.tools.browser import register
from zeta.tools.browser.adapter import (
    ActionObservation,
    BrowserError,
    BrowserTimeoutError,
    ElementRef,
    ElementUnavailableError,
    ExtractedData,
    FakeBrowserAdapter,
    NavigationRaceError,
    PageObservation,
    PlaywrightBrowserAdapter,
    SearchResultCandidate,
    SnapshotLimits,
    make_browser_adapter_factory,
)
from zeta.tools.registry import ToolRegistry


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
    assert extracted == ExtractedData({"href": None}, False, 4, 14)


@pytest.mark.asyncio
async def test_fake_bounds_attribute_names_and_values_by_total_bytes() -> None:
    observation = PageObservation(1, 1, "https://example.test", "One", "", (), True, True)
    adapter = FakeBrowserAdapter([observation])
    attributes = [f"attribute_{index}_{'x' * 200}" for index in range(100)]
    element = ElementRef(1, "e1", "article", "extract", "v" * 1000, "", None, None, False, True)

    extracted = await adapter.extract(element, ["text", *attributes], 800)

    assert extracted.truncated is True
    assert isinstance(extracted.value, dict)
    retained_size = sum(
        len(name.encode("utf-8")) + len((value or "").encode("utf-8"))
        for name, value in extracted.value.items()
    )
    assert retained_size <= 800
    assert extracted.full_size == 4 + 1000 + sum(
        len(name.encode("utf-8")) for name in attributes
    )


@pytest.mark.asyncio
async def test_fake_extracts_typed_search_results() -> None:
    observation = PageObservation(1, 1, "https://example.test", "One", "", (), True, True)
    results = (
        SearchResultCandidate("a", "first", "snippet", "https://one.example", "one", 1),
    )
    adapter = FakeBrowserAdapter([observation], search_results=results)

    extracted = await adapter.extract_search_results(None, 1_000)

    assert extracted.results == results
    assert extracted.truncated is False
    assert adapter.search_extractions == [(None, 1_000)]


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


@pytest.mark.asyncio
async def test_playwright_adapter_uses_local_fixture_for_the_browser_contract() -> None:
    pytest.importorskip("playwright")
    fixture = Path(__file__).parent / "fixtures" / "adapter.html"
    adapter = PlaywrightBrowserAdapter(headless=True, limits=SnapshotLimits())

    await adapter.launch()
    try:
        observation = await adapter.navigate(fixture.as_uri(), 2_000)
        assert observation.url.startswith("file://")
        assert observation.title == "adapter fixture"
        assert all(isinstance(element, ElementRef) for element in observation.elements)
        assert "locator" not in repr(observation)

        button = next(element for element in observation.elements if element.role == "button")
        action = await adapter.click(button, 1_000)
        assert action.changed is True

        refreshed = await adapter.observe(SnapshotLimits())
        query = next(element for element in refreshed.elements if element.name == "query")
        choice = next(element for element in refreshed.elements if element.name == "choice")
        await adapter.type_text(query, "updated", True, 1_000)
        refreshed = await adapter.observe(SnapshotLimits())
        choice = next(element for element in refreshed.elements if element.name == "choice")
        await adapter.select(choice, "two", 1_000)

        extracted = await adapter.extract(None, [], 2_000)
        assert isinstance(extracted.value, str)
        assert "updated" in extracted.value or "local browser fixture" in extracted.value
        search = await adapter.extract_search_results(None, 2_000)
        assert search.results is not None
        assert search.results[0].result_id == "result-1"
    finally:
        await adapter.close()
        await adapter.close()


@pytest.mark.asyncio
async def test_playwright_adapter_rejects_stale_and_detached_element_refs() -> None:
    pytest.importorskip("playwright")
    fixture = Path(__file__).parent / "fixtures" / "adapter.html"
    adapter = PlaywrightBrowserAdapter(headless=True, limits=SnapshotLimits())

    await adapter.launch()
    try:
        observation = await adapter.navigate(fixture.as_uri(), 2_000)
        button = next(element for element in observation.elements if element.role == "button")
        await adapter.observe(SnapshotLimits())
        with pytest.raises(ElementUnavailableError):
            await adapter.click(button, 1_000)

        current = await adapter.observe(SnapshotLimits())
        current_button = next(element for element in current.elements if element.role == "button")
        await adapter._page.evaluate("document.querySelector('#continue').remove()")
        with pytest.raises(ElementUnavailableError):
            await adapter.click(current_button, 1_000)
    finally:
        await adapter.close()


def test_browser_adapter_selection_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZETA_BROWSER_ADAPTER", raising=False)
    disabled = make_browser_adapter_factory()
    with pytest.raises(BrowserError, match="ZETA_BROWSER_ADAPTER=playwright"):
        disabled()

    selected = make_browser_adapter_factory(mode="playwright")
    adapter = selected()
    assert isinstance(adapter, PlaywrightBrowserAdapter)


def test_browser_register_wires_the_configured_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ZETA_BROWSER_ADAPTER", "playwright")
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )

    register(registry)

    assert isinstance(registry.browser_adapter_factory(), PlaywrightBrowserAdapter)
