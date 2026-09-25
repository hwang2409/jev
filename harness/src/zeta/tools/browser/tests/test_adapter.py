from __future__ import annotations

from pathlib import Path

import pytest

from zeta.skills import SkillCatalog
from zeta.tools.browser import register
from zeta.tools.browser.adapter import (
    ActionObservation,
    BrowserError,
    BrowserExecutableNotFoundError,
    BrowserTimeoutError,
    ElementRef,
    ElementUnavailableError,
    ExtractedData,
    FakeBrowserAdapter,
    NavigationBlockedError,
    NavigationRaceError,
    PageObservation,
    PlaywrightBrowserAdapter,
    SearchResultCandidate,
    SnapshotLimits,
    make_browser_adapter_factory,
)
from zeta.tools.registry import ToolRegistry


async def _allow_navigation(_destination: str, _current: str | None) -> None:
    return None


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
    element = ElementRef(
        1, "e1", "button", "click", "Next", "next", None, "main", False, True
    )
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
    second = PageObservation(
        2, 2, "https://example.test/next", "Two", "", (), True, True
    )
    element = ElementRef(
        1, "e1", "textbox", "type", "Name", "name", None, "main", False, True
    )
    select = ElementRef(
        1, "e2", "combobox", "select", "Plan", "plan", "basic", "main", False, True
    )
    adapter = FakeBrowserAdapter([first, second])
    adapter.install_navigation_guard(_allow_navigation)

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
    observation = PageObservation(
        1, 1, "https://example.test", "One", "", (), True, True
    )
    adapter = FakeBrowserAdapter([observation])
    attributes = [f"attribute_{index}_{'x' * 200}" for index in range(100)]
    element = ElementRef(
        1, "e1", "article", "extract", "v" * 1000, "", None, None, False, True
    )

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
    observation = PageObservation(
        1, 1, "https://example.test", "One", "", (), True, True
    )
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
    second = PageObservation(
        2, 2, "https://example.test/next", "Two", "", (), True, True
    )
    element = ElementRef(
        1, "e1", "button", "click", "Next", "next", None, "main", False, True
    )
    adapter = FakeBrowserAdapter([first, second])
    adapter.install_navigation_guard(_allow_navigation)

    action = await adapter.click(element, 100)

    assert action.snapshot_id == 2
    assert action.url == second.url
    assert adapter.clicks == [element]


@pytest.mark.asyncio
async def test_fake_navigation_records_url_and_launch_is_idempotent() -> None:
    observation = PageObservation(
        1, 1, "https://example.test", "One", "", (), True, True
    )
    adapter = FakeBrowserAdapter([observation])

    await adapter.launch()
    await adapter.launch()
    adapter.install_navigation_guard(_allow_navigation)
    result = await adapter.navigate("https://example.test/next", 100)

    assert result.url == "https://example.test/next"
    assert adapter.navigations == ["https://example.test/next"]


@pytest.mark.asyncio
async def test_fake_navigation_advances_scripted_snapshots() -> None:
    first = PageObservation(1, 1, "https://example.test", "One", "", (), True, True)
    second = PageObservation(
        2, 2, "https://example.test/two", "Two", "", (), True, True
    )
    third = PageObservation(
        3, 3, "https://example.test/three", "Three", "", (), True, True
    )
    adapter = FakeBrowserAdapter([first, second, third])
    adapter.install_navigation_guard(_allow_navigation)

    first_result = await adapter.navigate(second.url, 100)
    second_result = await adapter.navigate(third.url, 100)

    assert second_result.snapshot_id > first_result.snapshot_id


@pytest.mark.asyncio
async def test_element_from_raw_applies_the_final_utf8_byte_cap() -> None:
    class RawPage:
        async def evaluate(self, _script: str, _arguments: object) -> dict[str, object]:
            return {
                "url": "https://example.test",
                "title": "",
                "text": "",
                "loaded": True,
                "stable": True,
                "elements": [
                    {
                        "element_id": "e1",
                        "role": "button",
                        "affordance": "click",
                        "text": "abcdefgh",
                        "name": "abcdefgh",
                        "value_hint": "abcdefgh",
                        "landmark": "abcdefgh",
                    }
                ],
            }

        def locator(self, _selector: str) -> object:
            return object()

    adapter = PlaywrightBrowserAdapter(
        headless=True,
        limits=SnapshotLimits(element_text_bytes=4),
    )
    adapter._page = RawPage()
    observation = await adapter.observe(SnapshotLimits(element_text_bytes=4))
    element = observation.elements[0]

    assert element.text == "abcd"
    assert element.name == "abcd"
    assert element.value_hint == "abcd"
    assert element.landmark == "abcd"
    assert all(
        len(value.encode("utf-8")) <= 4
        for value in (element.text, element.name, element.value_hint, element.landmark)
        if value is not None
    )


@pytest.fixture(params=("fake", "playwright"), ids=("fake", "playwright"))
async def contract_adapter(
    request: pytest.FixtureRequest,
) -> FakeBrowserAdapter | PlaywrightBrowserAdapter:
    if request.param == "fake":
        element_sets = (
            (
                ElementRef(
                    snapshot_id,
                    "button",
                    "button",
                    "click",
                    "continue",
                    "continue",
                    None,
                    "main",
                    False,
                    True,
                    generation=snapshot_id,
                ),
                ElementRef(
                    snapshot_id,
                    "query",
                    "textbox",
                    "type",
                    "query",
                    "query",
                    "initial",
                    "main",
                    False,
                    True,
                    generation=snapshot_id,
                ),
                ElementRef(
                    snapshot_id,
                    "choice",
                    "combobox",
                    "select",
                    "choice",
                    "choice",
                    "one",
                    "main",
                    False,
                    True,
                    generation=snapshot_id,
                ),
            )
            for snapshot_id in range(1, 5)
        )
        results = (
            SearchResultCandidate(
                "result-1", "first", "snippet", "https://example.test/one", "page", 1
            ),
        )
        adapter = FakeBrowserAdapter(
            [
                PageObservation(
                    index,
                    index,
                    "https://example.test",
                    "Example",
                    "body",
                    elements,
                    True,
                    True,
                )
                for index, elements in enumerate(element_sets, 1)
            ],
            search_results=results,
        )
        adapter.install_navigation_guard(_allow_navigation)
        yield adapter
        return

    pytest.importorskip("playwright", reason="playwright package is absent")
    adapter = PlaywrightBrowserAdapter(headless=True, limits=SnapshotLimits())
    try:
        await adapter.launch()
    except BrowserExecutableNotFoundError as exc:
        await adapter.close()
        pytest.skip(f"playwright browser executable is absent: {exc}")
    adapter.install_navigation_guard(_allow_navigation)
    try:
        yield adapter
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_real_adapter_does_not_hide_launch_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenChromium:
        async def launch(self, *, headless: bool) -> object:
            del headless
            raise RuntimeError("browser executable path is invalid")

    class BrokenPlaywright:
        chromium = BrokenChromium()

    class BrokenManager:
        async def start(self) -> BrokenPlaywright:
            return BrokenPlaywright()

    monkeypatch.setattr(
        "zeta.tools.browser.adapter.load_playwright_page",
        lambda: type(
            "PlaywrightModule",
            (),
            {"async_playwright": staticmethod(lambda: BrokenManager())},
        )(),
    )
    adapter = PlaywrightBrowserAdapter(headless=True, limits=SnapshotLimits())

    with pytest.raises(BrowserError, match="browser launch failed") as raised:
        await adapter.launch()

    assert not isinstance(raised.value, BrowserExecutableNotFoundError)


@pytest.mark.asyncio
async def test_real_adapter_types_a_missing_browser_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingExecutableError(RuntimeError):
        pass

    MissingExecutableError.__name__ = "Error"

    class MissingChromium:
        async def launch(self, *, headless: bool) -> object:
            del headless
            raise MissingExecutableError(
                "BrowserType.launch: Executable doesn't exist at /missing/browser"
            )

    class MissingPlaywright:
        chromium = MissingChromium()

    class MissingManager:
        async def start(self) -> MissingPlaywright:
            return MissingPlaywright()

    adapter = PlaywrightBrowserAdapter(headless=True, limits=SnapshotLimits())
    monkeypatch.setattr(
        "zeta.tools.browser.adapter.load_playwright_page",
        lambda: type(
            "PlaywrightModule",
            (),
            {"async_playwright": staticmethod(lambda: MissingManager())},
        )(),
    )
    with pytest.raises(BrowserExecutableNotFoundError):
        await adapter.launch()


@pytest.mark.asyncio
async def test_adapters_fail_closed_without_a_navigation_guard() -> None:
    observation = PageObservation(
        1, 1, "https://example.test", "One", "", (), True, True
    )
    fake = FakeBrowserAdapter([observation])

    with pytest.raises(NavigationBlockedError):
        await fake.navigate("https://example.test/next", 100)

    class Request:
        def __init__(self, frame: object) -> None:
            self.url = "https://example.test/next"
            self.frame = frame

        def is_navigation_request(self) -> bool:
            return True

    class Page:
        def __init__(self) -> None:
            self.main_frame = object()

    class Route:
        def __init__(self, request: Request) -> None:
            self.request = request
            self.continued = False
            self.aborted = False

        async def continue_(self) -> None:
            self.continued = True

        async def abort(self) -> None:
            self.aborted = True

    real = PlaywrightBrowserAdapter(headless=True, limits=SnapshotLimits())
    page = Page()
    real._page = page
    route = Route(Request(page.main_frame))

    await real._handle_route(route)

    assert route.aborted is True
    assert route.continued is False
    assert isinstance(real._navigation_error, NavigationBlockedError)


@pytest.mark.asyncio
async def test_browser_adapters_share_the_action_contract(
    contract_adapter: FakeBrowserAdapter | PlaywrightBrowserAdapter,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "adapter.html"
    url = (
        fixture.as_uri()
        if isinstance(contract_adapter, PlaywrightBrowserAdapter)
        else "https://example.test"
    )
    observation = await contract_adapter.navigate(url, 2_000)
    assert observation.loaded and observation.stable
    assert all(isinstance(element, ElementRef) for element in observation.elements)
    button = next(
        element for element in observation.elements if element.affordance == "click"
    )
    action = await contract_adapter.click(button, 1_000)
    assert action.changed is True
    current = await contract_adapter.observe(SnapshotLimits())
    query = next(element for element in current.elements if element.name == "query")
    await contract_adapter.type_text(query, "updated", True, 1_000)
    current = await contract_adapter.observe(SnapshotLimits())
    choice = next(element for element in current.elements if element.name == "choice")
    await contract_adapter.select(choice, "two", 1_000)
    extracted = await contract_adapter.extract(None, [], 2_000)
    assert isinstance(extracted.value, str)
    search = await contract_adapter.extract_search_results(None, 2_000)
    assert search.results is not None


@pytest.mark.asyncio
async def test_real_adapter_rejects_a_replaced_dom_element(
    contract_adapter: FakeBrowserAdapter | PlaywrightBrowserAdapter,
) -> None:
    if isinstance(contract_adapter, FakeBrowserAdapter):
        observation = await contract_adapter.observe(SnapshotLimits())
        button = next(
            element
            for element in observation.elements
            if element.element_id == "button"
        )
        contract_adapter.detach(button.element_id)
        with pytest.raises(ElementUnavailableError):
            await contract_adapter.click(button, 1_000)
        return

    fixture = Path(__file__).parent / "fixtures" / "adapter.html"
    observation = await contract_adapter.navigate(fixture.as_uri(), 2_000)
    button = next(
        element for element in observation.elements if element.role == "button"
    )
    await contract_adapter._page.evaluate(
        "const node = document.querySelector('#continue'); node.replaceWith(node.cloneNode(true));"
    )
    with pytest.raises(ElementUnavailableError):
        await contract_adapter.click(button, 1_000)
    assert await contract_adapter._page.locator("#status").inner_text() == "ready"


def test_browser_adapter_selection_covers_all_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZETA_BROWSER_ADAPTER", raising=False)
    default = make_browser_adapter_factory()
    assert isinstance(default(), FakeBrowserAdapter)

    disabled = make_browser_adapter_factory(mode="disabled")
    with pytest.raises(BrowserError, match="adapter is disabled"):
        disabled()

    selected = make_browser_adapter_factory(mode="playwright")
    adapter = selected()
    assert isinstance(adapter, PlaywrightBrowserAdapter)


def test_browser_register_wires_the_configured_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ZETA_BROWSER_ADAPTER", "playwright")
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        browser_enabled=True,
        skill_catalog=SkillCatalog.empty(),
    )

    register(registry)

    assert isinstance(registry.browser_adapter_factory(), PlaywrightBrowserAdapter)
