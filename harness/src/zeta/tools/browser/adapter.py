"""Framework-neutral browser adapter values and deterministic fake."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from html import escape
from pathlib import PurePosixPath
from typing import Protocol
from urllib.parse import urlsplit

_logger = logging.getLogger(__name__)

NavigationInterceptor = Callable[[str, str | None], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class SnapshotLimits:
    page_text_bytes: int = 16_000
    element_text_bytes: int = 240
    catalog_bytes: int = 12_000
    extracted_bytes: int = 8_000


@dataclass(frozen=True, slots=True)
class ElementRef:
    snapshot_id: int
    element_id: str
    role: str
    affordance: str
    text: str
    name: str
    value_hint: str | None
    landmark: str | None
    disabled: bool
    visible: bool
    target_url: str | None = None
    form_action_origin: str | None = None
    download: bool = False
    durable_state_change: bool = False
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class PageObservation:
    snapshot_id: int
    generation: int
    url: str
    title: str
    text: str
    elements: tuple[ElementRef, ...]
    loaded: bool
    stable: bool


@dataclass(frozen=True, slots=True)
class ActionObservation:
    snapshot_id: int
    generation: int
    url: str
    loaded: bool
    stable: bool
    changed: bool


@dataclass(frozen=True, slots=True)
class ExtractedData:
    value: str | dict[str, str | None]
    truncated: bool
    full_size: int
    escaped_full_size: int | None = None


@dataclass(frozen=True, slots=True)
class SearchResultCandidate:
    result_id: str
    title: str
    snippet: str
    displayed_url: str
    source_section: str
    position: int


@dataclass(frozen=True, slots=True)
class SearchResultExtraction:
    """Typed search candidates, or ``None`` when the page is not a search page."""

    results: tuple[SearchResultCandidate, ...] | None
    truncated: bool
    full_size: int


class BrowserAdapter(Protocol):
    async def launch(self) -> None:
        raise NotImplementedError

    def set_navigation_interceptor(
        self, interceptor: NavigationInterceptor | None
    ) -> None:
        raise NotImplementedError

    async def navigate(self, url: str, timeout_ms: int) -> PageObservation:
        raise NotImplementedError

    async def observe(self, limits: SnapshotLimits) -> PageObservation:
        raise NotImplementedError

    async def click(
        self, element_ref: ElementRef, timeout_ms: int
    ) -> ActionObservation:
        raise NotImplementedError

    async def type_text(
        self,
        element_ref: ElementRef,
        text: str,
        replace: bool,
        timeout_ms: int,
    ) -> ActionObservation:
        raise NotImplementedError

    async def select(
        self,
        element_ref: ElementRef,
        value: str,
        timeout_ms: int,
    ) -> ActionObservation:
        raise NotImplementedError

    async def extract(
        self,
        target: ElementRef | None,
        attributes: list[str],
        limit: int,
    ) -> ExtractedData:
        raise NotImplementedError

    async def extract_search_results(
        self,
        target: ElementRef | None,
        limit: int,
    ) -> SearchResultExtraction:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class BrowserError(RuntimeError):
    """Base error translated into a stable browser tool error kind."""


class BrowserExecutableNotFoundError(BrowserError):
    """The configured Playwright browser executable is not installed."""


class ElementUnavailableError(BrowserError):
    """The requested element is detached or ambiguous."""


class BrowserTimeoutError(BrowserError):
    """The adapter exceeded its bounded timeout."""


class NavigationRaceError(BrowserError):
    """Navigation changed page generation during an action."""


class NavigationBlockedError(BrowserError):
    """The safety policy blocked a top-level browser navigation."""


def load_playwright_page() -> object:
    """Load Playwright lazily so the harness keeps its optional dependency."""

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise BrowserError(
            "the Playwright browser adapter requires the browser extra"
        ) from exc
    return async_playwright


OBSERVE_SCRIPT = """
({
  pageTextBytes,
  elementTextBytes,
}) => {
  const bounded = (value, limit) => String(value || '').slice(0, Math.max(limit * 2, 0));
  const domGeneration = () => {
    if (!Number.isInteger(window.__zetaBrowserDomGeneration)) {
      window.__zetaBrowserDomGeneration = 0;
      new MutationObserver((records) => {
        if (records.some((record) => record.attributeName !== 'data-zeta-browser-ref')) {
          window.__zetaBrowserDomGeneration += 1;
        }
      }).observe(document, {
        subtree: true,
        childList: true,
        attributes: true,
        characterData: true,
      });
    }
    return window.__zetaBrowserDomGeneration;
  };
  const visible = (element) => {
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden'
      && rect.width > 0 && rect.height > 0;
  };
  const text = (element) => bounded(element.innerText || element.textContent || '', elementTextBytes);
  const label = (element) => {
    const labelledBy = element.getAttribute('aria-labelledby');
    if (labelledBy) {
      return bounded(labelledBy.split(/\\s+/).map((id) => document.getElementById(id)?.innerText || '').join(' '), elementTextBytes);
    }
    const aria = element.getAttribute('aria-label');
    if (aria) return bounded(aria, elementTextBytes);
    if (element.labels && element.labels.length) return bounded(element.labels[0].innerText, elementTextBytes);
    return bounded(element.getAttribute('title') || element.innerText || element.textContent || element.getAttribute('name') || '', elementTextBytes);
  };
  const landmark = (element) => {
    const parent = element.closest('main,nav,form,header,footer,aside,[role="main"],[role="navigation"],[role="form"]');
    return parent ? bounded(parent.getAttribute('aria-label') || parent.tagName.toLowerCase(), elementTextBytes) : null;
  };
  const roleAndAffordance = (element) => {
    const explicit = element.getAttribute('role');
    const tag = element.tagName.toLowerCase();
    if (explicit === 'button' || tag === 'button') return ['button', element.type === 'submit' ? 'submit' : 'click'];
    if (explicit === 'link' || tag === 'a') return ['link', 'click'];
    if (explicit === 'textbox' || tag === 'textarea') return ['textbox', 'type'];
    if (explicit === 'combobox' || tag === 'select') return ['combobox', 'select'];
    if (explicit === 'checkbox' || (tag === 'input' && element.type === 'checkbox')) return ['checkbox', 'click'];
    if (explicit === 'radio' || (tag === 'input' && element.type === 'radio')) return ['radio', 'click'];
    if (tag === 'input' && ['button', 'submit', 'reset', 'hidden', 'image'].indexOf(element.type) < 0) return ['textbox', 'type'];
    if (explicit === 'tab') return ['tab', 'click'];
    if (explicit === 'heading' || /^h[1-6]$/.test(tag)) return ['heading', 'extract'];
    if (explicit === 'article' || tag === 'article') return ['article', 'extract'];
    return null;
  };
  const candidates = Array.from(document.querySelectorAll('a,button,input,select,textarea,textarea,article,h1,h2,h3,h4,h5,h6,[role]'));
  const elements = [];
  for (const element of candidates) {
    const role = roleAndAffordance(element);
    if (!role) continue;
    const [roleName, affordance] = role;
    const elementId = `e${elements.length + 1}`;
    element.setAttribute('data-zeta-browser-ref', elementId);
    const form = element.form;
    let formActionOrigin = null;
    if (form) {
      try { formActionOrigin = new URL(form.action || document.location.href, document.location.href).origin; } catch (_) { formActionOrigin = null; }
    }
    let targetUrl = null;
    if (element instanceof HTMLAnchorElement) {
      try { targetUrl = new URL(element.href, document.location.href).href; } catch (_) { targetUrl = null; }
    }
    let valueHint = null;
    if (element instanceof HTMLSelectElement) valueHint = element.value || null;
    else if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) valueHint = element.value || null;
    elements.push({
      element_id: elementId,
      role: roleName,
      affordance,
      text: text(element),
      name: label(element),
      value_hint: valueHint,
      landmark: landmark(element),
      disabled: Boolean(element.disabled || element.getAttribute('aria-disabled') === 'true'),
      visible: visible(element),
      target_url: targetUrl,
      form_action_origin: formActionOrigin,
      download: element.hasAttribute('download'),
      durable_state_change: Boolean(form && (form.method || '').toLowerCase() === 'post'),
    });
  }
  return {
    dom_generation: domGeneration(),
    url: document.location.href,
    title: document.title || '',
    text: bounded(document.body ? (document.body.innerText || document.body.textContent || '') : '', pageTextBytes),
    elements,
    loaded: document.readyState !== 'loading',
    stable: document.readyState === 'complete',
  };
}
"""


class PlaywrightBrowserAdapter:
    """Playwright implementation that keeps browser objects private."""

    def __init__(self, *, headless: bool, limits: SnapshotLimits) -> None:
        self._headless = headless
        self._limits = limits
        self._playwright: object | None = None
        self._browser: object | None = None
        self._context: object | None = None
        self._page: object | None = None
        self._locators: dict[str, object] = {}
        self._snapshot_id = 0
        self._generation = 0
        self._dom_generation: int | None = None
        self._url = ""
        self._navigation_origin_url: str | None = None
        self._closed = False
        self._navigation_interceptor: NavigationInterceptor | None = None
        self._navigation_error: NavigationBlockedError | None = None

    async def launch(self) -> None:
        if self._page is not None:
            return
        loaded = load_playwright_page()
        if _is_page_like(loaded):
            self._page = loaded
            self._closed = False
            await self._install_navigation_interception()
            return
        try:
            if hasattr(loaded, "async_playwright"):
                manager = loaded.async_playwright()
            else:
                manager = loaded() if callable(loaded) else loaded
            self._playwright = await _maybe_await(manager.start())
            chromium = self._playwright.chromium
            self._browser = await chromium.launch(headless=self._headless)
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()
            self._closed = False
            await self._install_navigation_interception()
        except Exception as exc:  # noqa: BLE001 - framework errors cross this seam
            await self.close()
            _raise_playwright_error(exc, "browser launch failed")

    async def navigate(self, url: str, timeout_ms: int) -> PageObservation:
        self._require_page()
        _validate_browser_url(url)
        self._navigation_origin_url = self._url or None
        self._url = ""
        self._navigation_error = None
        try:
            await self._page.goto(
                url, wait_until="domcontentloaded", timeout=timeout_ms
            )
            await self._page.wait_for_load_state("load", timeout=timeout_ms)
        except Exception as exc:
            if self._navigation_error is not None:
                error = self._navigation_error
                self._navigation_error = None
                raise error from exc
            _raise_playwright_error(exc, "browser navigation failed")
        finally:
            self._navigation_origin_url = None
        return await self.observe(self._limits)

    def set_navigation_interceptor(
        self, interceptor: NavigationInterceptor | None
    ) -> None:
        self._navigation_interceptor = interceptor

    async def observe(self, limits: SnapshotLimits) -> PageObservation:
        self._require_page()
        try:
            raw = await self._page.evaluate(
                OBSERVE_SCRIPT,
                {
                    "pageTextBytes": max(limits.page_text_bytes, 0),
                    "elementTextBytes": max(limits.element_text_bytes, 0),
                },
            )
        except Exception as exc:  # noqa: BLE001 - framework errors cross this seam
            _raise_playwright_error(exc, "browser observation failed")
        if not isinstance(raw, dict):
            raise BrowserError("browser observation returned an invalid shape")
        url = _raw_string(raw, "url")
        dom_generation = raw.get("dom_generation", 0)
        if not isinstance(dom_generation, int):
            raise BrowserError("browser observation returned an invalid generation")
        if url != self._url:
            self._generation += 1
            self._url = url
            self._dom_generation = dom_generation
        elif self._dom_generation != dom_generation:
            self._generation += 1
            self._dom_generation = dom_generation
        self._snapshot_id += 1
        self._locators.clear()
        elements: list[ElementRef] = []
        raw_elements = raw.get("elements")
        if not isinstance(raw_elements, list):
            raise BrowserError("browser observation returned invalid elements")
        for item in raw_elements:
            if not isinstance(item, dict):
                continue
            element = _element_from_raw(
                self._snapshot_id,
                self._generation,
                item,
                limits.element_text_bytes,
            )
            elements.append(element)
            self._locators[element.element_id] = self._page.locator(
                f'[data-zeta-browser-ref="{element.element_id}"]'
            )
        return PageObservation(
            snapshot_id=self._snapshot_id,
            generation=self._generation,
            url=url,
            title=_bounded_string(_raw_string(raw, "title"), limits.element_text_bytes),
            text=_bounded_string(_raw_string(raw, "text"), limits.page_text_bytes),
            elements=tuple(elements),
            loaded=bool(raw.get("loaded")),
            stable=bool(raw.get("stable")),
        )

    async def click(
        self, element_ref: ElementRef, timeout_ms: int
    ) -> ActionObservation:
        return await self._act(
            element_ref,
            "click",
            timeout_ms,
            lambda locator: locator.click(timeout=timeout_ms),
        )

    async def type_text(
        self,
        element_ref: ElementRef,
        text: str,
        replace: bool,
        timeout_ms: int,
    ) -> ActionObservation:
        async def action(locator: object) -> None:
            if replace:
                await locator.fill(text, timeout=timeout_ms)
            else:
                await locator.press_sequentially(text, timeout=timeout_ms)

        return await self._act(element_ref, "type", timeout_ms, action)

    async def select(
        self,
        element_ref: ElementRef,
        value: str,
        timeout_ms: int,
    ) -> ActionObservation:
        return await self._act(
            element_ref,
            "select",
            timeout_ms,
            lambda locator: locator.select_option(value=value, timeout=timeout_ms),
        )

    async def extract(
        self,
        target: ElementRef | None,
        attributes: list[str],
        limit: int,
    ) -> ExtractedData:
        locator = await self._target_locator(target)
        try:
            if not attributes:
                value = await locator.inner_text()
            else:
                values: dict[str, str | None] = {}
                for attribute in attributes:
                    if attribute == "text":
                        values[attribute] = await locator.inner_text()
                    elif attribute == "value":
                        values[attribute] = await locator.input_value()
                    else:
                        values[attribute] = await locator.get_attribute(attribute)
                value = values
        except Exception as exc:  # noqa: BLE001 - framework errors cross this seam
            _raise_playwright_error(exc, "browser extraction failed")
        return _bounded_extracted(value, limit)

    async def extract_search_results(
        self,
        target: ElementRef | None,
        limit: int,
    ) -> SearchResultExtraction:
        await self._require_reference(target)
        selector = "[data-search-result], article, [role=article]"
        if target is not None:
            selector = f'[data-zeta-browser-ref="{target.element_id}"] {selector}'
        try:
            raw_results = await self._page.locator(selector).evaluate_all(
                """
                (nodes) => nodes.map((node, index) => {
                  const link = node.querySelector('a[href]');
                  const heading = node.querySelector('h1,h2,h3,h4,h5,h6,[role=heading]');
                  const text = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
                  return {
                    result_id: node.getAttribute('data-result-id') || `r${index + 1}`,
                    title: text(heading?.innerText || link?.innerText || node.innerText),
                    snippet: text(node.innerText),
                    displayed_url: link?.href || '',
                    source_section: node.closest('main,nav,section,aside')?.getAttribute('aria-label') || 'page',
                    position: index + 1,
                  };
                })
                """
            )
        except Exception as exc:  # noqa: BLE001 - framework errors cross this seam
            _raise_playwright_error(exc, "browser search extraction failed")
        if not isinstance(raw_results, list):
            raise BrowserError("browser search extraction returned an invalid shape")
        results = tuple(
            SearchResultCandidate(
                result_id=_raw_string(item, "result_id"),
                title=_raw_string(item, "title"),
                snippet=_raw_string(item, "snippet"),
                displayed_url=_raw_string(item, "displayed_url"),
                source_section=_raw_string(item, "source_section"),
                position=int(item.get("position", index + 1)),
            )
            for index, item in enumerate(raw_results)
            if isinstance(item, dict)
        )
        return _bounded_search_results(results, limit)

    async def close(self) -> None:
        if self._closed and self._page is None:
            return
        self._closed = True
        self._locators.clear()
        for resource in (self._page, self._context, self._browser, self._playwright):
            if resource is not None:
                try:
                    if hasattr(resource, "close"):
                        await _maybe_await(resource.close())
                    else:
                        await _maybe_await(resource.stop())
                except Exception as exc:
                    _logger.debug("browser cleanup failed", exc_info=exc)
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    async def _act(
        self,
        element_ref: ElementRef,
        affordance: str,
        timeout_ms: int,
        action: Callable[[object], Awaitable[None] | None],
    ) -> ActionObservation:
        locator = await self._target_locator(element_ref)
        allowed_affordances = {affordance}
        if affordance == "click":
            allowed_affordances.add("submit")
        if element_ref.affordance not in allowed_affordances:
            raise ElementUnavailableError(element_ref.element_id)
        if self._url and hasattr(self._page, "url") and self._page.url != self._url:
            raise NavigationRaceError(element_ref.element_id)
        self._navigation_error = None
        try:
            await _maybe_await(action(locator))
        except Exception as exc:
            if self._navigation_error is not None:
                error = self._navigation_error
                self._navigation_error = None
                raise error from exc
            _raise_playwright_error(exc, "browser action failed")
        observation = await self.observe(self._limits)
        return ActionObservation(
            snapshot_id=observation.snapshot_id,
            generation=observation.generation,
            url=observation.url,
            loaded=observation.loaded,
            stable=observation.stable,
            changed=True,
        )

    async def _target_locator(self, target: ElementRef | None) -> object:
        self._require_page()
        if target is None:
            return self._page.locator("body")
        await self._require_reference(target)
        return self._locators[target.element_id]

    async def _require_reference(self, target: ElementRef | None) -> None:
        if target is None:
            return
        if target.snapshot_id != self._snapshot_id:
            raise ElementUnavailableError(target.element_id)
        await self._refresh_generation()
        if target.generation != self._generation:
            raise ElementUnavailableError(target.element_id)
        locator = self._locators.get(target.element_id)
        if locator is None:
            raise ElementUnavailableError(target.element_id)
        try:
            count = await _maybe_await(locator.count())
            if count != 1:
                raise ElementUnavailableError(target.element_id)
        except ElementUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - framework errors cross this seam
            _raise_playwright_error(exc, "browser element lookup failed")

    def _require_page(self) -> None:
        if self._page is None or self._closed:
            raise BrowserError("browser adapter is not launched")

    async def _refresh_generation(self) -> None:
        try:
            dom_generation = await self._page.evaluate(
                "() => Number(window.__zetaBrowserDomGeneration || 0)"
            )
        except Exception as exc:  # noqa: BLE001 - framework errors cross this seam
            _raise_playwright_error(exc, "browser generation check failed")
        if not isinstance(dom_generation, int):
            raise BrowserError("browser generation check returned an invalid value")
        if self._dom_generation != dom_generation:
            self._dom_generation = dom_generation
            self._generation += 1

    async def _install_navigation_interception(self) -> None:
        if self._page is None or not hasattr(self._page, "route"):
            return
        try:
            await _maybe_await(self._page.route("**/*", self._handle_route))
        except Exception as exc:
            raise BrowserError("browser navigation interception failed") from exc

    async def _handle_route(self, route: object) -> None:
        try:
            request = route.request
            if (
                self._is_top_level_navigation(request)
                and self._navigation_interceptor is not None
            ):
                await self._navigation_interceptor(
                    request.url,
                    self._url or self._navigation_origin_url,
                )
            await _maybe_await(route.continue_())
        except BaseException as exc:  # noqa: BLE001 - route must fail closed
            if isinstance(exc, NavigationBlockedError):
                error = exc
            else:
                error = NavigationBlockedError(
                    "browser navigation could not be classified safely"
                )
            self._navigation_error = error
            try:
                await _maybe_await(route.abort())
            except Exception as abort_error:
                _logger.debug("browser route abort failed", exc_info=abort_error)

    def _is_top_level_navigation(self, request: object) -> bool:
        try:
            is_navigation = request.is_navigation_request()
            frame = request.frame
            main_frame = self._page.main_frame
        except Exception as exc:
            raise NavigationBlockedError(
                "browser navigation request could not be classified safely"
            ) from exc
        return bool(is_navigation and frame == main_frame)


def make_browser_adapter_factory(
    *,
    headless: bool = True,
    limits: SnapshotLimits | None = None,
    mode: str | None = None,
) -> Callable[[], BrowserAdapter]:
    """Select the real adapter from the opt-in browser configuration."""

    selected = mode or os.environ.get("ZETA_BROWSER_ADAPTER", "fake")
    if selected == "fake":
        return lambda: FakeBrowserAdapter([_default_fake_observation()])
    if selected == "disabled":

        def disabled() -> BrowserAdapter:
            raise BrowserError("browser adapter is disabled; select fake or playwright")

        return disabled
    if selected != "playwright":
        raise ValueError(f"unknown browser adapter mode: {selected}")
    adapter_limits = limits or SnapshotLimits()
    return lambda: PlaywrightBrowserAdapter(headless=headless, limits=adapter_limits)


def _is_page_like(value: object) -> bool:
    return all(hasattr(value, name) for name in ("goto", "evaluate", "locator"))


async def _maybe_await(value: object) -> object:
    if hasattr(value, "__await__"):
        return await value
    return value


def _validate_browser_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme in {"http", "https"}:
        if not parts.hostname or parts.username or parts.password:
            raise BrowserError("browser navigation requires a valid origin")
        return
    if parts.scheme == "file":
        if parts.netloc or not PurePosixPath(parts.path).is_absolute():
            raise BrowserError("browser file navigation requires an absolute path")
        return
    raise BrowserError("browser navigation supports only http, https, and file URLs")


def _raw_string(raw: dict[str, object], name: str) -> str:
    value = raw.get(name, "")
    return value if isinstance(value, str) else str(value)


def _element_from_raw(
    snapshot_id: int,
    generation: int,
    raw: dict[str, object],
    element_text_bytes: int,
) -> ElementRef:
    return ElementRef(
        snapshot_id=snapshot_id,
        element_id=_raw_string(raw, "element_id"),
        role=_raw_string(raw, "role"),
        affordance=_raw_string(raw, "affordance"),
        text=_bounded_string(_raw_string(raw, "text"), element_text_bytes),
        name=_bounded_string(_raw_string(raw, "name"), element_text_bytes),
        value_hint=_bounded_optional_string(raw.get("value_hint"), element_text_bytes),
        landmark=_bounded_optional_string(raw.get("landmark"), element_text_bytes),
        disabled=bool(raw.get("disabled")),
        visible=bool(raw.get("visible")),
        target_url=raw.get("target_url")
        if isinstance(raw.get("target_url"), str)
        else None,
        form_action_origin=(
            raw.get("form_action_origin")
            if isinstance(raw.get("form_action_origin"), str)
            else None
        ),
        download=bool(raw.get("download")),
        durable_state_change=bool(raw.get("durable_state_change")),
        generation=generation,
    )


def _bounded_optional_string(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    return _bounded_string(value, limit)


def _default_fake_observation() -> PageObservation:
    return PageObservation(1, 1, "https://example.test/", "", "", (), True, True)


def _raise_playwright_error(exc: Exception, message: str) -> None:
    detail = f"{exc.__class__.__name__}: {exc}".lower()
    if exc.__class__.__name__ == "Error" and str(exc).startswith(
        "BrowserType.launch: Executable doesn't exist at "
    ):
        raise BrowserExecutableNotFoundError(message) from exc
    if (
        exc.__class__.__name__ in {"TimeoutError", "PlaywrightTimeoutError"}
        or "timeout" in detail
    ):
        raise BrowserTimeoutError(message) from exc
    if any(
        word in detail
        for word in ("detached", "not attached", "strict mode", "target closed")
    ):
        raise ElementUnavailableError(message) from exc
    raise BrowserError(message) from exc


class FakeBrowserAdapter:
    """Deterministic browser adapter for offline tests."""

    def __init__(
        self,
        observations: list[PageObservation],
        *,
        search_results: tuple[SearchResultCandidate, ...] | None = None,
    ) -> None:
        self._observations = list(observations)
        self._search_results = search_results
        self._observation_index = 0
        self._detached: set[str] = set()
        self._timeouts: set[str] = set()
        self._races: set[str] = set()
        self.navigations: list[str] = []
        self.clicks: list[ElementRef] = []
        self.typed: list[tuple[ElementRef, str, bool]] = []
        self.selected: list[tuple[ElementRef, str]] = []
        self.extractions: list[tuple[ElementRef | None, list[str], int]] = []
        self.search_extractions: list[tuple[ElementRef | None, int]] = []
        self._navigation_interceptor: NavigationInterceptor | None = None
        self._queued_navigation_url: str | None = None

    def detach(self, element_id: str) -> None:
        self._detached.add(element_id)

    def timeout_next(self, action: str) -> None:
        self._timeouts.add(action)

    def race_next(self, action: str) -> None:
        self._races.add(action)

    async def launch(self) -> None:
        return None

    def set_navigation_interceptor(
        self, interceptor: NavigationInterceptor | None
    ) -> None:
        self._navigation_interceptor = interceptor

    def queue_navigation(self, url: str) -> None:
        """Queue a top-level navigation caused by the next action."""

        self._queued_navigation_url = url

    async def trigger_navigation(self, url: str) -> None:
        """Drive a script-style top-level navigation in deterministic tests."""

        await self._intercept_navigation(url)
        self._replace_current_url(url)

    async def navigate(self, url: str, timeout_ms: int) -> PageObservation:
        self._maybe_fail("navigate")
        previous_index = self._observation_index
        await self._intercept_navigation(url)
        observation = self._advance_observation()
        if self._observation_index != previous_index and observation.url != url:
            await self._intercept_navigation(observation.url)
            self._replace_current_url(observation.url)
            result = self._current_observation()
        else:
            result = replace(observation, url=url)
        self.navigations.append(url)
        return result

    async def observe(self, limits: SnapshotLimits) -> PageObservation:
        del limits
        return self._current_observation()

    async def click(
        self, element_ref: ElementRef, timeout_ms: int
    ) -> ActionObservation:
        self._maybe_fail("click")
        self._check_element(element_ref)
        self.clicks.append(element_ref)
        return await self._next_action_observation()

    async def type_text(
        self,
        element_ref: ElementRef,
        text: str,
        replace: bool,
        timeout_ms: int,
    ) -> ActionObservation:
        del timeout_ms
        self._maybe_fail("type_text", "type")
        self._check_element(element_ref)
        self.typed.append((element_ref, text, replace))
        return await self._next_action_observation()

    async def select(
        self,
        element_ref: ElementRef,
        value: str,
        timeout_ms: int,
    ) -> ActionObservation:
        del timeout_ms
        self._maybe_fail("select")
        self._check_element(element_ref)
        self.selected.append((element_ref, value))
        return await self._next_action_observation()

    async def extract(
        self,
        target: ElementRef | None,
        attributes: list[str],
        limit: int,
    ) -> ExtractedData:
        self._maybe_fail("extract")
        if target is not None:
            self._check_element(target)
        self.extractions.append((target, list(attributes), limit))
        if attributes:
            value: str | dict[str, str | None] = {
                attribute: self._attribute_value(target, attribute)
                for attribute in attributes
            }
        else:
            value = "" if target is None else target.text
        return _bounded_extracted(value, limit)

    async def extract_search_results(
        self,
        target: ElementRef | None,
        limit: int,
    ) -> SearchResultExtraction:
        self._maybe_fail("extract_search_results")
        if target is not None:
            self._check_element(target)
        self.search_extractions.append((target, limit))
        if self._search_results is None:
            return SearchResultExtraction(None, False, 0)
        return _bounded_search_results(self._search_results, limit)

    async def close(self) -> None:
        return None

    def _current_observation(self) -> PageObservation:
        if not self._observations:
            raise BrowserError("fake browser has no observations")
        return self._observations[self._observation_index]

    def _advance_observation(self) -> PageObservation:
        if self._observation_index + 1 < len(self._observations):
            self._observation_index += 1
        return self._current_observation()

    async def _next_action_observation(self) -> ActionObservation:
        current = self._current_observation()
        queued_url = self._queued_navigation_url
        next_observation = self._peek_next_observation()
        target_url = queued_url
        if target_url is None and next_observation.url != current.url:
            target_url = next_observation.url
        if target_url is not None:
            await self._intercept_navigation(target_url)
        observation = self._advance_observation()
        self._queued_navigation_url = None
        if queued_url is not None:
            self._replace_current_url(queued_url)
            observation = self._current_observation()
        return ActionObservation(
            snapshot_id=observation.snapshot_id,
            generation=observation.generation,
            url=observation.url,
            loaded=observation.loaded,
            stable=observation.stable,
            changed=True,
        )

    def _peek_next_observation(self) -> PageObservation:
        if not self._observations:
            raise BrowserError("fake browser has no observations")
        index = min(self._observation_index + 1, len(self._observations) - 1)
        return self._observations[index]

    async def _intercept_navigation(self, url: str) -> None:
        if self._navigation_interceptor is None:
            return
        await self._navigation_interceptor(url, self._current_observation().url)

    def _replace_current_url(self, url: str) -> None:
        self._observations[self._observation_index] = replace(
            self._current_observation(), url=url
        )

    def _check_element(self, element_ref: ElementRef) -> None:
        if element_ref.element_id in self._detached:
            raise ElementUnavailableError(element_ref.element_id)

    def _maybe_fail(self, *actions: str) -> None:
        for action in actions:
            if action in self._timeouts:
                self._timeouts.remove(action)
                raise BrowserTimeoutError(action)
            if action in self._races:
                self._races.remove(action)
                raise NavigationRaceError(action)

    @staticmethod
    def _attribute_value(target: ElementRef | None, attribute: str) -> str | None:
        if target is None:
            return None
        values = {
            "text": target.text,
            "name": target.name,
            "value": target.value_hint,
            "href": target.target_url,
            "form_action_origin": target.form_action_origin,
            "landmark": target.landmark,
            "role": target.role,
        }
        return values.get(attribute)


def _bounded_extracted(
    value: str | dict[str, str | None],
    limit: int,
) -> ExtractedData:
    escaped_full_size = _escaped_size(value)
    if isinstance(value, dict):
        raw_full_size = sum(
            len(key.encode("utf-8")) + len((item or "").encode("utf-8"))
            for key, item in value.items()
        )
        remaining = max(limit, 0)
        bounded: dict[str, str | None] = {}
        for key, item in value.items():
            key_size = len(key.encode("utf-8"))
            if key_size > remaining:
                continue
            remaining -= key_size
            if item is None:
                bounded[key] = None
                continue
            bounded_item = _bounded_string(item, remaining)
            bounded[key] = bounded_item
            remaining -= len(bounded_item.encode("utf-8"))
        bounded_size = sum(
            len(key.encode("utf-8")) + len((item or "").encode("utf-8"))
            for key, item in bounded.items()
        )
        return ExtractedData(
            bounded,
            bounded_size < raw_full_size,
            raw_full_size,
            escaped_full_size,
        )
    encoded = value.encode("utf-8")
    bounded = _bounded_string(value, limit)
    return ExtractedData(
        bounded,
        len(bounded.encode("utf-8")) < len(encoded),
        len(encoded),
        escaped_full_size,
    )


def _escaped_size(value: str | dict[str, str | None]) -> int:
    serialized = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    return len(escape(serialized, quote=False).encode("utf-8"))


def _bounded_string(value: str, limit: int) -> str:
    return value.encode("utf-8")[: max(limit, 0)].decode("utf-8", errors="ignore")


def _bounded_search_results(
    results: tuple[SearchResultCandidate, ...],
    limit: int,
) -> SearchResultExtraction:
    full_size = sum(_search_result_size(result) for result in results)
    remaining = max(limit, 0)
    bounded: list[SearchResultCandidate] = []
    for result in results:
        result_size = _search_result_size(result)
        if result_size > remaining:
            break
        bounded.append(result)
        remaining -= result_size
    bounded_size = sum(_search_result_size(result) for result in bounded)
    return SearchResultExtraction(tuple(bounded), bounded_size < full_size, full_size)


def _search_result_size(result: SearchResultCandidate) -> int:
    return sum(
        len(value.encode("utf-8"))
        for value in (
            result.result_id,
            result.title,
            result.snippet,
            result.displayed_url,
            result.source_section,
            str(result.position),
        )
    )
