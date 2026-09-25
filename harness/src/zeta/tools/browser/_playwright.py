"""Playwright browser adapter implementation."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from . import adapter as _adapter
from .adapter import (
    ActionObservation,
    BrowserError,
    BrowserExecutableNotFoundError,
    BrowserTimeoutError,
    ElementRef,
    ElementUnavailableError,
    ExtractedData,
    NavigationBlockedError,
    NavigationInterceptor,
    NavigationRaceError,
    PageObservation,
    SearchResultCandidate,
    SearchResultExtraction,
    SnapshotLimits,
    _bounded_extracted,
    _bounded_search_results,
    _bounded_string,
)

_logger = logging.getLogger(__name__)

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
        self._navigation_hop_url: str | None = None
        self._closed = False
        self._navigation_guard: NavigationInterceptor | None = None
        self._navigation_error: NavigationBlockedError | None = None

    async def launch(self) -> None:
        if self._page is not None:
            return
        loaded = _adapter.load_playwright_page()
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
        self._navigation_hop_url = self._url or None
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
            self._navigation_hop_url = None
        return await self.observe(self._limits)

    def install_navigation_guard(self, guard: NavigationInterceptor) -> None:
        if self._navigation_guard is not None and self._navigation_guard is not guard:
            raise BrowserError("browser navigation guard is already installed")
        self._navigation_guard = guard

    async def observe(self, limits: SnapshotLimits) -> PageObservation:
        self._require_page()
        if self._navigation_error is not None:
            error = self._navigation_error
            self._navigation_error = None
            raise error
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
        self._navigation_hop_url = self._url or None
        try:
            await _maybe_await(action(locator))
        except Exception as exc:
            if self._navigation_error is not None:
                error = self._navigation_error
                self._navigation_error = None
                raise error from exc
            _raise_playwright_error(exc, "browser action failed")
        finally:
            self._navigation_hop_url = None
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
            raise BrowserError(
                "browser page does not support navigation interception; launch aborted"
            )
        try:
            await _maybe_await(self._page.route("**/*", self._handle_route))
        except Exception as exc:
            raise BrowserError("browser navigation interception failed") from exc

    async def _handle_route(self, route: object) -> None:
        """Guard top-level routes; sub-frame requests remain out of scope."""

        try:
            request = route.request
            if self._is_top_level_navigation(request):
                if self._navigation_guard is None:
                    raise NavigationBlockedError(
                        "browser navigation was blocked because no active browser "
                        "operation can classify it safely"
                    )
                await self._navigation_guard(
                    request.url,
                    self._redirect_source(request),
                )
                self._navigation_hop_url = request.url
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

    def _redirect_source(self, request: object) -> str | None:
        redirected_from = getattr(request, "redirected_from", None)
        if redirected_from is not None:
            source_url = getattr(redirected_from, "url", None)
            if isinstance(source_url, str):
                return source_url
        return self._navigation_hop_url or self._url or self._navigation_origin_url

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
