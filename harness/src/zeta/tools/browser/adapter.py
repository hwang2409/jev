"""Framework-neutral browser adapter values and public re-exports."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from html import escape
from typing import Protocol

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
    """Adapter contract for arc-3 actions in the main document.

    The guard covers top-level navigations. Sub-frame requests stay outside
    this scope because tools do not act inside sub-frames and extraction is
    bounded to the main document. Add sub-frame classification before adding
    sub-frame interaction or extraction.
    """

    async def launch(self) -> None:
        raise NotImplementedError

    def install_navigation_guard(self, guard: NavigationInterceptor) -> None:
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

from ._fake import FakeBrowserAdapter, _default_fake_observation
from ._playwright import (  # noqa: F401
    OBSERVE_SCRIPT,
    PlaywrightBrowserAdapter,
    load_playwright_page,
)


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
