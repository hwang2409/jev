"""Session lifecycle for the framework-neutral browser tools."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ...routing import browser_threshold_version
from .adapter import (
    ActionObservation,
    BrowserAdapter,
    BrowserError,
    ElementRef,
    ElementUnavailableError,
    ExtractedData,
    NavigationInterceptor,
    PageObservation,
    SearchResultExtraction,
    SnapshotLimits,
)
from .catalog import BrowserCatalog, CatalogEntry, SnapshotCatalogBuilder

BROWSER_NAVIGATION_TIMEOUT_MS = 30_000
BROWSER_ACTION_TIMEOUT_MS = 10_000
_logger = logging.getLogger(__name__)

AdapterFactory = Callable[[], BrowserAdapter | Awaitable[BrowserAdapter]]
CatalogSink = Callable[[BrowserCatalog | None], None]
Clock = Callable[[], float]


@dataclass(slots=True)
class BrowserBudget:
    """Bound one browser task across provider calls, tokens, actions, and time."""

    page_jev_call_limit: int = 8
    page_jev_token_limit: int = 12_000
    task_action_limit: int = 20
    task_wall_clock_seconds: float = 120.0
    started_at: float | None = None
    page_jev_calls: int = 0
    page_jev_tokens: int = 0
    task_actions: int = 0


class BrowserBudgetExhaustedError(BrowserError):
    """The browser task reached one of its named limits."""


@dataclass(frozen=True, slots=True)
class BrowserState:
    """The latest observation and bounded catalog owned by one session."""

    observation: PageObservation
    catalog: BrowserCatalog


class BrowserSession:
    """Own one lazy adapter and the latest snapshot-local element refs."""

    def __init__(
        self,
        factory: AdapterFactory,
        *,
        limits: SnapshotLimits | None = None,
        navigation_timeout_ms: int = BROWSER_NAVIGATION_TIMEOUT_MS,
        action_timeout_ms: int = BROWSER_ACTION_TIMEOUT_MS,
        catalog_sink: CatalogSink | None = None,
        page_jev_call_budget: int = 8,
        page_jev_token_budget: int = 12_000,
        task_action_budget: int = 20,
        task_wall_clock_seconds: float = 120.0,
        clock: Clock = time.monotonic,
        element_top1_confidence: float = 0.8,
        element_topn: int = 3,
        search_relevance_threshold: float = 0.7,
        search_tie_margin: float = 0.1,
        search_relevance_floor: float = 0.4,
        search_call_confidence_threshold: float = 0.8,
    ) -> None:
        self._factory = factory
        self.limits = limits or SnapshotLimits()
        self.navigation_timeout_ms = navigation_timeout_ms
        self.action_timeout_ms = action_timeout_ms
        self._catalog_sink = catalog_sink
        self._clock = clock
        self.element_top1_confidence = element_top1_confidence
        self.element_topn = element_topn
        self.search_relevance_threshold = search_relevance_threshold
        self.search_tie_margin = search_tie_margin
        self.search_relevance_floor = search_relevance_floor
        self.search_call_confidence_threshold = search_call_confidence_threshold
        self.threshold_version = browser_threshold_version(
            element_top1_confidence=element_top1_confidence,
            element_topn=element_topn,
            search_relevance_threshold=search_relevance_threshold,
            search_tie_margin=search_tie_margin,
            search_relevance_floor=search_relevance_floor,
            search_call_confidence_threshold=search_call_confidence_threshold,
        )
        self.budget = BrowserBudget(
            page_jev_call_limit=page_jev_call_budget,
            page_jev_token_limit=page_jev_token_budget,
            task_action_limit=task_action_budget,
            task_wall_clock_seconds=task_wall_clock_seconds,
            started_at=None,
        )
        self._adapter: BrowserAdapter | None = None
        self._adapter_lock = asyncio.Lock()
        self._state: BrowserState | None = None
        self._builder = SnapshotCatalogBuilder(self.limits)
        self._element_refs: dict[str, ElementRef] = {}
        self.recent_actions: list[str] = []
        self.recovery_attempts = 0

    @property
    def adapter_instance(self) -> BrowserAdapter | None:
        return self._adapter

    @property
    def state(self) -> BrowserState | None:
        return self._state

    @property
    def catalog(self) -> BrowserCatalog | None:
        return None if self._state is None else self._state.catalog

    async def adapter(self, *, check_budget: bool = True) -> BrowserAdapter:
        if check_budget:
            self.ensure_available()
        if self._adapter is not None:
            return self._adapter
        async with self._adapter_lock:
            if self._adapter is not None:
                return self._adapter
            adapter = self._factory()
            if inspect.isawaitable(adapter):
                adapter = await adapter
            try:
                await adapter.launch()
            except BaseException:
                try:
                    await adapter.close()
                except Exception as cleanup_error:
                    _logger.debug(
                        "browser startup cleanup failed",
                        exc_info=cleanup_error,
                    )
                raise
            self._adapter = adapter
            return adapter

    async def open(self) -> BrowserAdapter:
        """Open the lazy adapter on first use."""

        return await self.adapter()

    async def attach(self, adapter: BrowserAdapter) -> None:
        """Attach an already-created adapter for deterministic tests or reuse."""

        if self._adapter is not None and self._adapter is not adapter:
            raise BrowserError("browser session already has an adapter")
        self._adapter = adapter

    async def navigate(
        self,
        url: str,
        *,
        navigation_interceptor: NavigationInterceptor | None = None,
    ) -> BrowserState:
        self._ensure_task_available()
        adapter = await self.adapter(check_budget=False)
        self.budget.task_actions += 1
        self._set_navigation_interceptor(adapter, navigation_interceptor)
        try:
            observation = await adapter.navigate(url, self.navigation_timeout_ms)
        finally:
            self._set_navigation_interceptor(adapter, None)
        self._reset_page_budget()
        state = self._record_observation(observation)
        self._record_action(f"navigate:{url}")
        return state

    async def observe(self, *, check_budget: bool = True) -> BrowserState:
        if check_budget:
            self.ensure_available()
        adapter = self._adapter
        if adapter is None:
            adapter = await self.adapter()
        observation = await adapter.observe(self.limits)
        return self._record_observation(observation)

    async def action(
        self,
        action: str,
        element_ref: ElementRef,
        *,
        text: str | None = None,
        replace: bool = True,
        value: str | None = None,
        navigation_interceptor: NavigationInterceptor | None = None,
    ) -> tuple[ActionObservation, BrowserState]:
        adapter = await self.adapter()
        self.consume_action()
        previous_url = None if self._state is None else self._state.observation.url
        self._set_navigation_interceptor(adapter, navigation_interceptor)
        try:
            if action in {"click", "submit"}:
                result = await adapter.click(element_ref, self.action_timeout_ms)
            elif action == "type":
                if text is None:
                    raise ValueError("browser_type requires text")
                result = await adapter.type_text(
                    element_ref, text, replace, self.action_timeout_ms
                )
            elif action == "select":
                if value is None:
                    raise ValueError("browser_select requires value")
                result = await adapter.select(
                    element_ref, value, self.action_timeout_ms
                )
            else:
                raise ValueError(f"unsupported browser action: {action}")
            state = await self.observe(check_budget=False)
            if previous_url != state.observation.url:
                self._reset_page_budget()
        finally:
            self._set_navigation_interceptor(adapter, None)
        self._record_action(action)
        return result, state

    @staticmethod
    def _set_navigation_interceptor(
        adapter: BrowserAdapter,
        interceptor: NavigationInterceptor | None,
    ) -> None:
        setter = getattr(adapter, "set_navigation_interceptor", None)
        if callable(setter):
            setter(interceptor)

    async def extract(
        self,
        target: ElementRef | None,
        attributes: list[str],
        limit: int,
    ) -> ExtractedData:
        self.ensure_available()
        return await (await self.adapter()).extract(target, attributes, limit)

    async def extract_search_results(
        self,
        target: ElementRef | None,
        limit: int,
    ) -> SearchResultExtraction:
        self.ensure_available()
        return await (await self.adapter()).extract_search_results(target, limit)

    async def call_jev(
        self, operation: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Run one browser Jev call and charge its final provider usage once."""

        self.ensure_available()
        if self.budget.page_jev_calls >= self.budget.page_jev_call_limit:
            raise BrowserBudgetExhaustedError("browser page Jev call budget exhausted")
        self.budget.page_jev_calls += 1
        result = await operation(*args, **kwargs)
        usage = getattr(result, "usage", {})
        tokens = _reported_tokens(usage)
        total = self.budget.page_jev_tokens + tokens
        self.budget.page_jev_tokens = min(total, self.budget.page_jev_token_limit)
        if total > self.budget.page_jev_token_limit:
            raise BrowserBudgetExhaustedError("browser page Jev token budget exhausted")
        return result

    def ensure_available(self) -> None:
        self._ensure_task_available()
        if self.budget.page_jev_calls >= self.budget.page_jev_call_limit:
            raise BrowserBudgetExhaustedError("browser page Jev call budget exhausted")
        if self.budget.page_jev_tokens >= self.budget.page_jev_token_limit:
            raise BrowserBudgetExhaustedError("browser page Jev token budget exhausted")

    def _ensure_task_available(self) -> None:
        if self.budget.started_at is None:
            self.budget.started_at = self._clock()
        elapsed = self._clock() - self.budget.started_at
        if elapsed >= self.budget.task_wall_clock_seconds:
            raise BrowserBudgetExhaustedError(
                "browser task wall-clock budget exhausted"
            )
        if self.budget.task_actions >= self.budget.task_action_limit:
            raise BrowserBudgetExhaustedError("browser task action budget exhausted")

    def consume_action(self) -> None:
        self.ensure_available()
        self.budget.task_actions += 1

    def resolve_element(
        self,
        *,
        snapshot_id: int,
        element_id: str,
        role: str,
        affordance: str,
    ) -> ElementRef:
        catalog = self.catalog
        if catalog is None or catalog.snapshot_id != snapshot_id:
            raise StaleSnapshotError(snapshot_id)
        entry = next(
            (
                candidate
                for candidate in catalog.entries
                if candidate.element_id == element_id
            ),
            None,
        )
        if entry is None:
            raise ElementUnavailableError(element_id)
        if entry.role != role or entry.affordance != affordance:
            raise ElementUnavailableError(
                f"{element_id} role or affordance does not match the catalog"
            )
        try:
            return self._element_refs[element_id]
        except KeyError as exc:
            raise ElementUnavailableError(element_id) from exc

    def resolve_optional_element(
        self,
        *,
        snapshot_id: int | None,
        element_id: str | None,
    ) -> ElementRef | None:
        if snapshot_id is None and element_id is None:
            return None
        if snapshot_id is None or element_id is None:
            raise StaleSnapshotError(snapshot_id)
        catalog = self.catalog
        if catalog is None or catalog.snapshot_id != snapshot_id:
            raise StaleSnapshotError(snapshot_id)
        try:
            return self._element_refs[element_id]
        except KeyError as exc:
            raise ElementUnavailableError(element_id) from exc

    def reset_turn_state(self) -> None:
        """Reset router-local browser state without closing the page."""

        self.recent_actions.clear()
        self.recovery_attempts = 0
        self.budget.started_at = None
        self.budget.task_actions = 0

    async def close(self) -> None:
        adapter = self._adapter
        self._adapter = None
        self._state = None
        self._element_refs.clear()
        self._publish_catalog(None)
        if adapter is not None:
            try:
                await adapter.close()
            except Exception as exc:
                # Browser cleanup is best effort and must not mask the caller's error.
                _logger.debug("browser cleanup failed", exc_info=exc)

    def _record_observation(self, observation: PageObservation) -> BrowserState:
        catalog = self._builder.build(observation)
        self._state = BrowserState(observation, catalog)
        catalog_ids = {entry.element_id for entry in catalog.entries}
        self._element_refs = {
            element.element_id: element
            for element in observation.elements
            if element.element_id in catalog_ids
        }
        self._publish_catalog(catalog)
        return self._state

    def _reset_page_budget(self) -> None:
        self.budget.page_jev_calls = 0
        self.budget.page_jev_tokens = 0

    def _record_action(self, action: str) -> None:
        self.recent_actions.append(action)
        del self.recent_actions[:-3]

    def _publish_catalog(self, catalog: BrowserCatalog | None) -> None:
        if self._catalog_sink is not None:
            self._catalog_sink(catalog)


class StaleSnapshotError(BrowserError):
    """The action references a snapshot other than the current one."""


class BrowserSessionClosedError(BrowserError):
    """The registry closed the browser session permanently."""


def _reported_tokens(usage: object) -> int:
    if not isinstance(usage, dict):
        return 0
    return sum(
        value
        for key in ("input_tokens", "output_tokens")
        for value in [usage.get(key)]
        if type(value) is int and value >= 0
    )


def catalog_payload(catalog: BrowserCatalog) -> dict[str, object]:
    """Serialize a bounded catalog without exposing adapter objects."""

    return {
        "snapshot_id": catalog.snapshot_id,
        "generation": catalog.generation,
        "url": catalog.url,
        "title": catalog.title,
        "summary": catalog.summary,
        "entries": [_entry_payload(entry) for entry in catalog.entries],
        "invalidated_element_ids": sorted(catalog.invalidated_element_ids),
    }


def _entry_payload(entry: CatalogEntry) -> dict[str, object]:
    return {
        "element_id": entry.element_id,
        "role": entry.role,
        "text": entry.text,
        "affordance": entry.affordance,
        "name": entry.name,
        "value_hint": entry.value_hint,
        "landmark": entry.landmark,
        "disabled": entry.disabled,
        "visible": entry.visible,
    }
