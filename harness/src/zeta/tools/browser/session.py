"""Session lifecycle for the framework-neutral browser tools."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .adapter import (
    ActionObservation,
    BrowserAdapter,
    BrowserError,
    ElementRef,
    ElementUnavailableError,
    ExtractedData,
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
    ) -> None:
        self._factory = factory
        self.limits = limits or SnapshotLimits()
        self.navigation_timeout_ms = navigation_timeout_ms
        self.action_timeout_ms = action_timeout_ms
        self._catalog_sink = catalog_sink
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

    async def adapter(self) -> BrowserAdapter:
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

    async def navigate(self, url: str) -> BrowserState:
        observation = await (
            await self.adapter()
        ).navigate(url, self.navigation_timeout_ms)
        state = self._record_observation(observation)
        self._record_action(f"navigate:{url}")
        return state

    async def observe(self) -> BrowserState:
        observation = await (await self.adapter()).observe(self.limits)
        return self._record_observation(observation)

    async def action(
        self,
        action: str,
        element_ref: ElementRef,
        *,
        text: str | None = None,
        replace: bool = True,
        value: str | None = None,
    ) -> tuple[ActionObservation, BrowserState]:
        adapter = await self.adapter()
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
        state = await self.observe()
        self._record_action(action)
        return result, state

    async def extract(
        self,
        target: ElementRef | None,
        attributes: list[str],
        limit: int,
    ) -> ExtractedData:
        return await (await self.adapter()).extract(target, attributes, limit)

    async def extract_search_results(
        self,
        target: ElementRef | None,
        limit: int,
    ) -> SearchResultExtraction:
        return await (await self.adapter()).extract_search_results(target, limit)

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
            (candidate for candidate in catalog.entries if candidate.element_id == element_id),
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
