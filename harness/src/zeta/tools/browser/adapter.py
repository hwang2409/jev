"""Framework-neutral browser adapter values and deterministic fake."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol


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


class BrowserAdapter(Protocol):
    async def launch(self) -> None:
        raise NotImplementedError

    async def navigate(self, url: str, timeout_ms: int) -> PageObservation:
        raise NotImplementedError

    async def observe(self, limits: SnapshotLimits) -> PageObservation:
        raise NotImplementedError

    async def click(self, element_ref: ElementRef, timeout_ms: int) -> ActionObservation:
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

    async def close(self) -> None:
        raise NotImplementedError


class BrowserError(RuntimeError):
    """Base error translated into a stable browser tool error kind."""


class ElementUnavailableError(BrowserError):
    """The requested element is detached or ambiguous."""


class BrowserTimeoutError(BrowserError):
    """The adapter exceeded its bounded timeout."""


class NavigationRaceError(BrowserError):
    """Navigation changed page generation during an action."""


class FakeBrowserAdapter:
    """Deterministic browser adapter for offline tests."""

    def __init__(self, observations: list[PageObservation]) -> None:
        self._observations = list(observations)
        self._observation_index = 0
        self._detached: set[str] = set()
        self._timeouts: set[str] = set()
        self._races: set[str] = set()
        self.navigations: list[str] = []
        self.clicks: list[ElementRef] = []
        self.typed: list[tuple[ElementRef, str, bool]] = []
        self.selected: list[tuple[ElementRef, str]] = []
        self.extractions: list[tuple[ElementRef | None, list[str], int]] = []

    def detach(self, element_id: str) -> None:
        self._detached.add(element_id)

    def timeout_next(self, action: str) -> None:
        self._timeouts.add(action)

    def race_next(self, action: str) -> None:
        self._races.add(action)

    async def launch(self) -> None:
        return None

    async def navigate(self, url: str, timeout_ms: int) -> PageObservation:
        self._maybe_fail("navigate")
        observation = self._advance_observation()
        self.navigations.append(url)
        return replace(observation, url=url)

    async def observe(self, limits: SnapshotLimits) -> PageObservation:
        del limits
        return self._current_observation()

    async def click(self, element_ref: ElementRef, timeout_ms: int) -> ActionObservation:
        self._maybe_fail("click")
        self._check_element(element_ref)
        self.clicks.append(element_ref)
        return self._next_action_observation()

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
        return self._next_action_observation()

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
        return self._next_action_observation()

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
                attribute: self._attribute_value(target, attribute) for attribute in attributes
            }
        else:
            value = "" if target is None else target.text
        return _bounded_extracted(value, limit)

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

    def _next_action_observation(self) -> ActionObservation:
        observation = self._advance_observation()
        return ActionObservation(
            snapshot_id=observation.snapshot_id,
            generation=observation.generation,
            url=observation.url,
            loaded=observation.loaded,
            stable=observation.stable,
            changed=True,
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
    if isinstance(value, dict):
        full_size = sum(len((item or "").encode("utf-8")) for item in value.values())
        remaining = max(limit, 0)
        bounded: dict[str, str | None] = {}
        for key, item in value.items():
            if item is None:
                bounded[key] = None
                continue
            bounded_item = _bounded_string(item, remaining)
            bounded[key] = bounded_item
            remaining -= len(bounded_item.encode("utf-8"))
        bounded_size = sum(len((item or "").encode("utf-8")) for item in bounded.values())
        return ExtractedData(bounded, bounded_size < full_size, full_size)
    encoded = value.encode("utf-8")
    bounded = _bounded_string(value, limit)
    return ExtractedData(bounded, len(bounded.encode("utf-8")) < len(encoded), len(encoded))


def _bounded_string(value: str, limit: int) -> str:
    return value.encode("utf-8")[: max(limit, 0)].decode("utf-8", errors="ignore")
