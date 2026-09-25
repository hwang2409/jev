"""Deterministic fake browser adapter."""

from __future__ import annotations

from dataclasses import replace

from .adapter import (
    ActionObservation,
    BrowserError,
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
)


def _default_fake_observation() -> PageObservation:
    return PageObservation(1, 1, "https://example.test/", "", "", (), True, True)


class _FakePage:
    async def route(self, _pattern: str, _handler: object) -> None:
        return None


class FakeBrowserAdapter:
    """Deterministic browser adapter for offline tests."""

    def __init__(
        self,
        observations: list[PageObservation],
        *,
        search_results: tuple[SearchResultCandidate, ...] | None = None,
        page: object | None = None,
    ) -> None:
        self._observations = list(observations)
        self._search_results = search_results
        self._page = page if page is not None else _FakePage()
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
        self.navigation_classifications: list[tuple[str, str | None]] = []
        self._navigation_guard: NavigationInterceptor | None = None
        self._navigation_error: NavigationBlockedError | None = None
        self._queued_navigation_url: str | None = None

    def detach(self, element_id: str) -> None:
        self._detached.add(element_id)

    def timeout_next(self, action: str) -> None:
        self._timeouts.add(action)

    def race_next(self, action: str) -> None:
        self._races.add(action)

    async def launch(self) -> None:
        if self._page is None or not hasattr(self._page, "route"):
            raise BrowserError(
                "browser page does not support navigation interception; launch aborted"
            )

    def install_navigation_guard(self, guard: NavigationInterceptor) -> None:
        if self._navigation_guard is not None and self._navigation_guard is not guard:
            raise BrowserError("browser navigation guard is already installed")
        self._navigation_guard = guard

    def queue_navigation(self, url: str) -> None:
        """Queue a top-level navigation caused by the next action."""

        self._queued_navigation_url = url

    async def trigger_navigation(self, url: str) -> None:
        """Drive a script-style top-level navigation in deterministic tests."""

        try:
            await self._intercept_navigation(url)
        except NavigationBlockedError as exc:
            self._navigation_error = exc
            return
        self._replace_current_url(url)

    async def navigate(self, url: str, timeout_ms: int) -> PageObservation:
        self._maybe_fail("navigate")
        previous_index = self._observation_index
        previous_url = self._current_observation().url
        await self._intercept_navigation(url, previous_url)
        observation = self._advance_observation()
        if self._observation_index != previous_index and observation.url != url:
            await self._intercept_navigation(observation.url, url)
            self._replace_current_url(observation.url)
            result = self._current_observation()
        else:
            result = replace(observation, url=url)
        self.navigations.append(url)
        return result

    async def observe(self, limits: SnapshotLimits) -> PageObservation:
        del limits
        if self._navigation_error is not None:
            error = self._navigation_error
            self._navigation_error = None
            raise error
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

    async def _intercept_navigation(
        self, url: str, current_url: str | None = None
    ) -> None:
        if self._navigation_guard is None:
            raise NavigationBlockedError(
                "browser navigation was blocked because no active browser operation "
                "can classify it safely"
            )
        source_url = current_url or self._current_observation().url
        self.navigation_classifications.append((url, source_url))
        await self._navigation_guard(url, source_url)

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
