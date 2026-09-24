"""Bounded, framework-neutral browser catalogs and local filtering."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...routing import (
    SEARCH_RESULT_CALL_CONFIDENCE_THRESHOLD,
    SEARCH_RESULT_RELEVANCE_FLOOR,
    SEARCH_RESULT_RELEVANCE_THRESHOLD,
    SEARCH_RESULT_TIE_MARGIN,
)
from .adapter import ElementRef, PageObservation, SnapshotLimits

if TYPE_CHECKING:
    from ...protocol.jev import SearchResultScoreResult


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    element_id: str
    role: str
    text: str
    affordance: str
    name: str
    value_hint: str | None
    landmark: str | None
    disabled: bool
    visible: bool


@dataclass(frozen=True, slots=True)
class BrowserCatalog:
    snapshot_id: int
    generation: int
    url: str
    title: str
    summary: str
    entries: tuple[CatalogEntry, ...]
    invalidated_element_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class PrefilterResult:
    candidates: tuple[CatalogEntry, ...]
    reason: str | None
    considered: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    result_id: str
    title: str
    snippet: str
    displayed_url: str
    source_section: str
    position: int


@dataclass(frozen=True, slots=True)
class SearchTriageDecision:
    accepted: str | None
    exposed: tuple[str, ...]
    reason: str


def triage_search_results(
    scores: SearchResultScoreResult,
    results: Iterable[SearchResult] | Mapping[str, SearchResult] | None = None,
    *,
    relevance_threshold: float = SEARCH_RESULT_RELEVANCE_THRESHOLD,
    tie_margin: float = SEARCH_RESULT_TIE_MARGIN,
    relevance_floor: float = SEARCH_RESULT_RELEVANCE_FLOOR,
    top_n: int = 3,
) -> SearchTriageDecision:
    """Apply separate relevance, tie, floor, and confidence rules."""

    source_by_id = _result_sources(results)
    ranked = _rank_search_scores(scores.scores, source_by_id)
    if not ranked:
        return SearchTriageDecision(None, (), "relevance_floor")
    if ranked[0][1] < relevance_floor:
        return SearchTriageDecision(None, (), "relevance_floor")
    score_gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
    close_tie = score_gap is not None and score_gap + 1e-12 < tie_margin
    if scores.call_confidence < SEARCH_RESULT_CALL_CONFIDENCE_THRESHOLD or close_tie:
        candidates = (
            [item for item in ranked if ranked[0][1] - item[1] < tie_margin]
            if close_tie
            else [item for item in ranked if item[1] >= relevance_floor]
        )
        return SearchTriageDecision(
            None,
            tuple(item_id for item_id, _score in candidates[: max(top_n, 0)]),
            "expose_candidates",
        )
    if ranked[0][1] < relevance_threshold:
        return SearchTriageDecision(None, (), "relevance_threshold")
    return SearchTriageDecision(ranked[0][0], (), "accepted")


def rank_search_result_ids(
    scores: SearchResultScoreResult,
    results: Iterable[SearchResult] | Mapping[str, SearchResult] | None = None,
) -> tuple[str, ...]:
    """Return score-ranked result ids with source diversity for ties."""

    return tuple(
        result_id
        for result_id, _score in _rank_search_scores(
            scores.scores, _result_sources(results)
        )
    )


def _result_sources(
    results: Iterable[SearchResult] | Mapping[str, SearchResult] | None,
) -> dict[str, str]:
    if results is None:
        return {}
    values = results.values() if isinstance(results, Mapping) else results
    return {result.result_id: result.source_section for result in values}


def _rank_search_scores(
    scores: Mapping[str, float], source_by_id: Mapping[str, str]
) -> list[tuple[str, float]]:
    """Sort scores while spreading equal-score results across sources."""

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    output: list[tuple[str, float]] = []
    index = 0
    while index < len(ranked):
        score = ranked[index][1]
        end = index + 1
        while end < len(ranked) and ranked[end][1] == score:
            end += 1
        group = ranked[index:end]
        used_sources: set[str] = set()
        while group:
            next_index = next(
                (
                    candidate_index
                    for candidate_index, (result_id, _score) in enumerate(group)
                    if source_by_id.get(result_id, result_id) not in used_sources
                ),
                0,
            )
            result_id, result_score = group.pop(next_index)
            used_sources.add(source_by_id.get(result_id, result_id))
            output.append((result_id, result_score))
        index = end
    return output


class SnapshotCatalogBuilder:
    def __init__(self, limits: SnapshotLimits) -> None:
        self.limits = limits
        self._current: BrowserCatalog | None = None
        self._snapshot_id = 0
        self._generation: int | None = None
        self._source_generation: int | None = None
        self._source_url: str | None = None

    def build(self, observation: PageObservation) -> BrowserCatalog:
        self._snapshot_id = max(self._snapshot_id + 1, observation.snapshot_id)
        entries = tuple(
            _catalog_entry(element, self.limits.element_text_bytes)
            for element in observation.elements
            if element.visible or element.role in {"heading", "link", "button"}
        )
        title = _normalize_text(observation.title, self.limits.element_text_bytes)
        summary = _normalize_text(observation.text, self.limits.page_text_bytes)
        url = _truncate_utf8(observation.url, self.limits.catalog_bytes)
        previous = self._current
        previous_ids = (
            frozenset()
            if previous is None
            else {entry.element_id for entry in previous.entries}
        )
        generation_changed = previous is not None and (
            self._source_generation != observation.generation
            or self._source_url != observation.url
        )
        if self._generation is None:
            self._generation = observation.generation
        elif generation_changed:
            self._generation = max(self._generation + 1, observation.generation)
        self._source_generation = observation.generation
        self._source_url = observation.url
        entries, title, summary, url = _fit_catalog(
            self._snapshot_id,
            self._generation,
            url,
            title,
            summary,
            entries,
            self.limits.catalog_bytes,
            previous_ids if generation_changed else frozenset(),
            previous_element_ids=None if generation_changed else previous_ids,
        )
        current_ids = {entry.element_id for entry in entries}
        invalidated = previous_ids if generation_changed else previous_ids - current_ids
        self._current = BrowserCatalog(
            self._snapshot_id,
            self._generation,
            url,
            title,
            summary,
            entries,
            frozenset(invalidated),
        )
        return self._current

    def is_current(self, snapshot_id: int, element_id: str) -> bool:
        if self._current is None or self._current.snapshot_id != snapshot_id:
            return False
        return element_id in {entry.element_id for entry in self._current.entries}


def _catalog_entry(element: ElementRef, limit: int) -> CatalogEntry:
    text = _normalize_text(element.name or element.text, limit)
    return CatalogEntry(
        element_id=element.element_id,
        role=element.role,
        text=text,
        affordance=element.affordance,
        name=_normalize_text(element.name, limit),
        value_hint=_optional_normalized(element.value_hint, limit),
        landmark=_optional_normalized(element.landmark, limit),
        disabled=element.disabled,
        visible=element.visible,
    )


def _optional_normalized(value: str | None, limit: int) -> str | None:
    return None if value is None else _normalize_text(value, limit)


def _normalize_text(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    return _truncate_utf8(normalized, limit).rstrip()


def _truncate_utf8(value: str, limit: int) -> str:
    return value.encode("utf-8")[: max(limit, 0)].decode("utf-8", errors="ignore")


def _fit_catalog(
    snapshot_id: int,
    generation: int,
    url: str,
    title: str,
    summary: str,
    entries: tuple[CatalogEntry, ...],
    limit: int,
    invalidated_element_ids: frozenset[str] = frozenset(),
    *,
    previous_element_ids: frozenset[str] | None = None,
) -> tuple[tuple[CatalogEntry, ...], str, str, str]:
    bounded_limit = max(limit, 0)
    entry_sizes = tuple(_json_size(_entry_payload(entry)) for entry in entries)
    entry_prefix_sizes = [0]
    for entry_size in entry_sizes:
        entry_prefix_sizes.append(entry_prefix_sizes[-1] + entry_size)

    if previous_element_ids is None:
        invalidated_sizes = [
            _invalidated_array_size(invalidated_element_ids)
            for _ in range(len(entries) + 1)
        ]
    else:
        invalidated_sizes = _invalidated_sizes_by_prefix(
            entries,
            previous_element_ids,
            invalidated_element_ids,
        )

    def serialized_size(
        retained_count: int,
        candidate_url: str,
        candidate_title: str,
        candidate_summary: str,
    ) -> int:
        return _serialized_size_from_parts(
            snapshot_id,
            generation,
            candidate_url,
            candidate_title,
            candidate_summary,
            entry_prefix_sizes[retained_count],
            retained_count,
            invalidated_sizes[retained_count],
        )

    retained_count = len(entries)
    if serialized_size(retained_count, url, title, summary) > bounded_limit:
        summary = _largest_fitting_text(
            summary,
            lambda candidate: serialized_size(retained_count, url, title, candidate),
            bounded_limit,
        )
    if serialized_size(retained_count, url, title, summary) > bounded_limit:
        title = _largest_fitting_text(
            title,
            lambda candidate: serialized_size(retained_count, url, candidate, summary),
            bounded_limit,
        )
    if serialized_size(retained_count, url, title, summary) > bounded_limit:
        url = _largest_fitting_text(
            url,
            lambda candidate: serialized_size(
                retained_count, candidate, title, summary
            ),
            bounded_limit,
        )
    while (
        retained_count
        and serialized_size(retained_count, url, title, summary) > bounded_limit
    ):
        retained_count -= 1
    return entries[:retained_count], title, summary, url


def _serialized_size(
    snapshot_id: int,
    generation: int,
    url: str,
    title: str,
    summary: str,
    entries: tuple[CatalogEntry, ...],
    invalidated_element_ids: frozenset[str] = frozenset(),
) -> int:
    payload = {
        "snapshot_id": snapshot_id,
        "generation": generation,
        "url": url,
        "title": title,
        "summary": summary,
        "entries": [_entry_payload(entry) for entry in entries],
        "invalidated_element_ids": sorted(invalidated_element_ids),
    }
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


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


def _json_size(value: object) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _array_size(item_sizes: list[int] | tuple[int, ...]) -> int:
    return 2 + sum(item_sizes) + max(len(item_sizes) - 1, 0)


def _field_size(name: str, value: object) -> int:
    return _json_size(name) + 1 + _json_size(value)


def _serialized_size_from_parts(
    snapshot_id: int,
    generation: int,
    url: str,
    title: str,
    summary: str,
    entry_size: int,
    entry_count: int,
    invalidated_size: int,
) -> int:
    fields = (
        _field_size("snapshot_id", snapshot_id),
        _field_size("generation", generation),
        _field_size("url", url),
        _field_size("title", title),
        _field_size("summary", summary),
        _json_size("entries") + 1 + entry_size + max(entry_count - 1, 0) + 2,
        _json_size("invalidated_element_ids") + 1 + invalidated_size,
    )
    return 2 + sum(fields) + len(fields) - 1


def _invalidated_array_size(element_ids: frozenset[str]) -> int:
    return _array_size(
        tuple(_json_size(element_id) for element_id in sorted(element_ids))
    )


def _invalidated_sizes_by_prefix(
    entries: tuple[CatalogEntry, ...],
    previous_element_ids: frozenset[str],
    fixed_element_ids: frozenset[str],
) -> list[int]:
    first_index: dict[str, int] = {}
    for index, entry in enumerate(entries):
        if entry.element_id in previous_element_ids:
            first_index.setdefault(entry.element_id, index)
    contributions = [(0, 0) for _ in range(len(entries) + 1)]
    for element_id in previous_element_ids:
        index = first_index.get(element_id, len(entries))
        count, size = contributions[index]
        contributions[index] = count + 1, size + _json_size(element_id)
    fixed_count = len(fixed_element_ids)
    fixed_size = sum(_json_size(element_id) for element_id in fixed_element_ids)
    result = [0] * (len(entries) + 1)
    count = fixed_count
    size = fixed_size
    for index in range(len(entries), -1, -1):
        added_count, added_size = contributions[index]
        count += added_count
        size += added_size
        result[index] = 2 + size + max(count - 1, 0)
    return result


def _largest_fitting_text(
    value: str,
    serialized_size: Callable[[str], int],
    limit: int,
) -> str:
    if not value or serialized_size(value) > limit:
        if not value:
            return value
        if serialized_size("") > limit:
            return ""
    low = 0
    high = len(value.encode("utf-8"))
    while low < high:
        midpoint = (low + high + 1) // 2
        candidate = _truncate_utf8(value, midpoint)
        if serialized_size(candidate) <= limit:
            low = midpoint
        else:
            high = midpoint - 1
    return _truncate_utf8(value, low)


ELEMENT_PREFILTER_K = 40
ELEMENT_CATALOG_MAX = 24
ELEMENT_RELEVANCE_FLOOR = 3
SUPPORTED_ROLES = frozenset(
    {
        "article",
        "button",
        "checkbox",
        "combobox",
        "form",
        "heading",
        "input",
        "link",
        "listbox",
        "main",
        "radio",
        "searchbox",
        "select",
        "tab",
        "text",
        "textbox",
    }
)


def prefilter_catalog(
    goal: str,
    action: str,
    catalog: BrowserCatalog,
    *,
    prefilter_k: int = ELEMENT_PREFILTER_K,
    catalog_max: int = ELEMENT_CATALOG_MAX,
    prior_element_id: str | None = None,
) -> PrefilterResult:
    eligible: list[CatalogEntry] = []
    seen: set[str] = set()
    normalized_action = action.casefold()
    for entry in catalog.entries:
        if (
            not entry.visible
            or entry.disabled
            or entry.affordance.casefold() != normalized_action
            or entry.role.casefold() not in SUPPORTED_ROLES
            or entry.element_id in seen
        ):
            continue
        seen.add(entry.element_id)
        eligible.append(entry)
    if not eligible or prefilter_k <= 0 or catalog_max <= 0:
        return PrefilterResult((), "no_candidate", len(eligible))

    all_scored = sorted(
        ((_score_entry(goal, action, entry), entry) for entry in eligible),
        key=lambda item: (-item[0], _tie_group(item[1].role), item[1].element_id),
    )
    if all_scored[0][0] < ELEMENT_RELEVANCE_FLOOR:
        return PrefilterResult((), "no_candidate", len(eligible))
    scored = all_scored[:prefilter_k]
    if prior_element_id is not None and prior_element_id not in {
        entry.element_id for _, entry in scored
    }:
        prior = next(
            (item for item in all_scored if item[1].element_id == prior_element_id),
            None,
        )
        if prior is not None and scored:
            scored[-1] = prior
    retained = _retain_candidates(scored, catalog_max, prior_element_id)
    return PrefilterResult(
        tuple(entry for _, entry in retained),
        None if retained else "no_candidate",
        len(eligible),
    )


def _score_entry(goal: str, action: str, entry: CatalogEntry) -> int:
    goal_tokens = _tokens(goal)
    searchable = _tokens(" ".join((entry.text, entry.name, entry.landmark or "")))
    lexical = len(goal_tokens & searchable) * 3
    phrase = (
        2
        if goal.casefold().strip() and goal.casefold().strip() in entry.text.casefold()
        else 0
    )
    role = _role_score(action, entry.role)
    return lexical + phrase + role


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _role_score(action: str, role: str) -> int:
    role = role.casefold()
    action = action.casefold()
    role_groups = {
        "click": {"button", "link", "tab", "checkbox", "radio"},
        "submit": {"button", "form"},
        "type": {"textbox", "text", "searchbox", "input"},
        "select": {"combobox", "listbox", "select"},
        "extract": {"heading", "link", "article", "main"},
    }
    return 2 if role in role_groups.get(action, set()) else 0


def _tie_group(role: str) -> int:
    role = role.casefold()
    if role == "link":
        return 0
    if role == "heading":
        return 2
    return 1


def _retain_candidates(
    scored: list[tuple[int, CatalogEntry]],
    catalog_max: int,
    prior_element_id: str | None,
) -> list[tuple[int, CatalogEntry]]:
    retained = list(scored[:catalog_max])
    if not retained:
        return retained
    score_by_id = {entry.element_id: score for score, entry in scored}
    if (
        prior_element_id is not None
        and prior_element_id in score_by_id
        and prior_element_id not in {entry.element_id for _, entry in retained}
    ):
        replacement = _replaceable_index(retained, prior_element_id)
        retained[replacement] = (
            score_by_id[prior_element_id],
            next(entry for _, entry in scored if entry.element_id == prior_element_id),
        )
    represented_groups = {_tie_group(entry.role) for _, entry in scored}
    for group in sorted(represented_groups):
        if any(_tie_group(entry.role) == group for _, entry in retained):
            continue
        tied = next(
            (
                item
                for item in scored
                if _tie_group(item[1].role) == group
                and item[0] == min(score for score, _ in retained)
                and item[1].element_id
                not in {entry.element_id for _, entry in retained}
            ),
            None,
        )
        replacement = _diversity_replacement_index(retained, prior_element_id)
        if tied is not None and replacement is not None:
            retained[replacement] = tied
    return retained


def _diversity_replacement_index(
    retained: list[tuple[int, CatalogEntry]],
    prior_element_id: str | None,
) -> int | None:
    group_counts: dict[int, int] = {}
    for _, entry in retained:
        group = _tie_group(entry.role)
        group_counts[group] = group_counts.get(group, 0) + 1
    for index in range(len(retained) - 1, -1, -1):
        score, entry = retained[index]
        del score
        if (
            entry.element_id != prior_element_id
            and group_counts[_tie_group(entry.role)] > 1
        ):
            return index
    return None


def _replaceable_index(
    retained: list[tuple[int, CatalogEntry]],
    prior_element_id: str | None,
) -> int:
    for index in range(len(retained) - 1, -1, -1):
        if retained[index][1].element_id != prior_element_id:
            return index
    return len(retained) - 1
