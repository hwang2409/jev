"""Bounded, framework-neutral browser catalogs and local filtering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .browser_adapter import ElementRef, PageObservation, SnapshotLimits


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


class SnapshotCatalogBuilder:
    def __init__(self, limits: SnapshotLimits) -> None:
        self.limits = limits
        self._current: BrowserCatalog | None = None

    def build(self, observation: PageObservation) -> BrowserCatalog:
        entries = tuple(
            _catalog_entry(element, self.limits.element_text_bytes)
            for element in observation.elements
            if element.visible or element.role in {"heading", "link", "button"}
        )
        title = _normalize_text(observation.title, self.limits.element_text_bytes)
        summary = _normalize_text(observation.text, self.limits.page_text_bytes)
        url = _truncate_utf8(observation.url, self.limits.catalog_bytes)
        previous = self._current
        previous_ids = frozenset() if previous is None else {
            entry.element_id for entry in previous.entries
        }
        generation_changed = previous is not None and (
            previous.generation != observation.generation or previous.url != observation.url
        )
        entries, title, summary, url = _fit_catalog(
            observation.snapshot_id,
            observation.generation,
            url,
            title,
            summary,
            entries,
            self.limits.catalog_bytes,
        )
        current_ids = {entry.element_id for entry in entries}
        invalidated = previous_ids if generation_changed else previous_ids - current_ids
        self._current = BrowserCatalog(
            observation.snapshot_id,
            observation.generation,
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
) -> tuple[tuple[CatalogEntry, ...], str, str, str]:
    bounded_limit = max(limit, 0)
    retained = entries
    while _serialized_size(snapshot_id, generation, url, title, summary, retained) > bounded_limit:
        if summary:
            summary = _truncate_utf8(summary, len(summary.encode("utf-8")) - 1)
        elif title:
            title = _truncate_utf8(title, len(title.encode("utf-8")) - 1)
        elif url:
            url = _truncate_utf8(url, len(url.encode("utf-8")) - 1)
        elif retained:
            retained = retained[:-1]
        else:
            break
    return retained, title, summary, url


def _serialized_size(
    snapshot_id: int,
    generation: int,
    url: str,
    title: str,
    summary: str,
    entries: tuple[CatalogEntry, ...],
) -> int:
    payload = {
        "snapshot_id": snapshot_id,
        "generation": generation,
        "url": url,
        "title": title,
        "summary": summary,
        "entries": [
            {
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
            for entry in entries
        ],
    }
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


ELEMENT_PREFILTER_K = 40
ELEMENT_CATALOG_MAX = 24


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
    phrase = 2 if goal.casefold().strip() and goal.casefold().strip() in entry.text.casefold() else 0
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
    for group in (0, 1, 2):
        if any(_tie_group(entry.role) == group for _, entry in retained):
            continue
        tied = next(
            (
                item
                for item in scored
                if _tie_group(item[1].role) == group
                and item[0] == min(score for score, _ in retained)
                and item[1].element_id not in {entry.element_id for _, entry in retained}
            ),
            None,
        )
        if tied is not None:
            retained[_replaceable_index(retained, prior_element_id)] = tied
    return retained


def _replaceable_index(
    retained: list[tuple[int, CatalogEntry]],
    prior_element_id: str | None,
) -> int:
    for index in range(len(retained) - 1, -1, -1):
        if retained[index][1].element_id != prior_element_id:
            return index
    return len(retained) - 1
