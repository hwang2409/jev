"""Stable browser tools, session lifecycle, and page-state helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from ...protocol.types import StructuredToolResult
from ...providers import jev
from ...routing import BROWSER_ELEMENT_TOP1_CONFIDENCE, BROWSER_ELEMENT_TOPN
from ..registry import ToolRegistry, _error_result, _success_result, text_block
from .adapter import (
    BrowserError,
    BrowserTimeoutError,
    NavigationRaceError,
    SnapshotLimits,
)
from .adapter import (
    ElementUnavailableError as AdapterElementUnavailableError,
)
from .catalog import prefilter_catalog
from .gates import (
    PAGE_STATE_RECOVERY_ATTEMPT_CAP,
    PageStateDecision,
    conservative_provider_error_decision,
    evaluate_page_state,
    evaluate_page_state_with_provider,
)
from .session import (
    BROWSER_ACTION_TIMEOUT_MS,
    BROWSER_NAVIGATION_TIMEOUT_MS,
    BrowserSession,
    StaleSnapshotError,
    catalog_payload,
)

BROWSER_TOOL_NAMES = (
    "browser_navigate",
    "browser_state",
    "browser_click",
    "browser_type",
    "browser_select",
    "browser_extract",
    "browser_submit",
)

BASE_ELEMENT_PROPERTIES = {
    "snapshot_id": {"type": "integer", "minimum": 1},
    "element_id": {"type": "string", "minLength": 1},
    "role": {"type": "string", "minLength": 1},
    "affordance": {"type": "string", "minLength": 1},
}


def _session(registry: ToolRegistry) -> BrowserSession:
    session = registry.browser_session
    if not isinstance(session, BrowserSession):
        raise BrowserError("browser session is not registered")
    return session


def _adapter_factory(registry: ToolRegistry):
    factory = registry.browser_adapter_factory
    if factory is None:
        raise BrowserError("browser adapter factory is not configured")
    return factory()


async def _browser_navigate(
    registry: ToolRegistry, arguments: dict[str, object]
) -> StructuredToolResult:
    url = arguments["url"]
    if not isinstance(url, str) or not _is_absolute_http_url(url):
        return _browser_error(
            "browser_navigate requires an absolute http or https URL",
            "invalid_arguments",
        )
    try:
        state = await _session(registry).navigate(url)
    except Exception as exc:  # noqa: BLE001 - browser errors are structured
        return _browser_exception(exc)
    return _state_result(state)


async def _browser_state(
    registry: ToolRegistry, arguments: dict[str, object]
) -> StructuredToolResult:
    del arguments
    try:
        state = await _session(registry).observe()
    except Exception as exc:  # noqa: BLE001 - browser errors are structured
        return _browser_exception(exc)
    return _state_result(state)


async def _browser_click(
    registry: ToolRegistry, arguments: dict[str, object]
) -> StructuredToolResult:
    return await _run_element_action(registry, arguments, "click")


async def _browser_type(
    registry: ToolRegistry, arguments: dict[str, object]
) -> StructuredToolResult:
    return await _run_element_action(
        registry,
        arguments,
        "type",
        text=arguments.get("text"),
        replace=arguments.get("replace", True),
    )


async def _browser_select(
    registry: ToolRegistry, arguments: dict[str, object]
) -> StructuredToolResult:
    return await _run_element_action(
        registry, arguments, "select", value=arguments.get("value")
    )


async def _browser_submit(
    registry: ToolRegistry, arguments: dict[str, object]
) -> StructuredToolResult:
    return await _run_element_action(registry, arguments, "submit")


async def _run_element_action(
    registry: ToolRegistry,
    arguments: dict[str, object],
    action: str,
    *,
    text: object = None,
    replace: object = True,
    value: object = None,
) -> StructuredToolResult:
    try:
        snapshot_id = arguments["snapshot_id"]
        element_id = arguments["element_id"]
        role = arguments["role"]
        affordance = arguments["affordance"]
        if not all(
            isinstance(item, str)
            for item in (element_id, role, affordance)
        ) or not isinstance(snapshot_id, int):
            raise ValueError("browser element identity is invalid")
        if text is not None and not isinstance(text, str):
            raise ValueError("browser_type text must be a string")
        if not isinstance(replace, bool):
            raise TypeError("browser_type replace must be a boolean")
        if value is not None and not isinstance(value, str):
            raise ValueError("browser_select value must be a string")
        session = _session(registry)
        session.resolve_element(
            snapshot_id=snapshot_id,
            element_id=element_id,
            role=role,
            affordance=affordance,
        )
        _validate_action_affordance(action, affordance)
        current_state = session.state
        if current_state is None:
            raise StaleSnapshotError(snapshot_id)
        caller_entry = next(
            entry
            for entry in current_state.catalog.entries
            if entry.element_id == element_id
        )
        goal = _browser_goal(registry, action, element_id, caller_entry.text)
        filtered = prefilter_catalog(
            goal,
            action,
            current_state.catalog,
            prior_element_id=element_id,
        )
        candidates = _candidate_payloads(filtered.candidates)
        if not candidates:
            return _browser_error(
                "no browser element matches the requested action",
                "jev_routing_error",
            )
        page_state = catalog_payload(current_state.catalog)
        choice = await jev.choose_browser_element(
            goal,
            action,
            page_state,
            candidates,
            session.recent_actions,
        )
        if (
            choice.confidence < BROWSER_ELEMENT_TOP1_CONFIDENCE
            or choice.element_id is None
        ):
            return _choice_result(
                page_state,
                action,
                choice,
                candidates,
            )
        selected_entry = next(
            (
                candidate
                for candidate in filtered.candidates
                if candidate.element_id == choice.element_id
            ),
            None,
        )
        if selected_entry is None or choice.affordance != selected_entry.affordance:
            return _browser_error(
                "jev selected an element outside the current browser catalog",
                "jev_routing_error",
            )
        pre_gate = await evaluate_page_state_with_provider(
            goal=goal,
            action=action,
            page_state=page_state,
            candidates=candidates,
            deterministic_loaded=(
                current_state.observation.loaded and current_state.observation.stable
            ),
            deterministic_attached=True,
            recent_actions=session.recent_actions,
        )
        if not pre_gate.allow_action:
            return _browser_error(
                "browser page-state gate blocked the action",
                pre_gate.error_kind or "jev_routing_error",
            )
        element = session.resolve_element(
            snapshot_id=current_state.catalog.snapshot_id,
            element_id=selected_entry.element_id,
            role=selected_entry.role,
            affordance=selected_entry.affordance,
        )
        _action, state = await session.action(
            action,
            element,
            text=text if isinstance(text, str) else None,
            replace=replace,
            value=value if isinstance(value, str) else None,
        )
        post_payload = catalog_payload(state.catalog)
        post_gate = await evaluate_page_state_with_provider(
            goal=goal,
            action=action,
            page_state=post_payload,
            candidates=_candidate_payloads(state.catalog.entries),
            deterministic_loaded=state.observation.loaded and state.observation.stable,
            deterministic_attached=True,
            recent_actions=session.recent_actions,
            action_result={
                "changed": _action.changed,
                "snapshot_id": _action.snapshot_id,
                "generation": _action.generation,
                "url": _action.url,
                "loaded": _action.loaded,
                "stable": _action.stable,
            },
            previous_page_state=page_state,
            recovery_attempts=session.recovery_attempts,
        )
        session.recovery_attempts = post_gate.recovery_attempts
    except Exception as exc:  # noqa: BLE001 - browser errors are structured
        if isinstance(exc, jev.JevRouterError):
            return _browser_error(str(exc), "jev_routing_error")
        return _browser_exception(exc)
    if not post_gate.allow_action:
        return _action_state_result(
            state,
            action=action,
            action_result=_action,
            error_kind=post_gate.error_kind,
            recovery=post_gate.recovery,
        )
    return _state_result(state, action=action)


async def _browser_extract(
    registry: ToolRegistry, arguments: dict[str, object]
) -> StructuredToolResult:
    try:
        session = _session(registry)
        snapshot_id = arguments.get("snapshot_id")
        element_id = arguments.get("element_id")
        if snapshot_id is not None and not isinstance(snapshot_id, int):
            raise ValueError("browser_extract snapshot_id must be an integer or null")
        if element_id is not None and not isinstance(element_id, str):
            raise ValueError("browser_extract element_id must be a string or null")
        target = session.resolve_optional_element(
            snapshot_id=snapshot_id,
            element_id=element_id,
        )
        attributes = arguments.get("attributes", [])
        limit = arguments.get("limit", session.limits.extracted_bytes)
        if not isinstance(attributes, list) or not all(
            isinstance(attribute, str) for attribute in attributes
        ):
            raise ValueError("browser_extract attributes must be strings")
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("browser_extract limit must be positive")
        limit = min(limit, session.limits.extracted_bytes)
        extracted = await session.extract(target, attributes, limit)
    except Exception as exc:  # noqa: BLE001 - browser errors are structured
        return _browser_exception(exc)
    content = _extracted_text(extracted.value)
    structured: dict[str, object] = {
        "value": extracted.value,
        "truncated": extracted.truncated,
        "full_size": extracted.full_size,
    }
    return _success_result(text_block(content, full_size=extracted.full_size), structured_content=structured)


def register(registry: ToolRegistry) -> None:
    """Register the stable browser surface without opening a browser."""

    def session_factory(target: ToolRegistry) -> BrowserSession:
        return BrowserSession(
            lambda: _adapter_factory(target),
            limits=SnapshotLimits(),
            navigation_timeout_ms=BROWSER_NAVIGATION_TIMEOUT_MS,
            action_timeout_ms=BROWSER_ACTION_TIMEOUT_MS,
            catalog_sink=lambda catalog: setattr(target, "browser_catalog", catalog),
        )

    registry._browser_session_factory = session_factory
    registry._browser_session = session_factory(registry)
    registry.browser_catalog = None
    registry.register_session_tool(
        "browser_navigate",
        _browser_navigate,
        description="Open an allowed URL in the session page.",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "minLength": 1}},
            "required": ["url"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
    registry.register_session_tool(
        "browser_state",
        _browser_state,
        description="Return the current bounded page snapshot and element catalog.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        requires_approval=False,
    )
    for name, handler, description, parameters in (
        (
            "browser_click",
            _browser_click,
            "Click one catalog element by stable snapshot id.",
            _element_schema(),
        ),
        (
            "browser_type",
            _browser_type,
            "Replace or append text in one input by snapshot id.",
            _element_schema({"text": {"type": "string"}, "replace": {"type": "boolean"}}, ["text", "replace"]),
        ),
        (
            "browser_select",
            _browser_select,
            "Select one option in a select control by snapshot id and value.",
            _element_schema({"value": {"type": "string"}}, ["value"]),
        ),
        (
            "browser_submit",
            _browser_submit,
            "Submit a form or click the identified submit control after safety approval.",
            _element_schema(),
        ),
    ):
        registry.register_session_tool(
            name,
            handler,
            description=description,
            parameters=parameters,
            requires_approval=False,
        )
    registry.register_session_tool(
        "browser_extract",
        _browser_extract,
        description="Return bounded text or selected attributes from one element or the page.",
        parameters={
            "type": "object",
            "properties": {
                "snapshot_id": {"type": ["integer", "null"], "minimum": 1},
                "element_id": {"type": ["string", "null"], "minLength": 1},
                "attributes": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        requires_approval=False,
        parallel_safe=True,
    )


def _element_schema(
    extra: Mapping[str, object] | None = None,
    required_extra: list[str] | None = None,
) -> dict[str, object]:
    properties: dict[str, object] = dict(BASE_ELEMENT_PROPERTIES)
    if extra:
        properties.update(extra)
    return {
        "type": "object",
        "properties": properties,
        "required": [
            *BASE_ELEMENT_PROPERTIES,
            *(required_extra or []),
        ],
        "additionalProperties": False,
    }


def _is_absolute_http_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _browser_goal(
    registry: ToolRegistry,
    action: str,
    element_id: str,
    element_text: str,
) -> str:
    if isinstance(registry.browser_goal, str) and registry.browser_goal.strip():
        return registry.browser_goal
    recent_steps = registry.router_recent_steps or []
    if recent_steps:
        return recent_steps[-1]
    return f"{action} {element_text or f'browser element {element_id}'}"


def _candidate_payloads(entries: Any) -> list[dict[str, object]]:
    return [
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
    ]


def _choice_result(
    page_state: dict[str, object],
    action: str,
    choice: Any,
    candidates: list[dict[str, object]],
) -> StructuredToolResult:
    by_id = {
        item["element_id"]: item
        for item in candidates
        if isinstance(item.get("element_id"), str)
    }
    candidate_ids = [
        element_id
        for element_id in choice.candidate_ids
        if element_id in by_id
    ][:BROWSER_ELEMENT_TOPN]
    if not candidate_ids:
        candidate_ids = [
            element_id
            for element_id, _probability in sorted(
                choice.probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if element_id in by_id
        ][:BROWSER_ELEMENT_TOPN]
    payload = {
        "snapshot_id": page_state["snapshot_id"],
        "generation": page_state["generation"],
        "action": action,
        "routed": False,
        "requires_choice": True,
        "confidence": choice.confidence,
        "call_confidence": choice.call_confidence,
        "candidate_ids": candidate_ids,
        "candidates": [by_id[element_id] for element_id in candidate_ids],
        "usage": dict(choice.usage),
    }
    return _success_result(
        text_block(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        structured_content=payload,
    )


def _action_state_result(
    state: Any,
    *,
    action: str,
    action_result: Any,
    error_kind: str | None,
    recovery: str | None,
) -> StructuredToolResult:
    payload = catalog_payload(state.catalog)
    payload.update(
        {
            "action": action,
            "action_result": {
                "changed": action_result.changed,
                "snapshot_id": action_result.snapshot_id,
                "generation": action_result.generation,
                "url": action_result.url,
                "loaded": action_result.loaded,
                "stable": action_result.stable,
            },
            "progress_unknown": True,
            "recovery": recovery,
            "error": {
                "kind": error_kind or "action_outcome_unknown",
                "message": "browser action outcome is unknown",
            },
        }
    )
    return _success_result(
        text_block(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        structured_content=payload,
    )


def _validate_action_affordance(action: str, affordance: str) -> None:
    expected = "submit" if action == "submit" else action
    if affordance != expected:
        raise AdapterElementUnavailableError(
            f"element affordance {affordance!r} does not support {expected!r}"
        )


def _state_result(state: Any, *, action: str | None = None) -> StructuredToolResult:
    if not state.observation.loaded or not state.observation.stable:
        return _browser_error(
            "browser page is not loaded and stable",
            "page_load_failed",
        )
    payload = catalog_payload(state.catalog)
    if action is not None:
        payload["action"] = action
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return _success_result(text_block(text), structured_content=payload)


def _extracted_text(value: str | dict[str, str | None]) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _browser_exception(exc: Exception) -> StructuredToolResult:
    if isinstance(exc, StaleSnapshotError):
        return _browser_error(str(exc) or "browser snapshot is stale", "stale_snapshot")
    if isinstance(exc, AdapterElementUnavailableError):
        return _browser_error(str(exc) or "browser element is unavailable", "element_unavailable")
    if isinstance(exc, NavigationRaceError):
        return _browser_error(str(exc) or "browser navigation raced with the action", "navigation_race")
    if isinstance(exc, BrowserTimeoutError):
        return _browser_error(str(exc) or "browser operation timed out", "browser_timeout")
    if isinstance(exc, ValueError):
        return _browser_error(str(exc), "invalid_arguments")
    return _browser_error(str(exc) or "browser operation failed", "browser_start_failed")


def _browser_error(message: str, kind: str) -> StructuredToolResult:
    return _error_result(message, kind=kind)


__all__ = [
    "BASE_ELEMENT_PROPERTIES",
    "BROWSER_ACTION_TIMEOUT_MS",
    "BROWSER_NAVIGATION_TIMEOUT_MS",
    "BROWSER_TOOL_NAMES",
    "PAGE_STATE_RECOVERY_ATTEMPT_CAP",
    "BrowserSession",
    "PageStateDecision",
    "conservative_provider_error_decision",
    "evaluate_page_state",
    "evaluate_page_state_with_provider",
]
