"""Stable browser tools, session lifecycle, and page-state helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from ...protocol.types import StructuredToolResult
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
        element = session.resolve_element(
            snapshot_id=snapshot_id,
            element_id=element_id,
            role=role,
            affordance=affordance,
        )
        _validate_action_affordance(action, affordance)
        _action, state = await session.action(
            action,
            element,
            text=text if isinstance(text, str) else None,
            replace=replace,
            value=value if isinstance(value, str) else None,
        )
    except Exception as exc:  # noqa: BLE001 - browser errors are structured
        return _browser_exception(exc)
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
        extracted = await session.extract(target, attributes, limit)
    except Exception as exc:  # noqa: BLE001 - browser errors are structured
        return _browser_exception(exc)
    content = _extracted_text(extracted.value)
    structured: dict[str, object] = {
        "value": extracted.value,
        "truncated": extracted.truncated,
        "full_size": extracted.full_size,
    }
    if extracted.truncated:
        structured["error"] = {
            "kind": "extraction_truncated",
            "hint": "reduce the extraction scope or increase its bounded limit",
            "message": "browser extraction was truncated",
        }
        return {
            "content": [text_block(content, full_size=extracted.full_size)],
            "isError": True,
            "structuredContent": structured,
        }
    return _success_result(text_block(content, full_size=extracted.full_size), structured_content=structured)


def register(registry: ToolRegistry) -> None:
    """Register the stable browser surface without opening a browser."""

    registry._browser_session = BrowserSession(
        lambda: _adapter_factory(registry),
        limits=SnapshotLimits(),
        navigation_timeout_ms=BROWSER_NAVIGATION_TIMEOUT_MS,
        action_timeout_ms=BROWSER_ACTION_TIMEOUT_MS,
        catalog_sink=lambda catalog: setattr(registry, "browser_catalog", catalog),
    )
    registry.browser_catalog = None
    registry.register_session_tool(
        "browser_navigate",
        _browser_navigate,
        description="Open an absolute http or https URL in the browser session.",
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
        description="Return the current bounded browser page state and element catalog.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        requires_approval=False,
        parallel_safe=True,
    )
    for name, handler, description, parameters in (
        (
            "browser_click",
            _browser_click,
            "Click one element from the current browser snapshot.",
            _element_schema(),
        ),
        (
            "browser_type",
            _browser_type,
            "Type text into one element from the current browser snapshot.",
            _element_schema({"text": {"type": "string"}, "replace": {"type": "boolean"}}, ["text", "replace"]),
        ),
        (
            "browser_select",
            _browser_select,
            "Select one option in an element from the current browser snapshot.",
            _element_schema({"value": {"type": "string"}}, ["value"]),
        ),
        (
            "browser_submit",
            _browser_submit,
            "Submit one element from the current browser snapshot.",
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
        description="Extract bounded text or attributes from the current browser page.",
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
