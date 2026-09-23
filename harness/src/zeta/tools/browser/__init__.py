"""Stable browser tools, session lifecycle, and page-state helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from ...core.abort import AbortSignal
from ...core.safety import (
    BrowserRiskEvidence,
    SafetyOutcome,
    browser_action_requires_safety,
)
from ...protocol.types import StructuredToolResult
from ...providers import jev
from ...routing import BROWSER_ELEMENT_TOP1_CONFIDENCE, BROWSER_ELEMENT_TOPN
from ...runtime.execution import ToolExecutionContext
from ..registry import ToolRegistry, _error_result, _success_result, text_block
from .adapter import (
    BrowserError,
    BrowserTimeoutError,
    NavigationRaceError,
    SearchResultCandidate,
    SnapshotLimits,
)
from .adapter import (
    ElementUnavailableError as AdapterElementUnavailableError,
)
from .catalog import (
    SearchResult,
    prefilter_catalog,
    rank_search_result_ids,
    triage_search_results,
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
    BrowserSessionClosedError,
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
        if registry.browser_session_closed:
            raise BrowserSessionClosedError("browser session is closed")
        raise BrowserError("browser session is not registered")
    return session


def _adapter_factory(registry: ToolRegistry):
    factory = registry.browser_adapter_factory
    if factory is None:
        raise BrowserError("browser adapter factory is not configured")
    return factory()


async def _browser_navigate(
    registry: ToolRegistry,
    arguments: dict[str, object],
    abort_signal: AbortSignal | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult:
    url = arguments["url"]
    if not isinstance(url, str) or not _is_absolute_http_url(url):
        return _browser_error(
            "browser_navigate requires an absolute http or https URL",
            "invalid_arguments",
        )
    try:
        session = _session(registry)
        current_url = url
        if session.state is not None:
            current_url = session.state.observation.url
        evidence = BrowserRiskEvidence(
            action="navigate",
            role="navigation",
            text=url,
            current_origin=_origin(current_url) or "",
            target_url=url,
            form_action_origin=None,
            payment_language=False,
            authentication_language=False,
            download=False,
            durable_state_change=False,
        )
        if browser_action_requires_safety(evidence):
            safety_error = await _check_browser_safety(
                registry,
                evidence,
                arguments,
                abort_signal=abort_signal,
                execution_context=execution_context,
            )
            if safety_error is not None:
                return safety_error
        state = await session.navigate(url)
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
    registry: ToolRegistry,
    arguments: dict[str, object],
    abort_signal: AbortSignal | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult:
    return await _run_element_action(
        registry,
        arguments,
        "click",
        abort_signal=abort_signal,
        execution_context=execution_context,
    )


async def _browser_type(
    registry: ToolRegistry,
    arguments: dict[str, object],
    abort_signal: AbortSignal | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult:
    return await _run_element_action(
        registry,
        arguments,
        "type",
        text=arguments.get("text"),
        replace=arguments.get("replace", True),
        abort_signal=abort_signal,
        execution_context=execution_context,
    )


async def _browser_select(
    registry: ToolRegistry,
    arguments: dict[str, object],
    abort_signal: AbortSignal | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult:
    return await _run_element_action(
        registry,
        arguments,
        "select",
        value=arguments.get("value"),
        abort_signal=abort_signal,
        execution_context=execution_context,
    )


async def _browser_submit(
    registry: ToolRegistry,
    arguments: dict[str, object],
    abort_signal: AbortSignal | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult:
    return await _run_element_action(
        registry,
        arguments,
        "submit",
        abort_signal=abort_signal,
        execution_context=execution_context,
    )


async def _run_element_action(
    registry: ToolRegistry,
    arguments: dict[str, object],
    action: str,
    *,
    text: object = None,
    replace: object = True,
    value: object = None,
    abort_signal: AbortSignal | None = None,
    execution_context: ToolExecutionContext | None = None,
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
        evidence = BrowserRiskEvidence(
            action=action,
            role=element.role,
            text=" ".join(part for part in (element.text, element.name) if part),
            current_origin=_origin(current_state.observation.url) or "",
            target_url=element.target_url,
            form_action_origin=element.form_action_origin,
            payment_language=_has_payment_language(element.text, element.name),
            authentication_language=_has_authentication_language(
                element.text, element.name
            ),
            download=element.download,
            durable_state_change=element.durable_state_change or action == "submit",
        )
        if browser_action_requires_safety(evidence):
            safety_error = await _check_browser_safety(
                registry,
                evidence,
                arguments,
                abort_signal=abort_signal,
                execution_context=execution_context,
            )
            if safety_error is not None:
                return safety_error
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
        search_extracted = None
        if not attributes:
            search_extracted = await session.extract_search_results(target, limit)
        extracted = None
        if search_extracted is None or search_extracted.results is None:
            extracted = await session.extract(target, attributes, limit)
    except Exception as exc:  # noqa: BLE001 - browser errors are structured
        return _browser_exception(exc)
    triage = None
    if search_extracted is not None and search_extracted.results is not None:
        triage = await _triage_search_results(
            _browser_goal(registry, "extract", element_id or "page", None),
            search_extracted.results,
        )
        value = triage["value"]
        truncated = search_extracted.truncated
        full_size = search_extracted.full_size
    else:
        assert extracted is not None
        value = extracted.value
        truncated = extracted.truncated
        full_size = extracted.full_size
    content = _extracted_text(value)
    structured: dict[str, object] = {
        "value": value,
        "truncated": truncated,
        "full_size": full_size,
    }
    if triage is not None:
        structured.update(triage)
        content = f"{_triage_receipt(triage['triage'])}\n{content}"
    return _success_result(text_block(content, full_size=full_size), structured_content=structured)


async def _triage_search_results(
    goal: str, results: tuple[SearchResultCandidate, ...]
) -> dict[str, object]:
    records = [
        SearchResult(
            result_id=result.result_id,
            title=result.title,
            snippet=result.snippet,
            displayed_url=result.displayed_url,
            source_section=result.source_section,
            position=result.position,
        )
        for result in results
    ]
    provider_items = [
        {
            "result_id": result.result_id,
            "title": result.title,
            "snippet": result.snippet,
            "displayed_url": result.displayed_url,
            "source_section": result.source_section,
            "position": str(result.position),
        }
        for result in records
    ]
    if not provider_items:
        return {
            "results": [],
            "ranked_results": [],
            "triage": {
                "status": "ranked",
                "decision": "relevance_floor",
                "warnings": ["floor"],
            },
            "value": {"results": []},
        }
    try:
        scores = await jev.score_search_results(goal, provider_items)
    except Exception as exc:  # noqa: BLE001 - triage is advisory to extraction
        warning = f"search results returned unranked because Jev triage failed: {exc}"
        return {
            "results": provider_items,
            "ranked_results": provider_items,
            "warning": warning,
            "triage": {
                "status": "unranked",
                "decision": "degraded",
                "warnings": ["degraded"],
                "warning": warning,
            },
            "value": {"results": provider_items},
        }
    decision = triage_search_results(scores, records)
    items_by_id = {item["result_id"]: item for item in provider_items}
    ranked_results = [
        {
            **items_by_id[result_id],
            "rank": rank,
            "relevance_score": scores.scores[result_id],
        }
        for rank, result_id in enumerate(
            rank_search_result_ids(scores, records), start=1
        )
    ]
    triage_payload = {
        "status": "ranked",
        "decision": decision.reason,
        "accepted": decision.accepted,
        "exposed": list(decision.exposed),
        "warnings": _triage_warnings(decision.reason),
        "call_confidence": scores.call_confidence,
        "usage": dict(scores.usage),
    }
    return {
        "results": ranked_results,
        "ranked_results": ranked_results,
        "triage": triage_payload,
        "value": {"results": ranked_results},
    }


def _triage_warnings(decision: str) -> list[str]:
    if decision == "expose_candidates":
        return ["tie"]
    if decision == "relevance_floor":
        return ["floor"]
    return []


def _triage_receipt(triage: Mapping[str, object]) -> str:
    status = triage.get("status", "unknown")
    decision = triage.get("decision", "unknown")
    warnings = triage.get("warnings", [])
    warning_text = ""
    if isinstance(warnings, list) and warnings:
        warning_text = f" warnings={','.join(str(warning) for warning in warnings)}"
    return f"search triage: status={status} decision={decision}{warning_text}"


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


def _origin(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    host = parsed.hostname.casefold()
    if port is None or (parsed.scheme == "http" and port == 80) or (
        parsed.scheme == "https" and port == 443
    ):
        return f"{parsed.scheme.casefold()}://{host}"
    return f"{parsed.scheme.casefold()}://{host}:{port}"


def _has_payment_language(*values: str) -> bool:
    text = " ".join(values).casefold()
    return any(
        word in text
        for word in (
            "buy",
            "checkout",
            "donate",
            "pay",
            "payment",
            "purchase",
            "subscribe",
            "transfer",
        )
    )


def _has_authentication_language(*values: str) -> bool:
    text = " ".join(values).casefold()
    return any(
        word in text
        for word in (
            "account",
            "authenticate",
            "authentication",
            "login",
            "log-in",
            "password",
            "permission",
            "sign-in",
            "token",
        )
    )


async def _check_browser_safety(
    registry: ToolRegistry,
    evidence: BrowserRiskEvidence,
    arguments: dict[str, object],
    *,
    abort_signal: AbortSignal | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult | None:
    safety_tier = registry.safety_tier
    if safety_tier is None:
        return _browser_error(
            "browser safety tier is not configured for this risky action",
            "safety_denied",
        )
    try:
        outcome = await safety_tier.evaluate_browser_action(evidence)
    except Exception as exc:  # noqa: BLE001 - safety must fail closed
        outcome = safety_tier.fail_closed(exc)
    if outcome.decision == "allow":
        return None
    if (
        outcome.decision == "ask"
        and registry.approval_policy is not None
        and abort_signal is not None
        and execution_context is not None
    ):
        gate_result, _execution_signal = await registry._approval_gate.run(
            execution_context.tool_call,
            arguments,
            abort_signal,
            lambda current: registry._next_abort_generation(current),
            execution_context.lifecycle_sink,
            skip_approval=True,
            safety_outcome=outcome,
            approval_label=_browser_approval_label(registry, evidence, outcome),
        )
        if gate_result is None:
            return None
        return _browser_error(gate_result.content, "safety_denied")
    message = safety_tier.teaching_error(outcome)
    return _browser_error(message, "safety_denied")


def _browser_approval_label(
    registry: ToolRegistry,
    evidence: BrowserRiskEvidence,
    outcome: SafetyOutcome,
) -> str:
    safety_tier = registry.safety_tier
    if safety_tier is None:
        return "browser action requires approval"
    details = [
        f"action={evidence.action}",
        f"target_text={evidence.text or 'unknown'}",
    ]
    if evidence.target_url is not None:
        details.append(f"destination={evidence.target_url}")
    if evidence.form_action_origin is not None:
        details.append(f"form_action={evidence.form_action_origin}")
    details.append(f"risk_reason={outcome.reason or 'safety_threshold_not_met'}")
    return f"{safety_tier.approval_label(outcome)}; " + ", ".join(details)


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


def _extracted_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _browser_exception(exc: Exception) -> StructuredToolResult:
    if isinstance(exc, BrowserSessionClosedError):
        return _browser_error(
            str(exc) or "browser session is closed", "browser_session_closed"
        )
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
