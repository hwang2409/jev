from __future__ import annotations

import json as _json
import time as _time
from collections.abc import Callable as _Callable
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from ._transport import (
    _GATEWAY_MODEL,
    _GatewayTransport,
    _resolve_gateway_key,
    _transport_identity,
    _TransportResponse,
)
from .answers import (
    ErrorResponse as _ErrorResponse,
)
from .answers import (
    JudgeResponse as _JudgeResponse,
)
from .answers import (
    parse_judge_response as _parse_judge_response,
)
from .answers import (
    probability_keys_for_question as _probability_keys_for_question,
)
from .answers import (
    score_legend_for_question as _score_legend_for_question,
)
from .cache import CacheStore
from .runner import (
    ConfigurationError,
    EmitResult,
    FormationEvent,
    FormationReport,
    InputError,
    InputSidecar,
    ResultFilter,
    State,
    emit,
    judge,
    judge_async,
)

if _TYPE_CHECKING:
    from .runner import State as _State


class JevError(RuntimeError):
    """A safe error raised by the public Jev client."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.attempts = attempts

    @property
    def status_code(self) -> int | None:
        return self.http_status

    @property
    def error(self) -> str:
        return self.message


JevResponse = _JudgeResponse


@_dataclass(frozen=True, slots=True)
class CanonicalRequest:
    payload: dict[str, _Any]
    request_bytes: bytes
    transport_identity: dict[str, str]


def build_canonical_request(
    state: _State | _Mapping[str, _Any],
    questions: _Mapping[str, _Any],
    *,
    model: str = _GATEWAY_MODEL,
) -> CanonicalRequest:
    """Build the one request object used for sends and cache keys."""
    _validate_model(model)
    if isinstance(state, _Mapping):
        state_payload = dict(state)
    else:
        state_payload = state.api_payload
    payload = {
        "providerOptions": {"gateway": {"zeroDataRetention": True}},
        "state": state_payload,
        "questions": _gateway_questions(questions),
    }
    request_bytes = _json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return CanonicalRequest(payload, request_bytes, _transport_identity(model))


class JevClient:
    """Synchronous and asynchronous client for the Jev evaluation service."""

    def __init__(self, *, _transport: _GatewayTransport | None = None) -> None:
        self._transport = _transport or _GatewayTransport()

    def set_response_observer(self, observer: _Callable[[int], None] | None) -> None:
        self._transport.set_response_observer(observer)

    def evaluate(
        self,
        state: _State | _Mapping[str, _Any],
        questions: _Mapping[str, _Any],
        *,
        model: str = _GATEWAY_MODEL,
    ) -> JevResponse:
        _validate_model(model)
        api_key = _require_gateway_key()
        request = build_canonical_request(state, questions, model=model)
        started = _time.monotonic()
        response, attempts = self._transport.post(
            request.request_bytes, api_key, model=model
        )
        if isinstance(response, _ErrorResponse):
            raise _as_jev_error(response)

        parsed = self._parse_response(response, questions, started, attempts)
        if parsed.complete or attempts >= self._transport.max_attempts:
            return parsed

        response, retry_attempts = self._transport.post(
            request.request_bytes, api_key, attempts, model=model
        )
        if isinstance(response, _ErrorResponse):
            raise _as_jev_error(response)
        return self._parse_response(response, questions, started, retry_attempts)

    async def evaluate_async(
        self,
        state: _State | _Mapping[str, _Any],
        questions: _Mapping[str, _Any],
        *,
        model: str = _GATEWAY_MODEL,
    ) -> JevResponse:
        _validate_model(model)
        api_key = _require_gateway_key()
        request = build_canonical_request(state, questions, model=model)
        started = _time.monotonic()
        response, attempts = await self._transport.apost(
            request.request_bytes, api_key, model=model
        )
        if isinstance(response, _ErrorResponse):
            raise _as_jev_error(response)

        parsed = self._parse_response(response, questions, started, attempts)
        if parsed.complete or attempts >= self._transport.max_attempts:
            return parsed

        response, retry_attempts = await self._transport.apost(
            request.request_bytes, api_key, attempts, model=model
        )
        if isinstance(response, _ErrorResponse):
            raise _as_jev_error(response)
        return self._parse_response(response, questions, started, retry_attempts)

    def __call__(
        self,
        state: _State | _Mapping[str, _Any],
        questions: _Mapping[str, _Any],
        model: str = _GATEWAY_MODEL,
    ) -> JevResponse | _ErrorResponse:
        try:
            return self.evaluate(state, questions, model=model)
        except JevError as exc:
            return _ErrorResponse(
                exc.message,
                http_status=exc.http_status,
                attempts=exc.attempts,
            )

    def close(self) -> None:
        self._transport.close()

    async def aclose(self) -> None:
        await self._transport.aclose()

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> JevClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    @staticmethod
    def _parse_response(
        response: _TransportResponse,
        questions: _Mapping[str, _Any],
        started: float,
        attempts: int,
    ) -> JevResponse:
        parsed_response = _parse_gateway_response(response, questions)
        if parsed_response is None:
            raise JevError(
                "malformed answer",
                http_status=response.status_code,
                attempts=attempts,
            ) from None
        payload, parsed, usage = parsed_response
        return _replace(
            parsed,
            served_model=_served_model(payload),
            usage=usage,
            latency_ms=round((_time.monotonic() - started) * 1000),
        )


def _parse_gateway_response(
    response: _TransportResponse,
    questions: _Mapping[str, _Any],
) -> tuple[_Mapping[str, _Any], JevResponse, dict[str, _Any] | None] | None:
    try:
        payload = response.json()
        normalized = _normalize_gateway_response(payload, questions)
        parsed = _parse_judge_response(normalized, questions)
        usage = _normalize_usage(payload.get("usage"))
    except (TypeError, ValueError, KeyError):
        return None
    return payload, parsed, usage


def _require_gateway_key() -> str:
    api_key = _resolve_gateway_key()
    if not api_key:
        raise JevError("Vercel AI Gateway API key is not set")
    return api_key


def _validate_model(model: str) -> None:
    if not isinstance(model, str) or not model:
        raise JevError("model must be a non-empty string")


def _as_jev_error(response: _ErrorResponse) -> JevError:
    return JevError(
        response.error,
        http_status=response.http_status,
        attempts=response.attempts,
    )


def _request_payload(
    state: _State | _Mapping[str, _Any], questions: _Mapping[str, _Any]
) -> dict[str, _Any]:
    return build_canonical_request(state, questions).payload


def _gateway_questions(questions: _Mapping[str, _Any]) -> dict[str, _Any]:
    result: dict[str, _Any] = {}
    for question_id, question in questions.items():
        if isinstance(question, _Mapping) and question.get("type") == "noul":
            result[question_id] = {**question, "type": "boolean"}
        else:
            result[question_id] = question
    return result


def _normalize_gateway_response(
    payload: _Any, questions: _Mapping[str, _Any]
) -> _Mapping[str, _Any]:
    if not isinstance(payload, _Mapping):
        raise ValueError("response must be an object")
    answers = payload.get("answers")
    if not isinstance(answers, _Mapping):
        raise ValueError("response requires an answers object")
    normalized_answers: dict[str, _Any] = {}
    for question_id, raw_answer in answers.items():
        question = questions.get(question_id)
        question_type = question.get("type") if isinstance(question, _Mapping) else None
        if question_type == "noul":
            if not isinstance(raw_answer, _Mapping):
                raise ValueError("boolean answer must be an object")
            if set(raw_answer) != {"type", "probability"}:
                raise ValueError(
                    "boolean answer must contain exactly type and probability"
                )
            if raw_answer.get("type") != "boolean":
                raise ValueError("gateway noul answer must have boolean type")
            normalized_answers[question_id] = {
                "type": "noul",
                "noul": raw_answer["probability"],
            }
            continue
        if not isinstance(raw_answer, _Mapping):
            normalized_answers[question_id] = raw_answer
            continue
        if question_type == "score" and not raw_answer.get("legend"):
            raw_answer = {
                **raw_answer,
                "legend": _score_legend_for_question(question),
            }
        probabilities = raw_answer.get("probabilities")
        expected_probability_keys = set(_probability_keys_for_question(question))
        if expected_probability_keys and isinstance(probabilities, _Mapping):
            if set(probabilities) != expected_probability_keys:
                raise ValueError(
                    f"probabilities for {question_id} have an invalid shape"
                )
        normalized_answers[question_id] = raw_answer
    return {**payload, "answers": normalized_answers}


def _served_model(payload: _Mapping[str, _Any]) -> str | None:
    metadata = payload.get("providerMetadata")
    if not isinstance(metadata, _Mapping):
        return None
    gateway = metadata.get("gateway")
    if not isinstance(gateway, _Mapping):
        return None
    routing = gateway.get("routing")
    if isinstance(routing, _Mapping):
        canonical_slug = routing.get("canonicalSlug")
        if isinstance(canonical_slug, str) and canonical_slug:
            return canonical_slug
    attempts = gateway.get("modelAttempts")
    if not isinstance(attempts, (list, tuple)):
        return None
    for attempt in reversed(attempts):
        if not isinstance(attempt, _Mapping) or not _successful_attempt(attempt):
            continue
        canonical_slug = attempt.get("canonicalSlug")
        if isinstance(canonical_slug, str) and canonical_slug:
            return canonical_slug
    return None


def _successful_attempt(attempt: _Mapping[str, _Any]) -> bool:
    for key in ("success", "successful", "isSuccessful"):
        if attempt.get(key) is False:
            return False
    status = attempt.get("status")
    if isinstance(status, str) and status.lower() not in {
        "success",
        "succeeded",
        "complete",
        "completed",
        "ok",
    }:
        return False
    for key in ("statusCode", "status_code"):
        status_code = attempt.get(key)
        if isinstance(status_code, int) and not 200 <= status_code < 300:
            return False
    return True


async def evaluate_async(
    state: _Mapping[str, _Any],
    questions: _Mapping[str, _Any],
    *,
    model: str = _GATEWAY_MODEL,
) -> JevResponse:
    """Evaluate one request through a short-lived async client."""

    client = JevClient()
    try:
        return await client.evaluate_async(state, questions, model=model)
    finally:
        await client.aclose()


def make_judge() -> tuple[_Callable[..., _Any], _Callable[[], None]]:
    """Return a callable client and its close hook for legacy batch adapters."""

    client = JevClient()
    return client, client.close


def runtime_preset(questions: _Mapping[str, _Any], *, name: str = "harness") -> _Any:
    """Build a validated in-process preset for an arbitrary question battery."""

    from .presets import Preset

    data = {
        "schema": "jm.preset/v1",
        "name": name,
        "version": "1",
        "model": _GATEWAY_MODEL,
        "chunking": {
            "by": "file",
            "max_chunks": 512,
            "limits": {
                "focus_bytes": 65_536,
                "context_field_bytes": 65_536,
                "state_bytes": 131_072,
            },
        },
        "compatible_chunkers": ["file"],
        "questions": dict(questions),
        "thresholds": {},
        "output": {
            "default_format": "jsonl",
            "pretty_template": "{state_ref}",
            "fields": ["record_type", "state_ref", "answers", "meta"],
        },
    }
    return Preset(data, _Path("<runtime>"))


def _normalize_usage(usage: _Any) -> dict[str, _Any] | None:
    if usage is None:
        return None
    if not isinstance(usage, _Mapping):
        raise ValueError("usage must be an object")
    known = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "totalTokens": "total_tokens",
        "reasoningTokens": "reasoning_tokens",
        "cachedInputTokens": "cached_input_tokens",
        "cacheReadInputTokens": "cache_read_input_tokens",
        "cacheWriteInputTokens": "cache_write_input_tokens",
    }
    return {known.get(key, key): value for key, value in usage.items()}


__all__ = [
    "ConfigurationError",
    "CacheStore",
    "EmitResult",
    "FormationEvent",
    "FormationReport",
    "InputError",
    "InputSidecar",
    "JevClient",
    "JevError",
    "JevResponse",
    "ResultFilter",
    "State",
    "emit",
    "evaluate_async",
    "judge",
    "judge_async",
    "make_judge",
    "runtime_preset",
]
