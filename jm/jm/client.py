from __future__ import annotations

import time as _time
from collections.abc import Mapping as _Mapping
from dataclasses import replace as _replace
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from ._transport import (
    _GATEWAY_MODEL,
    _GatewayTransport,
    _resolve_gateway_key,
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


class JevClient:
    """Synchronous and asynchronous client for the Jev evaluation service."""

    def __init__(self, *, _transport: _GatewayTransport | None = None) -> None:
        self._transport = _transport or _GatewayTransport()

    def evaluate(
        self, state: _State | _Mapping[str, _Any], questions: _Mapping[str, _Any]
    ) -> JevResponse:
        api_key = _require_gateway_key()
        payload = _request_payload(state, questions)
        started = _time.monotonic()
        response, attempts = self._transport.post(payload, api_key)
        if isinstance(response, _ErrorResponse):
            raise _as_jev_error(response)

        parsed = self._parse_response(response, questions, started, attempts)
        if parsed.complete or attempts >= self._transport.max_attempts:
            return parsed

        response, retry_attempts = self._transport.post(payload, api_key, attempts)
        if isinstance(response, _ErrorResponse):
            raise _as_jev_error(response)
        return self._parse_response(response, questions, started, retry_attempts)

    async def evaluate_async(
        self, state: _State | _Mapping[str, _Any], questions: _Mapping[str, _Any]
    ) -> JevResponse:
        api_key = _require_gateway_key()
        payload = _request_payload(state, questions)
        started = _time.monotonic()
        response, attempts = await self._transport.apost(payload, api_key)
        if isinstance(response, _ErrorResponse):
            raise _as_jev_error(response)

        parsed = self._parse_response(response, questions, started, attempts)
        if parsed.complete or attempts >= self._transport.max_attempts:
            return parsed

        response, retry_attempts = await self._transport.apost(
            payload, api_key, attempts
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
        if model != _GATEWAY_MODEL:
            return _ErrorResponse(f"model must be {_GATEWAY_MODEL}")
        try:
            return self.evaluate(state, questions)
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
        try:
            payload = response.json()
            normalized = _normalize_gateway_response(payload, questions)
            parsed = _parse_judge_response(normalized, questions)
            usage = _normalize_usage(payload.get("usage"))
        except (TypeError, ValueError, KeyError) as exc:
            raise JevError(
                "malformed answer",
                http_status=response.status_code,
                attempts=attempts,
            ) from exc
        return _replace(
            parsed,
            served_model=_served_model(payload),
            usage=usage,
            latency_ms=round((_time.monotonic() - started) * 1000),
        )


def _require_gateway_key() -> str:
    api_key = _resolve_gateway_key()
    if not api_key:
        raise JevError("Vercel AI Gateway API key is not set")
    return api_key


def _as_jev_error(response: _ErrorResponse) -> JevError:
    return JevError(
        response.error,
        http_status=response.http_status,
        attempts=response.attempts,
    )


def _request_payload(
    state: _State | _Mapping[str, _Any], questions: _Mapping[str, _Any]
) -> dict[str, _Any]:
    if isinstance(state, _Mapping):
        state_payload = dict(state)
    else:
        state_payload = state.api_payload
    return {
        "providerOptions": {"gateway": {"zeroDataRetention": True}},
        "state": state_payload,
        "questions": _gateway_questions(questions),
    }


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
        question_type = (
            question.get("type") if isinstance(question, _Mapping) else None
        )
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
    for provider in ("typesafe", "gateway"):
        provider_data = metadata.get(provider)
        if not isinstance(provider_data, _Mapping):
            continue
        model = provider_data.get("model")
        if isinstance(model, str) and model:
            return model
    return None


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


__all__ = ["JevClient", "JevError", "JevResponse"]
