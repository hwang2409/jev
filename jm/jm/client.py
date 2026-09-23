from __future__ import annotations

import math
import os
import random
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .answers import (
    ErrorResponse,
    JudgeResponse,
    parse_judge_response,
    probability_keys_for_question,
    score_legend_for_question,
)

if TYPE_CHECKING:
    from .runner import State

GATEWAY_ENDPOINT = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
GATEWAY_MODEL = "typesafe-ai/jev"
GATEWAY_KEY_NAMES = ("VERCEL_AI_GATEWAY", "AI_GATEWAY_API_KEY", "VERCEL_JEV_KEY")
GATEWAY_HEADERS = {
    "Content-Type": "application/json",
    "Accept-Encoding": "identity",
    "ai-evaluation-model-specification-version": "4",
    "ai-gateway-auth-method": "api-key",
    "ai-gateway-protocol-version": "0.0.1",
    "ai-model-id": GATEWAY_MODEL,
}
DEFAULT_MAX_ATTEMPTS = 3
MAX_WAIT_SECONDS = 300.0
MAX_RESPONSE_BYTES = 1_048_576


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


JevResponse = JudgeResponse


class JevClient:
    """Synchronous and asynchronous Vercel AI Gateway client."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        sleep: Callable[[float], None] | None = None,
        async_sleep: Callable[[float], Any] | None = None,
        jitter: Callable[[], float] | None = None,
        backoff_base: float = 1.0,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if backoff_base < 0:
            raise ValueError("backoff_base must not be negative")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.http_client = http_client or httpx.Client(timeout=timeout)
        self.async_http_client = async_http_client
        self._owns_http_client = http_client is None
        self._owns_async_http_client = async_http_client is None
        self.max_attempts = min(max_attempts, DEFAULT_MAX_ATTEMPTS)
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.sleep = sleep or time.sleep
        self.async_sleep = async_sleep
        self.jitter = jitter or (lambda: random.uniform(0.0, backoff_base))
        self.backoff_base = backoff_base

    def evaluate(
        self, state: State | Mapping[str, Any], questions: Mapping[str, Any]
    ) -> JevResponse:
        api_key = _require_gateway_key()
        payload = _request_payload(state, questions)
        started = time.monotonic()
        response, attempts = self._post(payload, api_key)
        if isinstance(response, ErrorResponse):
            raise _as_jev_error(response)

        parsed = self._parse_response(response, questions, started, attempts)
        if parsed.complete or attempts >= self.max_attempts:
            return parsed

        response, retry_attempts = self._post(payload, api_key, attempts)
        if isinstance(response, ErrorResponse):
            raise _as_jev_error(response)
        return self._parse_response(response, questions, started, retry_attempts)

    async def evaluate_async(
        self, state: State | Mapping[str, Any], questions: Mapping[str, Any]
    ) -> JevResponse:
        if self.async_http_client is None:
            self.async_http_client = httpx.AsyncClient(timeout=self.timeout)
        api_key = _require_gateway_key()
        payload = _request_payload(state, questions)
        started = time.monotonic()
        response, attempts = await self._apost(payload, api_key)
        if isinstance(response, ErrorResponse):
            raise _as_jev_error(response)

        parsed = self._parse_response(response, questions, started, attempts)
        if parsed.complete or attempts >= self.max_attempts:
            return parsed

        response, retry_attempts = await self._apost(payload, api_key, attempts)
        if isinstance(response, ErrorResponse):
            raise _as_jev_error(response)
        return self._parse_response(response, questions, started, retry_attempts)

    def __call__(
        self,
        state: State | Mapping[str, Any],
        questions: Mapping[str, Any],
        model: str = GATEWAY_MODEL,
    ) -> JevResponse | ErrorResponse:
        if model != GATEWAY_MODEL:
            return ErrorResponse(f"model must be {GATEWAY_MODEL}")
        try:
            return self.evaluate(state, questions)
        except JevError as exc:
            return ErrorResponse(
                exc.message,
                http_status=exc.http_status,
                attempts=exc.attempts,
            )

    def close(self) -> None:
        if self._owns_http_client:
            self.http_client.close()

    async def aclose(self) -> None:
        if self._owns_async_http_client and self.async_http_client is not None:
            await self.async_http_client.aclose()
        if self._owns_http_client:
            self.http_client.close()

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
        response: httpx.Response,
        questions: Mapping[str, Any],
        started: float,
        attempts: int,
    ) -> JevResponse:
        try:
            payload = response.json()
            normalized = _normalize_gateway_response(payload, questions)
            parsed = parse_judge_response(normalized, questions)
            usage = normalize_usage(payload.get("usage"))
        except (TypeError, ValueError, KeyError) as exc:
            raise JevError(
                "malformed answer",
                http_status=response.status_code,
                attempts=attempts,
            ) from exc
        return replace(
            parsed,
            served_model=_served_model(payload),
            usage=usage,
            latency_ms=round((time.monotonic() - started) * 1000),
        )

    def _post(
        self,
        payload: Mapping[str, Any],
        api_key: str,
        attempts_used: int = 0,
    ) -> tuple[httpx.Response | ErrorResponse, int]:
        headers = {"Authorization": f"Bearer {api_key}", **GATEWAY_HEADERS}
        attempts = attempts_used
        if attempts >= self.max_attempts:
            return _attempt_budget_error(attempts), attempts
        while attempts < self.max_attempts:
            attempts += 1
            try:
                with self.http_client.stream(
                    "POST",
                    GATEWAY_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                ) as response:
                    if response.status_code in {429, 529}:
                        if attempts < self.max_attempts:
                            self._wait(attempts, _retry_after(response))
                            continue
                        return _status_error(response, attempts), attempts
                    if response.is_success:
                        content = _read_response(response, self.max_response_bytes)
                        if content is None:
                            return ErrorResponse(
                                "response too large",
                                http_status=response.status_code,
                                attempts=attempts,
                            ), attempts
                        return _buffered_response(response, content), attempts
                    return _status_error(response, attempts), attempts
            except httpx.TimeoutException:
                if attempts < self.max_attempts:
                    self._wait(attempts, None)
                    continue
                return ErrorResponse("request timed out", attempts=attempts), attempts
            except httpx.RequestError:
                return ErrorResponse("request failed", attempts=attempts), attempts
        raise AssertionError("retry loop exited without a response")

    async def _apost(
        self,
        payload: Mapping[str, Any],
        api_key: str,
        attempts_used: int = 0,
    ) -> tuple[httpx.Response | ErrorResponse, int]:
        headers = {"Authorization": f"Bearer {api_key}", **GATEWAY_HEADERS}
        attempts = attempts_used
        if self.async_http_client is None:
            raise RuntimeError("async http client is not initialized")
        if attempts >= self.max_attempts:
            return _attempt_budget_error(attempts), attempts
        while attempts < self.max_attempts:
            attempts += 1
            try:
                response = await self.async_http_client.post(
                    GATEWAY_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 529}:
                    if attempts < self.max_attempts:
                        await self._await_wait(attempts, _retry_after(response))
                        continue
                    return _status_error(response, attempts), attempts
                if response.is_success:
                    content = await response.aread()
                    if len(content) > self.max_response_bytes:
                        return ErrorResponse(
                            "response too large",
                            http_status=response.status_code,
                            attempts=attempts,
                        ), attempts
                    return _buffered_response(response, content), attempts
                return _status_error(response, attempts), attempts
            except httpx.TimeoutException:
                if attempts < self.max_attempts:
                    await self._await_wait(attempts, None)
                    continue
                return ErrorResponse("request timed out", attempts=attempts), attempts
            except httpx.RequestError:
                return ErrorResponse("request failed", attempts=attempts), attempts
        raise AssertionError("retry loop exited without a response")

    def _wait(self, attempt: int, retry_after: float | None) -> None:
        self.sleep(_wait_seconds(self, attempt, retry_after))

    async def _await_wait(self, attempt: int, retry_after: float | None) -> None:
        delay = _wait_seconds(self, attempt, retry_after)
        if self.async_sleep is not None:
            await self.async_sleep(delay)
        else:
            await _async_sleep(delay)


async def _async_sleep(delay: float) -> None:
    import asyncio

    await asyncio.sleep(delay)


def resolve_gateway_key() -> str | None:
    """Resolve the gateway key from the environment, then local shell config."""
    for name in GATEWAY_KEY_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    try:
        zshrc = (Path.home() / ".zshrc").read_text(encoding="utf-8")
    except OSError:
        return None
    for name in GATEWAY_KEY_NAMES:
        match = re.search(
            rf"^\s*(?:export\s+)?{name}\s*=\s*[\"']?([^\"'\s#]+)",
            zshrc,
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    return None


def normalize_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if not isinstance(usage, Mapping):
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


def _require_gateway_key() -> str:
    api_key = resolve_gateway_key()
    if not api_key:
        raise JevError("Vercel AI Gateway API key is not set")
    return api_key


def _as_jev_error(response: ErrorResponse) -> JevError:
    return JevError(
        response.error,
        http_status=response.http_status,
        attempts=response.attempts,
    )


def _attempt_budget_error(attempts: int) -> ErrorResponse:
    return ErrorResponse("request attempt budget exhausted", attempts=attempts)


def _request_payload(state: State, questions: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(state, Mapping):
        state_payload = dict(state)
    else:
        state_payload = state.api_payload
    return {
        "providerOptions": {"gateway": {"zeroDataRetention": True}},
        "state": state_payload,
        "questions": _gateway_questions(questions),
    }


def _gateway_questions(questions: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for question_id, question in questions.items():
        if isinstance(question, Mapping) and question.get("type") == "noul":
            result[question_id] = {**question, "type": "boolean"}
        else:
            result[question_id] = question
    return result


def _normalize_gateway_response(
    payload: Any, questions: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("response must be an object")
    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        raise ValueError("response requires an answers object")
    normalized_answers = {}
    for question_id, raw_answer in answers.items():
        question = questions.get(question_id)
        question_type = (
            question.get("type") if isinstance(question, Mapping) else None
        )
        if question_type == "noul":
            if not isinstance(raw_answer, Mapping):
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
        else:
            if not isinstance(raw_answer, Mapping):
                normalized_answers[question_id] = raw_answer
                continue
            if question_type == "score" and not raw_answer.get("legend"):
                raw_answer = {
                    **raw_answer,
                    "legend": score_legend_for_question(question),
                }
            probability_keys = probability_keys_for_question(question)
            probabilities = raw_answer.get("probabilities")
            if probability_keys and isinstance(probabilities, Mapping):
                raw_answer = {
                    **raw_answer,
                    "probabilities": {
                        **probabilities,
                        **{
                            key: probabilities.get(key, 0.0)
                            for key in probability_keys
                            if key not in probabilities
                        },
                    },
                }
            normalized_answers[question_id] = raw_answer
    return {**payload, "answers": normalized_answers}


def _served_model(payload: Mapping[str, Any]) -> str | None:
    metadata = payload.get("providerMetadata")
    if not isinstance(metadata, Mapping):
        return None
    for provider in ("typesafe", "gateway"):
        provider_data = metadata.get(provider)
        if not isinstance(provider_data, Mapping):
            continue
        model = provider_data.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = retry_at.timestamp() - time.time()
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _wait_seconds(
    client: JevClient, attempt: int, retry_after: float | None
) -> float:
    if retry_after is None:
        base = min(MAX_WAIT_SECONDS, max(0.0, client.backoff_base))
        jitter = min(MAX_WAIT_SECONDS, max(0.0, client.jitter()))
        delay = base * (2 ** (attempt - 1)) + jitter
    else:
        delay = retry_after
    return min(MAX_WAIT_SECONDS, max(0.0, delay))


def _read_response(response: httpx.Response, max_response_bytes: int) -> bytes | None:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > max_response_bytes:
                return None
        except ValueError:
            pass
    content = bytearray()
    for chunk in response.iter_bytes():
        content.extend(chunk)
        if len(content) > max_response_bytes:
            return None
    return bytes(content)


def _buffered_response(response: httpx.Response, content: bytes) -> httpx.Response:
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        content=content,
        request=response.request,
    )


def _status_error(response: httpx.Response, attempts: int) -> ErrorResponse:
    return ErrorResponse(
        f"request failed with HTTP status {response.status_code}",
        http_status=response.status_code,
        attempts=attempts,
    )
