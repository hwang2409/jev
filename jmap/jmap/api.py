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

from .answers import ErrorResponse, JudgeResponse, parse_judge_response

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
# Keep retry delays bounded so server hints and injected jitter cannot hang a run.
MAX_WAIT_SECONDS = 30.0
# Bound successful response bodies before parsing them into memory.
MAX_RESPONSE_BYTES = 1_048_576


class GatewayClient:
    """Synchronous Vercel AI Gateway client for the judge function seam."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        timeout: float = 30.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        sleep: Callable[[float], None] | None = None,
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
        self._owns_http_client = http_client is None
        self.max_attempts = min(max_attempts, DEFAULT_MAX_ATTEMPTS)
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.sleep = sleep or time.sleep
        self.jitter = jitter or (lambda: random.uniform(0.0, backoff_base))
        self.backoff_base = backoff_base

    def __call__(
        self, state: State, questions: Mapping[str, Any], model: str
    ) -> JudgeResponse | ErrorResponse:
        if model != GATEWAY_MODEL:
            return ErrorResponse(f"model must be {GATEWAY_MODEL}")
        api_key = resolve_gateway_key()
        if not api_key:
            return ErrorResponse("Vercel AI Gateway API key is not set")

        payload = {
            "providerOptions": {"gateway": {"zeroDataRetention": True}},
            "state": state.api_payload,
            "questions": _gateway_questions(questions),
        }
        started = time.monotonic()
        response, attempts = self._post(payload, api_key)
        if isinstance(response, ErrorResponse):
            return response

        parsed = self._parse_response(response, questions, started)
        if parsed is None:
            return ErrorResponse(
                "malformed answer",
                http_status=response.status_code,
                attempts=attempts,
            )
        if parsed.complete:
            return parsed

        response, retry_attempts = self._post(payload, api_key, attempts)
        if isinstance(response, ErrorResponse):
            return response
        parsed = self._parse_response(response, questions, started)
        if parsed is not None:
            return parsed
        return ErrorResponse(
            "malformed answer",
            http_status=response.status_code,
            attempts=retry_attempts,
        )

    @staticmethod
    def _parse_response(
        response: httpx.Response,
        questions: Mapping[str, Any],
        started: float,
    ) -> JudgeResponse | None:
        try:
            payload = response.json()
            normalized = _normalize_gateway_response(payload, questions)
            parsed = parse_judge_response(normalized, questions)
        except (TypeError, ValueError):
            return None
        return replace(
            parsed,
            served_model=_served_model(payload),
            usage=payload.get("usage"),
            latency_ms=round((time.monotonic() - started) * 1000),
        )

    def close(self) -> None:
        if self._owns_http_client:
            self.http_client.close()

    def _post(
        self, payload: Mapping[str, Any], api_key: str, attempts_used: int = 0
    ) -> tuple[httpx.Response | ErrorResponse, int]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            **GATEWAY_HEADERS,
        }
        attempts = attempts_used
        if attempts >= self.max_attempts:
            return (
                ErrorResponse("request attempt budget exhausted", attempts=attempts),
                attempts,
            )
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
                        return self._status_error(response, attempts), attempts
                    if response.is_success:
                        content = self._read_response(response)
                        if content is None:
                            return ErrorResponse(
                                "response too large",
                                http_status=response.status_code,
                                attempts=attempts,
                            ), attempts
                        return (
                            httpx.Response(
                                response.status_code,
                                headers=response.headers,
                                content=content,
                                request=response.request,
                            ),
                            attempts,
                        )
                    return self._status_error(response, attempts), attempts
            except httpx.TimeoutException:
                if attempts < self.max_attempts:
                    self._wait(attempts, None)
                    continue
                return ErrorResponse("request timed out", attempts=attempts), attempts
            except httpx.RequestError:
                return ErrorResponse("request failed", attempts=attempts), attempts

        raise AssertionError("retry loop exited without a response")

    def _read_response(self, response: httpx.Response) -> bytes | None:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_response_bytes:
                    return None
            except ValueError:
                pass

        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > self.max_response_bytes:
                return None
        return bytes(content)

    def _wait(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is None:
            base = min(MAX_WAIT_SECONDS, max(0.0, self.backoff_base))
            jitter = min(MAX_WAIT_SECONDS, max(0.0, self.jitter()))
            delay = base * (2 ** (attempt - 1)) + jitter
        else:
            delay = retry_after
        self.sleep(min(MAX_WAIT_SECONDS, max(0.0, delay)))

    @staticmethod
    def _status_error(response: httpx.Response, attempts: int) -> ErrorResponse:
        return ErrorResponse(
            f"request failed with HTTP status {response.status_code}",
            http_status=response.status_code,
            attempts=attempts,
        )


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
            rf"^\s*(?:export\s+)?{name}=[\"']?([^\"'\s#]+)",
            zshrc,
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    return None


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
            if raw_answer.get("type") == "noul":
                normalized_answers[question_id] = raw_answer
                continue
            if raw_answer.get("type") != "boolean":
                raise ValueError("gateway noul answer must have boolean type")
            normalized_answers[question_id] = {
                "type": "noul",
                "noul": raw_answer.get("probability"),
            }
        else:
            normalized_answers[question_id] = raw_answer
    return {**payload, "answers": normalized_answers}


def _served_model(payload: Mapping[str, Any]) -> str | None:
    metadata = payload.get("providerMetadata")
    if not isinstance(metadata, Mapping):
        return None
    typesafe = metadata.get("typesafe")
    if not isinstance(typesafe, Mapping):
        return None
    model = typesafe.get("model")
    return model if isinstance(model, str) and model else None


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


JevClient = GatewayClient
