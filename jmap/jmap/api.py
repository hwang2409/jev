from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Callable, Mapping
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import httpx

from .answers import ErrorResponse, JudgeResponse, parse_judge_response

if TYPE_CHECKING:
    from .runner import State

SYSTEMONE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MAX_ATTEMPTS = 3


class TypeSafeClient:
    """Synchronous TypeSafe SystemOne client for the judge function seam."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        timeout: float = 30.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] | None = None,
        jitter: Callable[[], float] | None = None,
        backoff_base: float = 1.0,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if backoff_base < 0:
            raise ValueError("backoff_base must not be negative")
        self.http_client = http_client or httpx.Client(timeout=timeout)
        self._owns_http_client = http_client is None
        self.max_attempts = max_attempts
        self.sleep = sleep or time.sleep
        self.jitter = jitter or (lambda: random.uniform(0.0, backoff_base))
        self.backoff_base = backoff_base

    def __call__(
        self, state: State, questions: Mapping[str, Any], model: str
    ) -> JudgeResponse | ErrorResponse:
        if model == "jev-latest":
            return ErrorResponse("model must be a pinned version")
        api_key = os.environ.get("JEV_API_KEY")
        if not api_key:
            return ErrorResponse("JEV_API_KEY is not set")

        payload = {
            "state": state.api_payload,
            "model": model,
            "questions": dict(questions),
        }
        response, attempts = self._post(payload, api_key)
        if isinstance(response, ErrorResponse):
            return response

        try:
            parsed = parse_judge_response(response.json(), questions)
        except (TypeError, ValueError):
            return ErrorResponse(
                "malformed answer",
                http_status=response.status_code,
                attempts=attempts,
            )
        if parsed.complete:
            return parsed

        response, retry_attempts = self._post(payload, api_key)
        if isinstance(response, ErrorResponse):
            return response
        try:
            return parse_judge_response(response.json(), questions)
        except (TypeError, ValueError):
            return ErrorResponse(
                "malformed answer",
                http_status=response.status_code,
                attempts=retry_attempts,
            )

    def close(self) -> None:
        if self._owns_http_client:
            self.http_client.close()

    def _post(
        self, payload: Mapping[str, Any], api_key: str
    ) -> tuple[httpx.Response | ErrorResponse, int]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                response = self.http_client.post(
                    SYSTEMONE_URL, headers=headers, json=payload
                )
            except httpx.TimeoutException:
                if attempts < self.max_attempts:
                    self._wait(attempts, None)
                    continue
                return ErrorResponse("request timed out", attempts=attempts), attempts
            except httpx.RequestError:
                return ErrorResponse("request failed", attempts=attempts), attempts

            if response.status_code in {429, 529}:
                if attempts < self.max_attempts:
                    self._wait(attempts, _retry_after(response))
                    continue
                return self._status_error(response, attempts), attempts
            if response.is_success:
                return response, attempts
            return self._status_error(response, attempts), attempts

        raise AssertionError("retry loop exited without a response")

    def _wait(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is None:
            delay = self.backoff_base * (2 ** (attempt - 1)) + max(0.0, self.jitter())
        else:
            delay = retry_after
        self.sleep(delay)

    @staticmethod
    def _status_error(response: httpx.Response, attempts: int) -> ErrorResponse:
        return ErrorResponse(
            f"request failed with HTTP status {response.status_code}",
            http_status=response.status_code,
            attempts=attempts,
        )


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


JevClient = TypeSafeClient
SystemOneClient = TypeSafeClient
