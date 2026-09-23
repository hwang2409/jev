from __future__ import annotations

import math as _math
import os as _os
import random as _random
import re as _re
import time as _time
from collections.abc import Callable as _Callable
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass
from datetime import UTC as _UTC
from email.utils import parsedate_to_datetime as _parsedate_to_datetime
from pathlib import Path as _Path
from typing import Any as _Any

import httpx as _httpx

from .answers import ErrorResponse as _ErrorResponse

_GATEWAY_ENDPOINT = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
_GATEWAY_MODEL = "typesafe-ai/jev"
_GATEWAY_KEY_NAMES = (
    "VERCEL_AI_GATEWAY",
    "AI_GATEWAY_API_KEY",
    "VERCEL_JEV_KEY",
)
_GATEWAY_HEADERS = {
    "Content-Type": "application/json",
    "Accept-Encoding": "identity",
    "ai-evaluation-model-specification-version": "4",
    "ai-gateway-auth-method": "api-key",
    "ai-gateway-protocol-version": "0.0.1",
    "ai-model-id": _GATEWAY_MODEL,
}
_DEFAULT_MAX_ATTEMPTS = 3
_MAX_WAIT_SECONDS = 300.0
_MAX_RESPONSE_BYTES = 1_048_576


@_dataclass(frozen=True, slots=True)
class _TransportResponse:
    status_code: int
    content: bytes

    def json(self) -> _Any:
        return _httpx.Response(self.status_code, content=self.content).json()


class _GatewayTransport:
    def __init__(
        self,
        *,
        http_client: _httpx.Client | None = None,
        async_http_client: _httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        max_response_bytes: int = _MAX_RESPONSE_BYTES,
        sleep: _Callable[[float], None] | None = None,
        async_sleep: _Callable[[_Any], _Any] | None = None,
        jitter: _Callable[[], float] | None = None,
        backoff_base: float = 1.0,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if backoff_base < 0:
            raise ValueError("backoff_base must not be negative")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self._http_client = http_client or _httpx.Client(timeout=timeout)
        self._async_http_client = async_http_client
        self._owns_http_client = http_client is None
        self._owns_async_http_client = async_http_client is None
        self.max_attempts = min(max_attempts, _DEFAULT_MAX_ATTEMPTS)
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.sleep = sleep or _time.sleep
        self.async_sleep = async_sleep
        self.jitter = jitter or (lambda: _random.uniform(0.0, backoff_base))
        self.backoff_base = backoff_base

    def post(
        self,
        payload: _Mapping[str, _Any],
        api_key: str,
        attempts_used: int = 0,
    ) -> tuple[_TransportResponse | _ErrorResponse, int]:
        headers = {"Authorization": f"Bearer {api_key}", **_GATEWAY_HEADERS}
        attempts = attempts_used
        if attempts >= self.max_attempts:
            return _attempt_budget_error(attempts), attempts
        while attempts < self.max_attempts:
            attempts += 1
            try:
                with self._http_client.stream(
                    "POST",
                    _GATEWAY_ENDPOINT,
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
                            return _ErrorResponse(
                                "response too large",
                                http_status=response.status_code,
                                attempts=attempts,
                            ), attempts
                        return (
                            _TransportResponse(response.status_code, content),
                            attempts,
                        )
                    return _status_error(response, attempts), attempts
            except _httpx.TimeoutException:
                if attempts < self.max_attempts:
                    self._wait(attempts, None)
                    continue
                return _ErrorResponse("request timed out", attempts=attempts), attempts
            except _httpx.RequestError:
                return _ErrorResponse("request failed", attempts=attempts), attempts
        raise AssertionError("retry loop exited without a response")

    async def apost(
        self,
        payload: _Mapping[str, _Any],
        api_key: str,
        attempts_used: int = 0,
    ) -> tuple[_TransportResponse | _ErrorResponse, int]:
        headers = {"Authorization": f"Bearer {api_key}", **_GATEWAY_HEADERS}
        attempts = attempts_used
        if self._async_http_client is None:
            self._async_http_client = _httpx.AsyncClient(timeout=self.timeout)
        if attempts >= self.max_attempts:
            return _attempt_budget_error(attempts), attempts
        while attempts < self.max_attempts:
            attempts += 1
            try:
                response = await self._async_http_client.post(
                    _GATEWAY_ENDPOINT,
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
                        return _ErrorResponse(
                            "response too large",
                            http_status=response.status_code,
                            attempts=attempts,
                        ), attempts
                    return _TransportResponse(response.status_code, content), attempts
                return _status_error(response, attempts), attempts
            except _httpx.TimeoutException:
                if attempts < self.max_attempts:
                    await self._await_wait(attempts, None)
                    continue
                return _ErrorResponse("request timed out", attempts=attempts), attempts
            except _httpx.RequestError:
                return _ErrorResponse("request failed", attempts=attempts), attempts
        raise AssertionError("retry loop exited without a response")

    def _wait(self, attempt: int, retry_after: float | None) -> None:
        self.sleep(_wait_seconds(self, attempt, retry_after))

    async def _await_wait(self, attempt: int, retry_after: float | None) -> None:
        delay = _wait_seconds(self, attempt, retry_after)
        if self.async_sleep is not None:
            await self.async_sleep(delay)
        else:
            await _async_sleep(delay)

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    async def aclose(self) -> None:
        if self._owns_async_http_client and self._async_http_client is not None:
            await self._async_http_client.aclose()
        if self._owns_http_client:
            self._http_client.close()


async def _async_sleep(delay: float) -> None:
    import asyncio

    await asyncio.sleep(delay)


def _resolve_gateway_key() -> str | None:
    for name in _GATEWAY_KEY_NAMES:
        value = _os.environ.get(name)
        if value:
            return value
    try:
        zshrc = (_Path.home() / ".zshrc").read_text(encoding="utf-8")
    except OSError:
        return None
    for name in _GATEWAY_KEY_NAMES:
        match = _re.search(
            rf"^\s*(?:export\s+)?{name}\s*=\s*[\"']?([^\"'\s#]+)",
            zshrc,
            _re.MULTILINE,
        )
        if match:
            return match.group(1)
    return None


def _read_response(
    response: _httpx.Response, max_response_bytes: int
) -> bytes | None:
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


def _status_error(response: _httpx.Response, attempts: int) -> _ErrorResponse:
    return _ErrorResponse(
        f"request failed with HTTP status {response.status_code}",
        http_status=response.status_code,
        attempts=attempts,
    )


def _attempt_budget_error(attempts: int) -> _ErrorResponse:
    return _ErrorResponse("request attempt budget exhausted", attempts=attempts)


def _retry_after(response: _httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = _parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=_UTC)
        seconds = retry_at.timestamp() - _time.time()
    if not _math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _wait_seconds(
    transport: _GatewayTransport, attempt: int, retry_after: float | None
) -> float:
    if retry_after is None:
        base = min(_MAX_WAIT_SECONDS, max(0.0, transport.backoff_base))
        jitter = min(_MAX_WAIT_SECONDS, max(0.0, transport.jitter()))
        delay = base * (2 ** (attempt - 1)) + jitter
    else:
        delay = retry_after
    return min(_MAX_WAIT_SECONDS, max(0.0, delay))


__all__ = ()
