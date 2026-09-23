"""Compatibility names for the pre-v2 jm api module."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from ._transport import (
    _DEFAULT_MAX_ATTEMPTS,
    _GATEWAY_ENDPOINT,
    _GATEWAY_HEADERS,
    _GATEWAY_KEY_NAMES,
    _GATEWAY_MODEL,
    _MAX_RESPONSE_BYTES,
    _MAX_WAIT_SECONDS,
    _GatewayTransport,
    _resolve_gateway_key,
)
from .client import JevClient, JevError, JevResponse, _normalize_usage

DEFAULT_MAX_ATTEMPTS = _DEFAULT_MAX_ATTEMPTS
GATEWAY_ENDPOINT = _GATEWAY_ENDPOINT
GATEWAY_HEADERS = _GATEWAY_HEADERS
GATEWAY_KEY_NAMES = _GATEWAY_KEY_NAMES
GATEWAY_MODEL = _GATEWAY_MODEL
MAX_RESPONSE_BYTES = _MAX_RESPONSE_BYTES
MAX_WAIT_SECONDS = _MAX_WAIT_SECONDS
normalize_usage = _normalize_usage
resolve_gateway_key = _resolve_gateway_key


class GatewayClient(JevClient):
    """Compatibility adapter for the pre-v2 synchronous client."""

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
        super().__init__(
            _transport=_GatewayTransport(
                http_client=http_client,
                timeout=timeout,
                max_attempts=max_attempts,
                max_response_bytes=max_response_bytes,
                sleep=sleep,
                jitter=jitter,
                backoff_base=backoff_base,
            )
        )


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "GATEWAY_ENDPOINT",
    "GATEWAY_HEADERS",
    "GATEWAY_KEY_NAMES",
    "GATEWAY_MODEL",
    "GatewayClient",
    "JevClient",
    "JevError",
    "JevResponse",
    "MAX_RESPONSE_BYTES",
    "MAX_WAIT_SECONDS",
    "normalize_usage",
    "resolve_gateway_key",
]
