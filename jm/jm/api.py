"""Compatibility names for the pre-v2 jm api module."""

from .client import (
    DEFAULT_MAX_ATTEMPTS,
    GATEWAY_ENDPOINT,
    GATEWAY_HEADERS,
    GATEWAY_KEY_NAMES,
    GATEWAY_MODEL,
    MAX_RESPONSE_BYTES,
    MAX_WAIT_SECONDS,
    JevClient,
    JevError,
    JevResponse,
    normalize_usage,
    resolve_gateway_key,
)

GatewayClient = JevClient

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
