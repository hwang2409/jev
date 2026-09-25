"""Browser action safety classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from ._types import BrowserRiskEvidence

_BROWSER_ACTIONS = frozenset({"click", "navigate", "select", "submit", "type"})
_BROWSER_DESTRUCTIVE_WORDS = re.compile(
    r"\b(?:cancel|delete|destroy|discard|remove|revoke|terminate|unsubscribe)\b",
    re.IGNORECASE,
)
_BROWSER_PAYMENT_WORDS = re.compile(
    r"\b(?:buy|checkout|donate|pay|payment|purchase|subscribe|transfer)\b",
    re.IGNORECASE,
)
_BROWSER_AUTH_WORDS = re.compile(
    r"\b(?:account|authenticate|authentication|login|log[ -]?in|password|permission|sign[ -]?in|token)\b",
    re.IGNORECASE,
)


def _url_origin(value: str) -> str | None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    host = parsed.hostname.casefold()
    if (
        port is None
        or (parsed.scheme == "http" and port == 80)
        or (parsed.scheme == "https" and port == 443)
    ):
        return f"{parsed.scheme.casefold()}://{host}"
    return f"{parsed.scheme.casefold()}://{host}:{port}"


def normalize_origin(value: str) -> str | None:
    """Normalize one HTTP origin with its effective port."""

    return _url_origin(value)


@dataclass(frozen=True, slots=True)
class BrowserOriginPolicy:
    """Allow only configured origins before browser actions reach the adapter."""

    allowed_origins: frozenset[str] = frozenset()

    @classmethod
    def from_values(cls, origins: object) -> BrowserOriginPolicy:
        if not isinstance(origins, (tuple, list, set, frozenset)):
            return cls()
        normalized = {
            origin
            for value in origins
            if isinstance(value, str)
            for origin in [normalize_origin(value)]
            if origin is not None
        }
        return cls(frozenset(normalized))

    def allows(self, value: str | None) -> bool:
        origin = normalize_origin(value or "")
        return origin is not None and origin in self.allowed_origins


def _browser_layer0_reason(evidence: BrowserRiskEvidence) -> str | None:
    _classification, reason = _browser_layer0_classify(evidence)
    return reason


def _browser_layer0_classify(
    evidence: BrowserRiskEvidence,
) -> tuple[str, str | None]:
    """Classify browser evidence before calling the shared Jev scorer."""

    if evidence.action not in _BROWSER_ACTIONS:
        return "escalate", "unclassifiable_action"
    if not evidence.role or _url_origin(evidence.current_origin) is None:
        return "escalate", "unclassifiable_origin"
    if evidence.target_url is not None:
        target_origin = _url_origin(evidence.target_url)
        if target_origin is None:
            return "escalate", "unclassifiable_target_url"
        if (
            target_origin != _url_origin(evidence.current_origin)
            and not evidence.origin_allowed
        ):
            return "escalate", "external_origin"
    if evidence.form_action_origin is not None:
        form_origin = _url_origin(evidence.form_action_origin)
        if form_origin is None:
            return "escalate", "unclassifiable_form_action_origin"
        if (
            form_origin != _url_origin(evidence.current_origin)
            and not evidence.origin_allowed
        ):
            return "escalate", "external_form_action_origin"
    if evidence.payment_language or _BROWSER_PAYMENT_WORDS.search(evidence.text):
        return "analyzable", "payment_or_financial_commitment"
    if evidence.authentication_language or _BROWSER_AUTH_WORDS.search(evidence.text):
        return "analyzable", "authentication_or_permission_change"
    if evidence.download:
        return "analyzable", "download"
    if evidence.durable_state_change:
        return "analyzable", "durable_state_change"
    if evidence.action == "click" and _BROWSER_DESTRUCTIVE_WORDS.search(evidence.text):
        return "analyzable", "destructive_action"
    return "analyzable", None


def browser_action_requires_safety(evidence: BrowserRiskEvidence) -> bool:
    return _browser_layer0_reason(evidence) is not None
