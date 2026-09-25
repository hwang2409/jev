"""Shared safety outcomes and policy constants."""

from __future__ import annotations

from dataclasses import dataclass

SAFE_MAX = 1.0
SAFETY_CONFIDENCE = 0.8
"""Minimum call confidence for Jev safety gating."""
SAFETY_OUTSIDE_CWD_THRESHOLD = 0.5
"""Maximum outside-cwd probability for an allowed command."""
SAFETY_IRREVERSIBLE_THRESHOLD = 0.5
"""Maximum irreversible probability for an allowed command."""
SHELL_TOOLS = frozenset({"bash", "exec", "run_background"})


@dataclass(frozen=True, slots=True)
class SafetyOutcome:
    decision: str
    layer: str
    score: int | None = None
    confidence: float | None = None
    reason: str | None = None
    usage: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class BrowserRiskEvidence:
    action: str
    role: str
    text: str
    current_origin: str
    target_url: str | None
    form_action_origin: str | None
    payment_language: bool
    authentication_language: bool
    download: bool
    durable_state_change: bool
    origin_allowed: bool = False
    operation_token: int | None = None


__all__ = [
    "SAFETY_CONFIDENCE",
    "SAFETY_IRREVERSIBLE_THRESHOLD",
    "SAFETY_OUTSIDE_CWD_THRESHOLD",
    "SAFE_MAX",
    "SHELL_TOOLS",
    "BrowserRiskEvidence",
    "SafetyOutcome",
]
