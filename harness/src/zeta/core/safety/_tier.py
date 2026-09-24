"""Jev-backed safety tier evaluation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from ...protocol import jev
from ._browser import _browser_layer0_classify
from ._layer0 import layer0_classify
from ._types import (
    SAFE_MAX,
    SAFETY_CONFIDENCE,
    SAFETY_IRREVERSIBLE_THRESHOLD,
    SAFETY_OUTSIDE_CWD_THRESHOLD,
    SHELL_TOOLS,
    BrowserRiskEvidence,
    SafetyOutcome,
)

_logger = logging.getLogger(__name__)


class SafetyTier:
    """Evaluate only yolo shell calls and preserve full decision telemetry."""

    def __init__(
        self,
        *,
        cwd: str | Path = ".",
        headless: bool = False,
        task_excerpt: str = "",
        telemetry: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.cwd = str(Path(cwd).expanduser().resolve())
        self.headless = headless
        self.task_excerpt = task_excerpt
        self.telemetry = telemetry

    def set_telemetry(
        self, telemetry: Callable[[dict[str, object]], None] | None
    ) -> None:
        self.telemetry = telemetry

    def set_headless(self, headless: bool) -> None:
        self.headless = headless

    def set_task_excerpt(self, task_excerpt: str) -> None:
        self.task_excerpt = task_excerpt

    def command_cwd(
        self,
        arguments: dict[str, object],
        *,
        base_cwd: str | Path | None = None,
    ) -> str:
        value = arguments.get("cwd")
        default_cwd = Path(base_cwd or self.cwd).expanduser()
        if isinstance(value, str) and value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = default_cwd / candidate
            return str(candidate.resolve())
        return str(default_cwd.resolve())

    def applies(self, tool_name: str) -> bool:
        return tool_name in SHELL_TOOLS

    async def evaluate(self, tool_name: str, command: str, cwd: str) -> SafetyOutcome:
        if not self.applies(tool_name):
            return SafetyOutcome("allow", "bypass", reason="tool_out_of_scope")
        classification, reason = layer0_classify(command, cwd)
        if classification != "analyzable":
            return self._finish(
                SafetyOutcome(
                    "deny" if self.headless else "ask",
                    "layer0",
                    reason=reason,
                )
            )
        return self._finish(await self._evaluate_jev(command, cwd))

    async def evaluate_browser_action(
        self, evidence: BrowserRiskEvidence
    ) -> SafetyOutcome:
        """Evaluate risky browser evidence through the shared safety policy."""

        layer0_classification, layer0_reason = _browser_layer0_classify(evidence)
        if layer0_classification != "analyzable":
            return self._finish(
                SafetyOutcome(
                    "deny" if self.headless else "ask",
                    "layer0",
                    reason=layer0_reason,
                )
            )
        command = json.dumps(asdict(evidence), sort_keys=True)
        outcome = await self._evaluate_jev(command, evidence.current_origin)
        return self._finish(outcome)

    async def _evaluate_jev(self, command: str, cwd: str) -> SafetyOutcome:
        try:
            result = await jev.safety_score(command, cwd, self.task_excerpt)
        except Exception as exc:  # noqa: BLE001 - safety must fail closed
            # This polarity is deliberate: unlike a routing failure, a safety
            # failure must never turn an uncertain command into auto-approval.
            return SafetyOutcome(
                "deny" if self.headless else "ask",
                "jev_error_failclosed",
                reason=str(exc),
            )
        safe = (
            result.score <= SAFE_MAX
            and result.call_confidence >= SAFETY_CONFIDENCE
            and result.touches_outside_cwd < SAFETY_OUTSIDE_CWD_THRESHOLD
            and result.plausibly_irreversible < SAFETY_IRREVERSIBLE_THRESHOLD
        )
        return SafetyOutcome(
            "allow" if safe else ("deny" if self.headless else "ask"),
            "jev",
            score=result.score,
            confidence=result.call_confidence,
            reason=(None if safe else _triggering_judgment(result)),
            usage=result.usage,
        )

    def fail_closed(self, error: BaseException) -> SafetyOutcome:
        """Return the safe side when local safety machinery itself fails."""

        # This polarity is deliberate: every safety error is uncertain, so it
        # must never become an auto-approval.
        return self._finish(
            SafetyOutcome(
                "deny" if self.headless else "ask",
                "safety_error_failclosed",
                reason=str(error),
            )
        )

    def teaching_error(self, outcome: SafetyOutcome) -> str:
        score = "unknown" if outcome.score is None else str(outcome.score)
        confidence = (
            "unknown" if outcome.confidence is None else f"{outcome.confidence:.2f}"
        )
        reason = outcome.reason or "the command did not meet the safety threshold"
        return (
            "safety tier denied this shell command: "
            f"score={score}, confidence={confidence}, level={_level_meaning(outcome.score)}, "
            f"trigger={reason}; narrow the command or ask the user"
        )

    def approval_label(self, outcome: SafetyOutcome) -> str:
        score = "unknown" if outcome.score is None else str(outcome.score)
        return (
            "safety tier: "
            f"score={score}, level={_level_meaning(outcome.score)}, "
            f"trigger={outcome.reason or 'safety_threshold_not_met'}"
        )

    def _finish(self, outcome: SafetyOutcome) -> SafetyOutcome:
        event = {
            "service": "jev" if outcome.layer == "jev" else "zeta",
            "score": outcome.score,
            "confidence": outcome.confidence,
            "decision": outcome.decision,
            "layer": outcome.layer,
            "skip_reason": _skip_reason(outcome),
            "trigger": outcome.reason,
        }
        if outcome.usage is not None:
            event["usage"] = dict(outcome.usage)
        if self.telemetry is not None:
            self.telemetry(event)
        _logger.info("safety tier decision", extra={"safety_tier": event})
        return outcome


def _triggering_judgment(result: jev.SafetyScoreResult) -> str:
    if result.call_confidence < SAFETY_CONFIDENCE:
        return "low_confidence"
    if result.touches_outside_cwd >= SAFETY_OUTSIDE_CWD_THRESHOLD:
        return "touches paths outside cwd"
    if result.plausibly_irreversible >= SAFETY_IRREVERSIBLE_THRESHOLD:
        return "plausibly irreversible"
    if result.score > SAFE_MAX:
        return "score_exceeds"
    return "safety_threshold_not_met"


def _level_meaning(score: int | None) -> str:
    return {
        0: "read-only inspection",
        1: "reversible workspace write",
        2: "scoped destructive action",
        3: "dangerous or out of scope",
    }.get(score, "unknown")


def _skip_reason(outcome: SafetyOutcome) -> str | None:
    if outcome.layer == "layer0":
        return "layer0_escalated"
    if outcome.layer == "jev" and outcome.decision != "allow":
        return "threshold_not_met"
    if outcome.layer in {"jev_error_failclosed", "safety_error_failclosed"}:
        return outcome.layer
    return outcome.reason if outcome.decision != "allow" else None
