"""The optional Jev safety tier for yolo shell commands."""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..providers import jev

SAFE_MAX = 1.0
SAFETY_CONFIDENCE = 0.8
SHELL_TOOLS = frozenset({"bash", "exec", "run_background"})
_logger = logging.getLogger(__name__)

# Match privilege escalation, including env wrappers, absolute paths, and aliases.
_LAYER0_TEXT_PATTERNS = (
    ("sudo", re.compile(r"(?:^|[;&|]\s*)(?:(?:env|command|exec)\s+)?(?:[A-Za-z_]\w*=\S+\s+)*(?:sudo|/[^\s;&|]*/sudo)(?:\s|$)|(?:^|[;&|]\s*)alias\b[^\n;|&]*\bsudo\b", re.IGNORECASE)),
    # Catch network or decoded input that streams directly into a shell.
    ("pipe_to_shell", re.compile(r"(?:\b(?:curl|wget|fetch)\b[^|\n]*\|\s*(?:[^\s|]+/)?(?:ba|z|fi)?sh\b|<\([^)]*\b(?:curl|wget|fetch)\b[^)]*\)|\bbase64\b[^|\n]*\|\s*(?:[^\s|]+/)?(?:ba|z|fi)?sh\b)", re.IGNORECASE)),
)


@dataclass(frozen=True, slots=True)
class SafetyOutcome:
    decision: str
    layer: str
    score: int | None = None
    confidence: float | None = None
    reason: str | None = None
    usage: dict[str, int] | None = None


def _command_words(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _command_positions(words: list[str]) -> list[tuple[int, str]]:
    positions: list[tuple[int, str]] = []
    command_start = True
    for index, word in enumerate(words):
        if word in {";", "&&", "||", "|"}:
            command_start = True
            continue
        if not command_start:
            continue
        if word in {"env", "command", "exec"} or (
            "=" in word and word.split("=", 1)[0].replace("_", "a").isalnum()
        ):
            continue
        positions.append((index, word))
        command_start = False
    return positions


def _path_is_outside(path_text: str, cwd_path: Path) -> bool:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = cwd_path / path
    try:
        path.resolve().relative_to(cwd_path)
    except ValueError:
        return True
    return False


def _credential_path(path_text: str, cwd_path: Path) -> bool:
    if path_text.startswith("-"):
        return False
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = cwd_path / path
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path.absolute()
    parts = {part.casefold() for part in resolved.parts}
    name = resolved.name.casefold()
    return bool(
        {".ssh", ".aws"} & parts
        or name.endswith(".pem")
        or "keychain" in name
        or "login.keychain" in name
    )


def _rm_reason(words: list[str], cwd_path: Path) -> str | None:
    for index, command_word in _command_positions(words):
        if Path(command_word).name.casefold() != "rm":
            continue
        recursive = False
        force = False
        targets: list[str] = []
        after_options = False
        for word in words[index + 1 :]:
            if word in {";", "&&", "||", "|"}:
                break
            if word == "--":
                after_options = True
                continue
            if not after_options and word.startswith("--"):
                recursive |= word == "--recursive"
                force |= word == "--force"
                continue
            if not after_options and word.startswith("-") and word != "-":
                recursive |= "r" in word.casefold()
                force |= "f" in word.casefold()
                continue
            targets.append(word)
        if any("$" in target for target in targets):
            return "rm_unresolved_target"
        if recursive and force and any(
            target in {"/", "~"} or target.startswith(("/*", "~/"))
            for target in targets
        ):
            return "rm_root"
    return None


def _recursive_permission_reason(words: list[str], cwd_path: Path) -> str | None:
    for index, command_word in _command_positions(words):
        if Path(command_word).name.casefold() not in {"chmod", "chown"}:
            continue
        args = words[index + 1 :]
        recursive = any(
            word == "--recursive" or (word.startswith("-") and "R" in word)
            for word in args
        )
        if not recursive:
            continue
        targets = [
            word
            for word in args
            if not word.startswith("-") and not word.isdecimal()
        ]
        if any(_path_is_outside(target, cwd_path) for target in targets):
            return "recursive_permission_change_outside_cwd"
    return None


def layer0_reason(command: str, cwd: str | Path) -> str | None:
    """Return the small, deterministic always-escalate pattern that matches."""

    cwd_path = Path(cwd).expanduser().resolve()
    words = _command_words(command)
    for reason, pattern in _LAYER0_TEXT_PATTERNS:
        if pattern.search(command):
            return reason
    reason = _rm_reason(words, cwd_path)
    if reason is not None:
        return reason
    reason = _recursive_permission_reason(words, cwd_path)
    if reason is not None:
        return reason
    readers = {"cat", "head", "tail", "less", "more", "sed", "awk", "grep", "rg", "openssl"}
    if (
        words
        and Path(words[0]).name.casefold() in readers
        and any(_credential_path(word, cwd_path) for word in words[1:])
    ):
        return "credential_file_read"
    profile_path = re.compile(
        r"(?:~?/\.bash_history|~?/\.zsh_history|~?/\.bashrc|~?/\.zshrc|~?/\.profile|~?/\.bash_profile|~?/\.zprofile)",
        re.IGNORECASE,
    )
    if re.search(r"\bhistory\s+-[wc]\b", command) or (
        profile_path.search(command)
        and re.search(r">>{0,1}|\btee\b", command)
    ):
        return "history_or_shell_profile_write"
    return None


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
        reason = layer0_reason(command, cwd)
        if reason is not None:
            return self._finish(
                SafetyOutcome(
                    "deny" if self.headless else "ask",
                    "layer0",
                    reason=reason,
                )
            )
        try:
            result = await jev.safety_score(command, cwd, self.task_excerpt)
        except Exception as exc:  # noqa: BLE001 - safety must fail closed
            # This polarity is deliberate: unlike a routing failure, a safety
            # failure must never turn an uncertain command into auto-approval.
            return self._finish(
                SafetyOutcome(
                    "deny" if self.headless else "ask",
                    "jev_error_failclosed",
                    reason=str(exc),
                )
            )
        safe = result.score <= SAFE_MAX and result.call_confidence >= SAFETY_CONFIDENCE
        return self._finish(
            SafetyOutcome(
                "allow" if safe else ("deny" if self.headless else "ask"),
                "jev",
                score=result.score,
                confidence=result.call_confidence,
                reason=(None if safe else _triggering_judgment(result)),
                usage=result.usage,
            )
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
            event["usage"] = {"service": "jev", "usage": dict(outcome.usage)}
        if self.telemetry is not None:
            self.telemetry(event)
        _logger.info("safety tier decision", extra={"safety_tier": event})
        return outcome


def _triggering_judgment(result: jev.SafetyScoreResult) -> str:
    if result.call_confidence < SAFETY_CONFIDENCE:
        return "low_confidence"
    if result.score > SAFE_MAX:
        return "score_exceeds"
    if result.touches_outside_cwd >= 0.5:
        return "touches paths outside cwd"
    if result.plausibly_irreversible >= 0.5:
        return "plausibly irreversible"
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


__all__ = [
    "SAFETY_CONFIDENCE",
    "SAFE_MAX",
    "SHELL_TOOLS",
    "SafetyOutcome",
    "SafetyTier",
    "layer0_reason",
]
