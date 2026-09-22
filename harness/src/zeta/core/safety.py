"""The optional Jev safety tier for yolo shell commands."""

from __future__ import annotations

import logging
import os
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..providers import jev

SAFE_MAX = 1.0
SAFETY_CONFIDENCE = 0.8
NOUL_THRESHOLD = 0.5
SHELL_TOOLS = frozenset({"bash", "exec", "run_background"})
_logger = logging.getLogger(__name__)

# Layer 0 is a proof gate. It never tries to enumerate every shell spelling.
# The table records the only accepted classifications and their evidence.
#
#   deny: certain-dangerous evidence; Jev is not consulted.
#   escalate: the input is not fully analyzable; Jev cannot grant a bypass.
#   analyzable: every grammar, wrapper, expansion, and path condition passed.
_LAYER0_RULES = (
    ("deny", "privilege_escalation", "sudo, doas, or pkexec"),
    ("deny", "credential_file_read", "a plausible credential path argument"),
    ("deny", "destructive_system_path", "a destructive target at a system prefix"),
    ("deny", "pipe_to_shell", "a pipeline executes a shell"),
    ("escalate", "parse_error", "lexing or the small grammar fails"),
    ("escalate", "nested_shell", "a shell, interpreter, or shell payload appears"),
    ("escalate", "unresolved_expansion", "a variable or path cannot be resolved"),
    (
        "escalate",
        "destructive_target_outside_workspace",
        "a destructive target is not scoped",
    ),
    ("analyzable", "", "a simple command or list passed every positive check"),
)

# ANALYZABLE requires all of the following: clean lexing with this grammar;
# simple commands joined only by ;, &&, ||, or |; recursively resolved wrappers;
# literal or provided-environment-resolved words; no command or process
# substitution, backticks, heredoc, source, eval, exec, backgrounding, shell,
# or exec-capable interpreter; no system-path redirection or system-path glob;
# and literal destructive targets resolved inside the workspace. Any unknown
# construct is ESCALATE. DENY evidence is checked before this result.
_SHELL_INTERPRETERS = frozenset(
    {"ash", "bash", "csh", "dash", "fish", "ksh", "sh", "tcsh", "zsh"}
)
_SHELL_RESERVED_WORDS = frozenset(
    {
        "!",
        "case",
        "coproc",
        "do",
        "done",
        "elif",
        "else",
        "esac",
        "fi",
        "for",
        "function",
        "if",
        "in",
        "select",
        "then",
        "until",
        "while",
    }
)
_INTERPRETER_NAMES = frozenset({"awk", "node", "perl", "php", "ruby"})
_COMMAND_WRAPPERS = frozenset(
    {"env", "nice", "nohup", "setsid", "stdbuf", "time", "timeout", "command", "xargs"}
)
_LIST_OPERATORS = frozenset({";", "&&", "||", "|"})
_REDIRECTION_OPERATORS = frozenset({"<", ">", ">>", "<>"})
_SYNTAX_OPERATORS = frozenset({"&", "(", ")"})
_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")
_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)=(.*)", re.DOTALL)
_GLOB = re.compile(r"[*?\[\]{}]")
_SYSTEM_PREFIXES = (
    "/etc",
    "/var",
    "/usr",
    "/dev",
    "/bin",
    "/sbin",
    "/lib",
    "/boot",
    "/system",
    "/library",
)
_DESTRUCTIVE_NAMES = frozenset(
    {"chmod", "chown", "dd", "mkfs", "mv", "rm", "rmdir", "shred", "truncate"}
)
_XARGS_OPTION_ARGUMENTS = frozenset(
    {
        "-E",
        "-I",
        "-L",
        "-n",
        "-P",
        "-R",
        "--max-args",
        "--max-lines",
        "--process-slot-var",
        "--replace",
        "--eof",
    }
)


@dataclass(frozen=True, slots=True)
class SafetyOutcome:
    decision: str
    layer: str
    score: int | None = None
    confidence: float | None = None
    reason: str | None = None
    usage: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class _ParsedShell:
    segments: tuple[tuple[str, ...], ...]
    redirections: tuple[tuple[str, str], ...]
    has_pipeline: bool


def _basename(word: str) -> str:
    if word in {".", ".."}:
        return word
    return Path(word).name.casefold()


def _syntax_reason(command: str) -> str | None:
    """Reject shell syntax that the small grammar does not model."""

    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if char == "'":
            if quote is None:
                quote = "'"
            elif quote == "'":
                quote = None
            index += 1
            continue
        if char == '"':
            if quote is None:
                quote = '"'
            elif quote == '"':
                quote = None
            index += 1
            continue
        if quote != "'" and char == "`":
            return "nested_shell"
        if quote != "'" and command.startswith("$(", index):
            return "command_substitution"
        if (
            quote is None
            and char in "<>"
            and index + 1 < len(command)
            and command[index + 1] == "("
        ):
            return "process_substitution"
        index += 1
    if escaped or quote is not None:
        return "parse_error"
    return None


def _lex(command: str) -> tuple[list[str] | None, str | None]:
    syntax = _syntax_reason(command)
    if syntax is not None:
        return None, syntax
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer), None
    except ValueError:
        return None, "parse_error"


def _parse(command: str) -> tuple[_ParsedShell | None, str | None]:
    tokens, error = _lex(command)
    if error is not None or tokens is None:
        return None, error or "parse_error"
    segments: list[tuple[str, ...]] = []
    redirections: list[tuple[str, str]] = []
    current: list[str] = []
    has_pipeline = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _LIST_OPERATORS:
            if not current:
                return None, "parse_error"
            segments.append(tuple(current))
            current = []
            has_pipeline |= token == "|"
            index += 1
            continue
        if token in _SYNTAX_OPERATORS or token in {"<<", "<<<", "&>"}:
            return None, "parse_error"
        if token in _REDIRECTION_OPERATORS:
            if index + 1 >= len(tokens):
                return None, "parse_error"
            target = tokens[index + 1]
            if target in _LIST_OPERATORS or target in _SYNTAX_OPERATORS:
                return None, "parse_error"
            redirections.append((token, target))
            index += 2
            continue
        current.append(token)
        index += 1
    if not current:
        return None, "parse_error"
    segments.append(tuple(current))
    if any(token in {"{", "}"} for segment in segments for token in segment):
        return None, "compound_command"
    return _ParsedShell(tuple(segments), tuple(redirections), has_pipeline), None


def _expand_word(word: str, environment: Mapping[str, str]) -> str | None:
    unresolved = False

    def replace(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = match.group(1) or match.group(2)
        if name not in environment:
            unresolved = True
            return match.group(0)
        return environment[name]

    expanded = _VARIABLE.sub(replace, word)
    if unresolved or "$" in expanded:
        return None
    return os.path.expanduser(expanded)


def _resolve_segments(
    parsed: _ParsedShell, environment: Mapping[str, str] | None
) -> tuple[_ParsedShell | None, dict[str, str], str | None]:
    resolved_environment = dict(os.environ if environment is None else environment)
    segments: list[tuple[str, ...]] = []
    for segment in parsed.segments:
        resolved: list[str] = []
        command_seen = False
        for word in segment:
            assignment = _ASSIGNMENT.fullmatch(word)
            if assignment is not None and not command_seen:
                value = _expand_word(assignment.group(2), resolved_environment)
                if value is None:
                    return None, resolved_environment, "unresolved_expansion"
                resolved_environment[assignment.group(1)] = value
                resolved.append(f"{assignment.group(1)}={value}")
                continue
            command_seen = True
            value = _expand_word(word, resolved_environment)
            if value is None:
                return None, resolved_environment, "unresolved_expansion"
            resolved.append(value)
        segments.append(tuple(resolved))
    redirections: list[tuple[str, str]] = []
    for operator, target in parsed.redirections:
        value = _expand_word(target, resolved_environment)
        if value is None:
            return None, resolved_environment, "unresolved_expansion"
        redirections.append((operator, value))
    return (
        _ParsedShell(tuple(segments), tuple(redirections), parsed.has_pipeline),
        resolved_environment,
        None,
    )


def _skip_options(segment: tuple[str, ...], index: int, name: str) -> int:
    """Return the wrapped argv index for the supported option-bearing wrappers."""

    index += 1
    consumed_timeout = False
    while index < len(segment):
        word = segment[index]
        if word == "--":
            return index + 1
        if name == "timeout" and not word.startswith("-") and not consumed_timeout:
            consumed_timeout = True
            index += 1
            continue
        if not word.startswith("-") or word == "-":
            return index
        elif (
            name == "env"
            and word in {"-u", "--unset", "-S", "--split-string"}
            or name == "nice"
            and word in {"-n", "--adjustment"}
            or name == "stdbuf"
            and word in {"-i", "-o", "-e"}
            or name == "time"
            and word in {"-f", "--format"}
        ):
            index += 2
        else:
            index += 1
    return index


def _resolved_argv(segment: tuple[str, ...]) -> tuple[int, str] | None:
    index = 0
    while index < len(segment):
        if _ASSIGNMENT.fullmatch(segment[index]) is not None:
            index += 1
            continue
        name = _basename(segment[index])
        if name == "xargs":
            index += 1
            while index < len(segment):
                word = segment[index]
                if word == "--":
                    index += 1
                    break
                if word in _XARGS_OPTION_ARGUMENTS:
                    index += 2
                    continue
                if word.startswith("-") and word != "-":
                    index += 1
                    continue
                return _resolved_argv(segment[index:])
            return (len(segment), "echo")
        if name in _COMMAND_WRAPPERS - {"xargs"}:
            index = _skip_options(segment, index, name)
            continue
        return index, name
    return None


def _assignment_only(segment: tuple[str, ...]) -> bool:
    return bool(segment) and all(_ASSIGNMENT.fullmatch(word) for word in segment)


def _path_value(text: str, cwd: Path) -> Path:
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        return candidate.resolve(strict=False)
    except OSError:
        return candidate.absolute()


def _inside(path_text: str, cwd: Path) -> bool:
    try:
        _path_value(path_text, cwd).relative_to(cwd)
    except ValueError:
        return False
    return True


def _system_path(text: str, cwd: Path) -> bool:
    raw = text.casefold()
    if raw in {"/", "~"}:
        return True
    normalized_raw = os.path.normpath(raw)
    if any(
        normalized_raw == prefix or normalized_raw.startswith(f"{prefix}/")
        for prefix in _SYSTEM_PREFIXES
    ):
        return True
    resolved = str(_path_value(text, cwd)).casefold()
    return any(
        resolved == prefix or resolved.startswith(f"{prefix}/")
        for prefix in _SYSTEM_PREFIXES
    )


def _credential_path(text: str, cwd: Path) -> bool:
    if text.startswith("-"):
        return False
    value = text.partition("=")[2] if "=" in text else text
    plausible = (
        "/" in value
        or value.startswith(("~", "$HOME"))
        or _path_value(value, cwd).exists()
    )
    if not plausible:
        return False
    resolved = _path_value(value, cwd)
    parts = {part.casefold() for part in resolved.parts}
    name = resolved.name.casefold()
    return bool(
        {".ssh", ".aws"} & parts
        or name.endswith(".pem")
        or "keychain" in name
        or "login.keychain" in name
    )


def _destructive_targets(name: str, arguments: tuple[str, ...]) -> list[str]:
    targets: list[str] = []
    after_options = False
    skip_next = False
    for word in arguments:
        if skip_next:
            skip_next = False
            continue
        if name == "dd" and word.partition("=")[0].casefold() == "of":
            targets.append(word.partition("=")[2])
            continue
        if not after_options and word == "--":
            after_options = True
            continue
        if not after_options and word.startswith("-") and word != "-":
            if name == "truncate" and word in {"-s", "--size"}:
                skip_next = True
            continue
        if name in {"chmod", "chown"} and (word.isdecimal() or ":" in word):
            continue
        targets.append(word)
    return targets


def _destructive_reason(
    name: str, arguments: tuple[str, ...], cwd: Path
) -> tuple[str, str] | None:
    if name not in _DESTRUCTIVE_NAMES:
        return None
    targets = _destructive_targets(name, arguments)
    if not targets:
        return "escalate", "destructive_target_unresolved"
    recursive_permission = name in {"chmod", "chown"} and any(
        argument == "--recursive" or (argument.startswith("-") and "R" in argument)
        for argument in arguments
    )
    for target in targets:
        if target in {"/*", "~/*"}:
            return "deny", "rm_root"
        if _GLOB.search(target) and (target.startswith(("/", "~")) or name == "dd"):
            if _system_path(target, cwd):
                return "deny", "root_scope_expansion"
            return "escalate", "root_scope_expansion"
        if _system_path(target, cwd):
            if target in {"/", "~"}:
                return "deny", "rm_root"
            if recursive_permission:
                return "deny", "recursive_permission_change_outside_cwd"
            return "deny", "destructive_system_path"
        if not _inside(target, cwd):
            if recursive_permission:
                return "escalate", "recursive_permission_change_outside_cwd"
            return "escalate", "destructive_target_outside_workspace"
    return None


def _profile_reason(parsed: _ParsedShell) -> str | None:
    for _operator, target in parsed.redirections:
        name = Path(target).name.casefold()
        if name in {
            ".bash_history",
            ".zsh_history",
            ".bashrc",
            ".zshrc",
            ".profile",
            ".bash_profile",
            ".zprofile",
        }:
            return "history_or_shell_profile_write"
    for segment in parsed.segments:
        argv = _resolved_argv(segment)
        if argv is None:
            continue
        index, name = argv
        arguments = segment[index + 1 :]
        if name == "history" and any(
            argument in {"-w", "-c", "-d"} for argument in arguments
        ):
            return "history_or_shell_profile_write"
        if name == "tee" and any(
            Path(argument).name.casefold().endswith("rc") for argument in arguments
        ):
            return "history_or_shell_profile_write"
    return None


def _credential_reason(parsed: _ParsedShell, cwd: Path) -> str | None:
    for segment in parsed.segments:
        argv = _resolved_argv(segment)
        if argv is None:
            continue
        index, name = argv
        arguments = segment[index + 1 :]
        git_add = name == "git" and arguments and arguments[0] == "add"
        for argument in arguments:
            if git_add and argument.endswith(".pem") and _inside(argument, cwd):
                continue
            if _credential_path(argument, cwd):
                return "credential_file_read"
    return None


def _shell_reason(parsed: _ParsedShell) -> str | None:
    argv: list[tuple[int, str] | None] = [
        _resolved_argv(segment) for segment in parsed.segments
    ]
    if parsed.has_pipeline:
        final = argv[-1]
        if final is not None and final[1] in _SHELL_INTERPRETERS:
            return "pipe_to_shell"
    for segment, resolved in zip(parsed.segments, argv, strict=True):
        if resolved is None:
            if _assignment_only(segment):
                continue
            return "parse_error"
        _index, name = resolved
        if name == "alias":
            if any(
                word.casefold() == "sudo" or "sudo" in word.casefold()
                for word in segment
            ):
                return "sudo"
            return "nested_shell"
        if name in {"source", ".", "eval", "exec"}:
            return "nested_shell"
        if name in _SHELL_RESERVED_WORDS:
            return "nested_shell"
        if (
            name in _SHELL_INTERPRETERS
            or name in _INTERPRETER_NAMES
            or name.startswith("python")
        ):
            return "nested_shell"
        if name in {"sudo", "doas", "pkexec"}:
            return "sudo"
    return None


def layer0_classify(
    command: str,
    cwd: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, str | None]:
    """Classify a command as deny, escalate, or analyzable."""

    cwd_path = Path(cwd).expanduser().resolve()
    parsed, parse_error = _parse(command)
    if parse_error is not None or parsed is None:
        if re.search(
            r"\|\s*(?:\([^)]*\b(?:sh|bash|zsh|dash|ksh)\b[^)]*\)|\{[^}]*\b(?:sh|bash|zsh|dash|ksh)\b[^}]*\})",
            command,
            re.IGNORECASE,
        ):
            return "deny", "pipe_to_shell"
        if parse_error == "process_substitution":
            return "escalate", "nested_shell"
        return "escalate", parse_error or "parse_error"
    raw_shell_reason = _shell_reason(parsed)
    if raw_shell_reason == "sudo":
        return "deny", "sudo"
    if raw_shell_reason == "pipe_to_shell":
        return "deny", raw_shell_reason
    if raw_shell_reason is not None:
        return "escalate", raw_shell_reason
    for segment in parsed.segments:
        resolved = _resolved_argv(segment)
        if resolved is None:
            continue
        index, name = resolved
        if name not in _DESTRUCTIVE_NAMES:
            continue
        for target in _destructive_targets(name, segment[index + 1 :]):
            if _system_path(target, cwd_path):
                if target in {"/", "~", "/*", "~/*"}:
                    return "deny", "rm_root"
                if name in {"chmod", "chown"} and any(
                    argument == "--recursive"
                    or (argument.startswith("-") and "R" in argument)
                    for argument in segment[index + 1 :]
                ):
                    return "deny", "recursive_permission_change_outside_cwd"
                if _GLOB.search(target):
                    return "deny", "root_scope_expansion"
                return "deny", "destructive_system_path"
    parsed, _resolved_environment, expansion_error = _resolve_segments(
        parsed, environment
    )
    if expansion_error is not None or parsed is None:
        if expansion_error == "unresolved_expansion" and re.search(
            r"(?:^|[;|&])\s*(?:env\s+)?rm\b", command
        ):
            return "escalate", "rm_unresolved_target"
        return "escalate", expansion_error or "unresolved_expansion"
    shell_reason = _shell_reason(parsed)
    if shell_reason == "sudo":
        return "deny", "sudo"
    if shell_reason == "pipe_to_shell":
        return "deny", shell_reason
    if shell_reason is not None:
        return "escalate", shell_reason
    for segment in parsed.segments:
        resolved = _resolved_argv(segment)
        if resolved is None:
            if _assignment_only(segment):
                continue
            return "escalate", "parse_error"
        index, name = resolved
        destructive = _destructive_reason(name, segment[index + 1 :], cwd_path)
        if destructive is not None:
            return destructive
    reason = _profile_reason(parsed)
    if reason is not None:
        return "deny", reason
    for _operator, target in parsed.redirections:
        if _system_path(target, cwd_path):
            return "escalate", "system_path_redirection"
        if _GLOB.search(target):
            return "escalate", "unresolved_expansion"
    for segment in parsed.segments:
        resolved = _resolved_argv(segment)
        if resolved is None:
            continue
        index, _name = resolved
        if any(
            _GLOB.search(argument) and _system_path(argument, cwd_path)
            for argument in segment[index + 1 :]
        ):
            return "escalate", "system_path_glob"
    reason = _credential_reason(parsed, cwd_path)
    if reason is not None:
        return "deny", reason
    return "analyzable", None


def layer0_reason(command: str, cwd: str | Path) -> str | None:
    """Return the deterministic layer-0 reason, if one blocks auto-approval."""

    _classification, reason = layer0_classify(command, cwd)
    return reason


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
        safe = (
            result.score <= SAFE_MAX
            and result.call_confidence >= SAFETY_CONFIDENCE
            and result.touches_outside_cwd < NOUL_THRESHOLD
            and result.plausibly_irreversible < NOUL_THRESHOLD
        )
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
            event["usage"] = dict(outcome.usage)
        if self.telemetry is not None:
            self.telemetry(event)
        _logger.info("safety tier decision", extra={"safety_tier": event})
        return outcome


def _triggering_judgment(result: jev.SafetyScoreResult) -> str:
    if result.call_confidence < SAFETY_CONFIDENCE:
        return "low_confidence"
    if result.touches_outside_cwd >= NOUL_THRESHOLD:
        return "touches paths outside cwd"
    if result.plausibly_irreversible >= NOUL_THRESHOLD:
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


__all__ = [
    "NOUL_THRESHOLD",
    "SAFETY_CONFIDENCE",
    "SAFE_MAX",
    "SHELL_TOOLS",
    "SafetyOutcome",
    "SafetyTier",
    "layer0_classify",
    "layer0_reason",
]
