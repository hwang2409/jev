from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .runner import (
    State,
    StateAdmission,
    StateLimitError,
    StateLimits,
    StateRejection,
    admit_states,
    validate_state,
)

DEFAULT_FOCUS_BYTES = 16_384
DEFAULT_CONTEXT_FIELD_BYTES = 4_096
DEFAULT_STATE_BYTES = 32_768
SURROUNDING_TRUNCATION_MARKER = "[... surrounding truncated ...]"


def decode_stdin(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _as_text(value: str | bytes) -> str:
    return decode_stdin(value) if isinstance(value, bytes) else value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fits_context(value: Any, limit: int) -> bool:
    return len(_canonical_json(value).encode("utf-8")) <= limit


def _prefix_with_marker(value: str, limit: int) -> str:
    marker = SURROUNDING_TRUNCATION_MARKER
    low = 0
    high = len(value)
    best = marker
    while low <= high:
        midpoint = (low + high) // 2
        candidate = value[:midpoint] + marker
        if _fits_context(candidate, limit):
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    if not _fits_context(marker, limit):
        raise ValueError(
            "context_field_bytes is too small for the surrounding truncation marker"
        )
    return best


def _truncate_surrounding_list(value: list[Any], limit: int) -> list[Any]:
    marker = SURROUNDING_TRUNCATION_MARKER
    if not _fits_context([marker], limit):
        raise ValueError(
            "context_field_bytes is too small for the surrounding truncation marker"
        )

    retained: list[Any] = []
    for item in value:
        candidate = [*retained, item, marker]
        if _fits_context(candidate, limit):
            retained.append(item)
            continue
        if not isinstance(item, str):
            return [*retained, marker]

        low = 0
        high = len(item)
        best: list[Any] | None = None
        while low <= high:
            midpoint = (low + high) // 2
            truncated = [*retained, item[:midpoint] + marker]
            if _fits_context(truncated, limit):
                best = truncated
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best is None:
            raise ValueError(
                "context_field_bytes is too small for the surrounding truncation marker"
            )
        return best
    return [*retained, marker]


def _truncate_surrounding(value: Any, limit: int) -> Any:
    """Bound surrounding context while keeping its list-shaped contract."""

    if _fits_context(value, limit):
        return value
    if isinstance(value, str):
        return _prefix_with_marker(value, limit)
    if isinstance(value, list):
        return _truncate_surrounding_list(value, limit)
    return _prefix_with_marker(_canonical_json(value), limit)


def _bounded_context(
    context: Mapping[str, Any], limits: StateLimits
) -> dict[str, Any]:
    bounded = dict(context)
    surrounding = bounded.get("surrounding")
    if surrounding is not None:
        bounded["surrounding"] = _truncate_surrounding(
            surrounding, limits.context_field_bytes
        )
    return bounded


def _split_utf8(value: str, max_bytes: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for character in value:
        character_bytes = len(character.encode("utf-8"))
        if character_bytes > max_bytes:
            raise StateLimitError(
                f"focus contains a character larger than focus_bytes={max_bytes}"
            )
        if current and current_bytes + character_bytes > max_bytes:
            pieces.append("".join(current))
            current = []
            current_bytes = 0
        current.append(character)
        current_bytes += character_bytes
    if current or not pieces:
        pieces.append("".join(current))
    return pieces


def _state(
    state_ref: str,
    focus: str,
    context: Mapping[str, Any],
    limits: StateLimits,
    split_focus: bool = False,
    rejections: list[StateRejection] | None = None,
    source_ref: str | None = None,
) -> list[State]:
    state = State(
        state_ref,
        focus,
        _bounded_context(context, limits),
        source_ref=source_ref,
    )
    try:
        validate_state(state, limits)
    except StateLimitError as exc:
        if not split_focus or not str(exc).startswith("focus exceeds"):
            if rejections is None:
                raise
            rejections.append(
                StateRejection(state_ref, "context_limit", str(exc), source_ref)
            )
            return []
        try:
            pieces = _split_utf8(focus, limits.focus_bytes)
        except StateLimitError as split_exc:
            if rejections is None:
                raise
            rejections.append(
                StateRejection(state_ref, "context_limit", str(split_exc), source_ref)
            )
            return []
        total = len(pieces)
        states = []
        for index, piece in enumerate(pieces, start=1):
            states.extend(
                _state(
                    f"{state_ref}/{index}",
                    piece,
                    {**context, "subunit": f"{index}/{total}"},
                    limits,
                    split_focus=False,
                    rejections=rejections,
                    source_ref=source_ref,
                )
            )
        return states
    return [state]


def _line_units(text: str) -> list[str]:
    return text.splitlines()


def _jsonl_lines(value: str | bytes) -> Iterable[tuple[int, int, str | None]]:
    offset = 0
    if isinstance(value, bytes):
        lines = value.splitlines(keepends=True)
        for line_number, raw_line in enumerate(lines, start=1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = None
            yield line_number, offset, line
            offset += len(raw_line)
        return

    for line_number, line in enumerate(value.splitlines(keepends=True), start=1):
        yield line_number, offset, line
        offset += len(line.encode("utf-8"))


def _input_error(
    rejections: list[StateRejection] | None,
    message: str,
    source_ref: str,
) -> None:
    if rejections is None:
        raise ValueError(message)
    rejections.append(StateRejection(None, "input_error", message, source_ref))


def _with_parameters(
    context: Mapping[str, Any], parameters: Mapping[str, str] | None
) -> dict[str, Any]:
    result = dict(context)
    if parameters:
        result.update(parameters)
    return result


def chunk_line(
    text: str | bytes,
    *,
    source: str = "stdin",
    adjacent_lines: int = 1,
    query: str | None = None,
    predicate: str | None = None,
    parameters: Mapping[str, str] | None = None,
    limits: StateLimits = StateLimits(),
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    lines = _line_units(_as_text(text))
    states: list[State] = []
    for index, focus in enumerate(lines, start=1):
        start = max(0, index - 1 - adjacent_lines)
        end = min(len(lines), index + adjacent_lines)
        surrounding = [
            line
            for offset, line in enumerate(lines[start:end], start=start + 1)
            if offset != index
        ]
        context: dict[str, Any] = {
            "source": source,
            "unit": "line",
            "line": index,
            "surrounding": surrounding,
        }
        if query is not None:
            context["query"] = query
        if predicate is not None:
            context["predicate"] = predicate
        context = _with_parameters(context, parameters)
        states.extend(
            _state(
                f"{source}#L{index}",
                focus,
                context,
                limits,
                split_focus=True,
                rejections=_rejections,
            )
        )
    return states


def _paragraph_units(text: str) -> list[tuple[int, str]]:
    units: list[tuple[int, str]] = []
    current: list[str] = []
    start_line = 1
    line_number = 0
    for line in text.splitlines():
        line_number += 1
        if not line.strip():
            if current:
                units.append((start_line, "\n".join(current)))
                current = []
            start_line = line_number + 1
            continue
        if not current:
            start_line = line_number
        current.append(line)
    if current:
        units.append((start_line, "\n".join(current)))
    return units


def _heading_before(lines: list[str], line_number: int) -> str | None:
    heading: str | None = None
    for line in lines[:line_number]:
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            heading = line.strip()
    return heading


def chunk_para(
    text: str | bytes,
    *,
    source: str = "stdin",
    adjacent_paragraphs: int = 1,
    query: str | None = None,
    predicate: str | None = None,
    parameters: Mapping[str, str] | None = None,
    limits: StateLimits = StateLimits(),
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    decoded = _as_text(text)
    all_units = _paragraph_units(decoded)
    units: list[tuple[int, int, str]] = []
    for paragraph, (line_number, focus) in enumerate(all_units, start=1):
        if all(
            re.match(r"^\s{0,3}#{1,6}\s+", line)
            for line in focus.splitlines()
        ):
            if _rejections is not None:
                _rejections.append(
                    StateRejection(
                        f"{source}#P{paragraph}",
                        "heading_only",
                        "heading-only paragraph skipped",
                    )
                )
            continue
        units.append((paragraph, line_number, focus))
    lines = decoded.splitlines()
    states: list[State] = []
    for index, (paragraph, line_number, focus) in enumerate(units, start=1):
        start = max(0, index - 1 - adjacent_paragraphs)
        end = min(len(units), index + adjacent_paragraphs)
        surrounding = [
            paragraph
            for offset, (_, _, paragraph) in enumerate(
                units[start:end], start=start + 1
            )
            if offset != index
        ]
        context: dict[str, Any] = {
            "source": source,
            "unit": "para",
            "paragraph": paragraph,
            "heading": _heading_before(lines, line_number),
            "surrounding": surrounding,
        }
        if query is not None:
            context["query"] = query
        if predicate is not None:
            context["predicate"] = predicate
        context = _with_parameters(context, parameters)
        states.extend(
            _state(
                f"{source}#P{paragraph}",
                focus,
                context,
                limits,
                split_focus=True,
                rejections=_rejections,
            )
        )
    return states


_HUNK_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(?:\s.*)?$"
)


def _normalise_path(value: str, *, strip_git_prefix: bool = False) -> str:
    path = value.strip()
    if strip_git_prefix and path.startswith(("a/", "b/")):
        path = path[2:]
    return PurePosixPath(path).as_posix()


def _decode_git_path(value: str) -> str:
    if not (value.startswith('"') and value.endswith('"')):
        return value

    decoded = bytearray()
    index = 1
    while index < len(value) - 1:
        character = value[index]
        if character != "\\":
            decoded.extend(character.encode("utf-8"))
            index += 1
            continue
        index += 1
        escaped = value[index]
        simple_escapes = {
            '"': b'"',
            "\\": b"\\",
            "b": b"\b",
            "f": b"\f",
            "n": b"\n",
            "r": b"\r",
            "t": b"\t",
            "v": b"\v",
        }
        if escaped in simple_escapes:
            decoded.extend(simple_escapes[escaped])
            index += 1
            continue
        if escaped in "01234567":
            digits = escaped
            index += 1
            while index < len(value) - 1 and len(digits) < 3:
                if value[index] not in "01234567":
                    break
                digits += value[index]
                index += 1
            decoded.append(int(digits, 8))
            continue
        decoded.extend(escaped.encode("utf-8"))
        index += 1
    return decoded.decode("utf-8")


def _diff_path(line: str) -> str:
    path = _decode_git_path(line.split("\t", 1)[0].strip())
    return _normalise_path(path, strip_git_prefix=True)


_NO_NEWLINE_MARKER = r"\ No newline at end of file"


def _strict_hunk_lines(
    diff: str | bytes,
    rejections: list[StateRejection] | None,
) -> Iterable[tuple[int, int, str | None]]:
    if isinstance(diff, str):
        offset = 0
        for line_number, raw_line in enumerate(
            diff.splitlines(keepends=True), start=1
        ):
            line = raw_line.rstrip("\r\n")
            yield line_number, offset, line
            offset += len(raw_line.encode("utf-8"))
        return

    offset = 0
    for line_number, raw_line in enumerate(diff.splitlines(keepends=True), start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            _input_error(
                rejections,
                f"hunk input line {line_number} is not valid UTF-8",
                f"stdin:byte={offset},line={line_number}",
            )
            line = None
        else:
            line = line.rstrip("\r\n")
        yield line_number, offset, line
        offset += len(raw_line)


def chunk_hunk(
    diff: str | bytes,
    *,
    limits: StateLimits = StateLimits(),
    changed_test_paths: Iterable[str] | None = None,
    parameters: Mapping[str, str] | None = None,
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    file_path = "stdin"
    current_header: str | None = None
    current_body: list[str] = []
    hunks: list[tuple[str, str, list[str]]] = []
    pending_old_path: str | None = None
    old_remaining = 0
    new_remaining = 0
    no_newline_can_attach = False
    last_line_number = 0
    input_byte_length = (
        len(diff.encode("utf-8")) if isinstance(diff, str) else len(diff)
    )

    def finish_hunk() -> None:
        nonlocal current_header, current_body, old_remaining, new_remaining
        nonlocal no_newline_can_attach
        if current_header is not None:
            hunks.append((file_path, current_header, current_body))
        current_header = None
        current_body = []
        old_remaining = 0
        new_remaining = 0
        no_newline_can_attach = False

    for line_number, byte_offset, line in _strict_hunk_lines(diff, _rejections):
        last_line_number = line_number
        if line is None:
            finish_hunk()
            continue
        if current_header is not None:
            if old_remaining > 0 or new_remaining > 0:
                if not line.startswith((" ", "+", "-")) and line != _NO_NEWLINE_MARKER:
                    _input_error(
                        _rejections,
                        "invalid unified diff hunk body line",
                        f"stdin:byte={byte_offset},line={line_number}",
                    )
                    finish_hunk()
                elif line == _NO_NEWLINE_MARKER:
                    if no_newline_can_attach:
                        current_body.append(line)
                        no_newline_can_attach = False
                        continue
                    _input_error(
                        _rejections,
                        "extra unified diff hunk body line",
                        f"stdin:byte={byte_offset},line={line_number}",
                    )
                    finish_hunk()
                else:
                    excess = (
                        line.startswith((" ", "-")) and old_remaining == 0
                    ) or (line.startswith((" ", "+")) and new_remaining == 0)
                    if excess:
                        _input_error(
                            _rejections,
                            "extra unified diff hunk body line",
                            f"stdin:byte={byte_offset},line={line_number}",
                        )
                        finish_hunk()
                    else:
                        current_body.append(line)
                        if line.startswith((" ", "-")):
                            old_remaining -= 1
                        if line.startswith((" ", "+")):
                            new_remaining -= 1
                        no_newline_can_attach = True
                        continue
            elif line == _NO_NEWLINE_MARKER and no_newline_can_attach:
                current_body.append(line)
                no_newline_can_attach = False
                continue
            elif not line.startswith(("diff --git ", "--- ", "+++ ", "@@ ")):
                _input_error(
                    _rejections,
                    "extra unified diff hunk body line",
                    f"stdin:byte={byte_offset},line={line_number}",
                )
                finish_hunk()
            else:
                finish_hunk()
        if line.startswith("diff --git "):
            pending_old_path = None
        elif line.startswith("--- "):
            pending_old_path = _diff_path(line[4:])
        elif line.startswith("+++ "):
            new_path = _diff_path(line[4:])
            file_path = pending_old_path if new_path == "/dev/null" else new_path
            pending_old_path = None
        match = _HUNK_RE.match(line)
        if match:
            current_header = line
            current_body = []
            old_remaining = int(match.group(2) or "1")
            new_remaining = int(match.group(4) or "1")
            no_newline_can_attach = False
    if current_header is not None:
        if old_remaining > 0 or new_remaining > 0:
            _input_error(
                _rejections,
                "unified diff hunk body ended before the declared counts",
                f"stdin:byte={input_byte_length},line={last_line_number + 1}",
            )
        finish_hunk()

    tests = sorted({_normalise_path(path) for path in changed_test_paths or ()})
    if not tests:
        tests = sorted(
            {
                file_path
                for file_path, _, _ in hunks
                if "/test" in f"/{file_path}" or file_path.startswith("test")
            }
        )

    states: list[State] = []
    for file_path, header, body in hunks:
        position = header.split("@@", 2)[1].replace(" ", "").strip()
        state_ref = f"{file_path}@@{position}"
        focus = "\n".join([header, *body])
        surrounding = [line[1:] for line in body if line.startswith(" ")]
        context = {
            "file": file_path,
            "unit": "hunk",
            "hunk_header": header,
            "surrounding": surrounding,
            "changed_tests": tests,
        }
        context = _with_parameters(context, parameters)
        states.extend(
            _state(
                state_ref,
                focus,
                context,
                limits,
                split_focus=True,
                rejections=_rejections,
            )
        )
    return states


def _normalise_file_path(path: str | Path) -> str:
    return _normalise_path(Path(path).as_posix())


def chunk_file(
    path: str | Path,
    content: str | bytes | None = None,
    *,
    limits: StateLimits = StateLimits(),
    query: str | None = None,
    predicate: str | None = None,
    parameters: Mapping[str, str] | None = None,
    _rejections: list[StateRejection] | None = None,
    _source_ref: str | None = None,
) -> list[State]:
    normalized = _normalise_file_path(path)
    if content is None:
        content = Path(path).read_bytes()
    decoded = _as_text(content)
    suffix = Path(normalized).suffix.lower().lstrip(".") or "text"
    context = {
        "path": normalized,
        "language": suffix,
        "metadata": {"size_bytes": len(decoded.encode("utf-8"))},
    }
    if query is not None:
        context["query"] = query
    if predicate is not None:
        context["predicate"] = predicate
    context = _with_parameters(context, parameters)
    return _state(
        normalized,
        decoded,
        context,
        limits,
        split_focus=False,
        rejections=_rejections,
        source_ref=_source_ref,
    )


def chunk_files(
    records: str | bytes | Mapping[str, Any],
    *,
    limits: StateLimits = StateLimits(),
    query: str | None = None,
    predicate: str | None = None,
    parameters: Mapping[str, str] | None = None,
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    if isinstance(records, Mapping):
        if "path" not in records or "content" not in records:
            raise ValueError("file input requires path and content")
        return chunk_file(
            records["path"],
            records["content"],
            limits=limits,
            query=query,
            predicate=predicate,
            parameters=parameters,
            _rejections=_rejections,
        )
    states: list[State] = []
    for line_number, byte_offset, line in _jsonl_lines(records):
        source_ref = f"stdin:byte={byte_offset},line={line_number}"
        if line is None:
            _input_error(
                _rejections,
                f"file JSONL line {line_number} is not valid UTF-8",
                source_ref,
            )
            continue
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _input_error(
                _rejections,
                f"file JSONL line {line_number} is invalid JSON: {exc.msg}",
                source_ref,
            )
            continue
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("content"), str)
        ):
            _input_error(
                _rejections,
                f"file JSONL line {line_number} requires string path and content",
                source_ref,
            )
            continue
        states.extend(
            chunk_file(
                record["path"],
                record["content"],
                limits=limits,
                query=query,
                predicate=predicate,
                parameters=parameters,
                _rejections=_rejections,
                _source_ref=source_ref,
            )
        )
    return states


def chunk_record(
    records: str | bytes | Mapping[str, Any],
    *,
    state_ref_field: str = "id",
    metadata_fields: Iterable[str] | None = None,
    query: str | None = None,
    predicate: str | None = None,
    parameters: Mapping[str, str] | None = None,
    limits: StateLimits = StateLimits(),
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    if isinstance(records, Mapping):
        values = [(records, None)]
    else:
        values = []
        for line_number, byte_offset, line in _jsonl_lines(records):
            source_ref = f"stdin:byte={byte_offset},line={line_number}"
            if line is None:
                _input_error(
                    _rejections,
                    f"record JSONL line {line_number} is not valid UTF-8",
                    source_ref,
                )
                continue
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                _input_error(
                    _rejections,
                    f"record JSONL line {line_number} is invalid JSON: {exc.msg}",
                    source_ref,
                )
                continue
            if not isinstance(value, dict):
                _input_error(
                    _rejections,
                    f"record JSONL line {line_number} must be an object",
                    source_ref,
                )
                continue
            values.append((value, source_ref))

    states: list[State] = []
    selected_fields = tuple(metadata_fields or ())
    for record_index, (record, source_ref) in enumerate(values, start=1):
        if state_ref_field not in record:
            _input_error(
                _rejections,
                "record is missing selected identity field "
                f"{state_ref_field!r}",
                source_ref or f"stdin:byte=0,line={record_index}",
            )
            continue
        identity = record[state_ref_field]
        if type(identity) not in (str, int) or (
            isinstance(identity, str) and not identity
        ):
            message = (
                f"record identity field {state_ref_field!r} must be a non-empty "
                "string or integer"
            )
            _input_error(
                _rejections,
                message,
                source_ref or f"stdin:byte=0,line={record_index}",
            )
            continue
        state_ref = str(identity)
        focus = _canonical_json(record)
        metadata = {key: record[key] for key in selected_fields if key in record}
        context: dict[str, Any] = {"unit": "record", "metadata": metadata}
        if query is not None:
            context["query"] = query
        if predicate is not None:
            context["predicate"] = predicate
        context = _with_parameters(context, parameters)
        states.extend(
            _state(
                state_ref,
                focus,
                context,
                limits,
                split_focus=False,
                rejections=_rejections,
                source_ref=source_ref,
            )
        )
    return states


def chunk_state(
    states: str | bytes | Mapping[str, Any],
    *,
    limits: StateLimits = StateLimits(),
    parameters: Mapping[str, str] | None = None,
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    """Read universal raw states from a finite JSONL stream."""

    values: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(states, Mapping):
        values.append(("stdin:byte=0,line=1", states))
    else:
        for line_number, byte_offset, line in _jsonl_lines(states):
            source_ref = f"stdin:byte={byte_offset},line={line_number}"
            if line is None:
                _input_error(
                    _rejections,
                    f"state JSONL line {line_number} is not valid UTF-8",
                    source_ref,
                )
                continue
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                _input_error(
                    _rejections,
                    f"state JSONL line {line_number} is invalid JSON: {exc.msg}",
                    source_ref,
                )
                continue
            if not isinstance(value, dict):
                _input_error(
                    _rejections,
                    f"state JSONL line {line_number} must be an object",
                    source_ref,
                )
                continue
            values.append((source_ref, value))

    formed: list[State] = []
    for source_ref, value in values:
        unknown = set(value) - {"state_ref", "focus", "context"}
        if unknown:
            _input_error(
                _rejections,
                "state contains unknown field(s): "
                + ", ".join(sorted(str(item) for item in unknown)),
                source_ref,
            )
            continue
        missing = [
            field
            for field in ("state_ref", "focus", "context")
            if field not in value
        ]
        if missing:
            _input_error(
                _rejections,
                f"state is missing required field(s): {', '.join(missing)}",
                source_ref,
            )
            continue
        state_ref = value["state_ref"]
        focus = value["focus"]
        context = value["context"]
        if not isinstance(state_ref, str) or not state_ref:
            _input_error(
                _rejections,
                "state_ref must be a non-empty string",
                source_ref,
            )
            continue
        if not isinstance(focus, str):
            _input_error(_rejections, "focus must be a string", source_ref)
            continue
        if not isinstance(context, dict):
            _input_error(_rejections, "context must be an object", source_ref)
            continue
        merged_context = _with_parameters(context, parameters)
        try:
            state = State(state_ref, focus, merged_context, source_ref=source_ref)
            validate_state(state, limits)
        except (StateLimitError, TypeError, ValueError) as exc:
            _input_error(_rejections, str(exc), source_ref)
            continue
        formed.append(state)
    return formed


@dataclass(frozen=True, slots=True)
class ChunkResult:
    formed: tuple[State, ...]
    admission: StateAdmission

    @property
    def discovered(self) -> int:
        return self.admission.discovered

    @property
    def admitted(self) -> tuple[State, ...]:
        return self.admission.admitted

    @property
    def states(self) -> tuple[State, ...]:
        return self.formed

    @property
    def judged(self) -> int:
        return self.admission.judged

    @property
    def rejections(self) -> tuple[StateRejection, ...]:
        return self.admission.rejections

    @property
    def rejected(self) -> tuple[StateRejection, ...]:
        return self.rejections

    @property
    def skipped_count(self) -> int:
        return self.admission.skipped_count

    @property
    def skipped(self) -> tuple[State, ...]:
        return self.admission.skipped


def chunk_input(
    by: str,
    value: str | bytes | Mapping[str, Any],
    *,
    max_chunks: int | None = None,
    limits: StateLimits = StateLimits(),
    **kwargs: Any,
) -> ChunkResult:
    functions = {
        "line": chunk_line,
        "para": chunk_para,
        "hunk": chunk_hunk,
        "file": chunk_files,
        "record": chunk_record,
        "state": chunk_state,
    }
    try:
        chunker = functions[by]
    except KeyError as exc:
        raise ValueError(f"unknown chunker {by!r}") from exc
    rejections: list[StateRejection] = []
    formed = tuple(chunker(value, limits=limits, _rejections=rejections, **kwargs))
    return ChunkResult(
        formed,
        admit_states(formed, max_chunks, rejections),
    )
