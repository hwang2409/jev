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


def decode_stdin(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _as_text(value: str | bytes) -> str:
    return decode_stdin(value) if isinstance(value, bytes) else value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    state = State(state_ref, focus, context)
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


def _jsonl_lines(value: str | bytes) -> Iterable[tuple[int, int, str]]:
    offset = 0
    if isinstance(value, bytes):
        lines = value.splitlines(keepends=True)
        for line_number, raw_line in enumerate(lines, start=1):
            yield line_number, offset, decode_stdin(raw_line)
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


def chunk_line(
    text: str | bytes,
    *,
    source: str = "stdin",
    adjacent_lines: int = 1,
    query: str | None = None,
    predicate: str | None = None,
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
    limits: StateLimits = StateLimits(),
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    decoded = _as_text(text)
    units = _paragraph_units(decoded)
    lines = decoded.splitlines()
    states: list[State] = []
    for index, (line_number, focus) in enumerate(units, start=1):
        start = max(0, index - 1 - adjacent_paragraphs)
        end = min(len(units), index + adjacent_paragraphs)
        surrounding = [
            paragraph
            for offset, (_, paragraph) in enumerate(units[start:end], start=start + 1)
            if offset != index
        ]
        context: dict[str, Any] = {
            "source": source,
            "unit": "para",
            "paragraph": index,
            "heading": _heading_before(lines, line_number),
            "surrounding": surrounding,
        }
        if query is not None:
            context["query"] = query
        if predicate is not None:
            context["predicate"] = predicate
        states.extend(
            _state(
                f"{source}#P{index}",
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


def chunk_hunk(
    diff: str | bytes,
    *,
    limits: StateLimits = StateLimits(),
    changed_test_paths: Iterable[str] | None = None,
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    lines = _as_text(diff).splitlines()
    file_path = "stdin"
    current_header: str | None = None
    current_body: list[str] = []
    hunks: list[tuple[str, str, list[str]]] = []
    pending_old_path: str | None = None
    old_remaining = 0
    new_remaining = 0
    for line in lines:
        if current_header is not None:
            if old_remaining or new_remaining:
                current_body.append(line)
                if line != r"\ No newline at end of file":
                    if line.startswith((" ", "-")):
                        old_remaining -= 1
                    if line.startswith((" ", "+")):
                        new_remaining -= 1
                continue
            hunks.append((file_path, current_header, current_body))
            current_header = None
            current_body = []
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
    if current_header is not None:
        hunks.append((file_path, current_header, current_body))

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
            _rejections=_rejections,
        )
    states: list[State] = []
    for line_number, byte_offset, line in _jsonl_lines(records):
        if not line.strip():
            continue
        source_ref = f"stdin:byte={byte_offset},line={line_number}"
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
    limits: StateLimits = StateLimits(),
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    if isinstance(records, Mapping):
        values = [(records, None)]
    else:
        values = []
        for line_number, byte_offset, line in _jsonl_lines(records):
            if not line.strip():
                continue
            source_ref = f"stdin:byte={byte_offset},line={line_number}"
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
