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


_HUNK_RE = re.compile(r"^@@\s+(.+?)\s+@@(?:\s.*)?$")


def _normalise_path(value: str) -> str:
    path = value.strip()
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return PurePosixPath(path).as_posix()


def _diff_path(line: str) -> str:
    path = line.split("\t", 1)[0].split(" ", 1)[0]
    return _normalise_path(path)


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
    for line in lines:
        if line.startswith("diff --git "):
            if current_header is not None:
                hunks.append((file_path, current_header, current_body))
                current_header = None
                current_body = []
            pending_old_path = None
        elif line.startswith("--- "):
            pending_old_path = _diff_path(line[4:])
        elif line.startswith("+++ "):
            if current_header is not None:
                hunks.append((file_path, current_header, current_body))
                current_header = None
                current_body = []
            new_path = _diff_path(line[4:])
            file_path = pending_old_path if new_path == "/dev/null" else new_path
            pending_old_path = None
        match = _HUNK_RE.match(line)
        if match:
            if current_header is not None:
                hunks.append((file_path, current_header, current_body))
            current_header = line
            current_body = []
        elif current_header is not None:
            current_body.append(line)
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
    _rejections: list[StateRejection] | None = None,
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
    return _state(
        normalized,
        decoded,
        context,
        limits,
        split_focus=False,
        rejections=_rejections,
    )


def chunk_files(
    records: str | bytes | Mapping[str, Any],
    *,
    limits: StateLimits = StateLimits(),
    _rejections: list[StateRejection] | None = None,
) -> list[State]:
    if isinstance(records, Mapping):
        if "path" not in records or "content" not in records:
            raise ValueError("file input requires path and content")
        return chunk_file(
            records["path"],
            records["content"],
            limits=limits,
            _rejections=_rejections,
        )
    states: list[State] = []
    for line_number, line in enumerate(_as_text(records).splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            not isinstance(record, dict)
            or "path" not in record
            or "content" not in record
        ):
            raise ValueError(f"file JSONL line {line_number} requires path and content")
        states.extend(
            chunk_file(
                record["path"],
                record["content"],
                limits=limits,
                _rejections=_rejections,
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
        values = [records]
    else:
        values = []
        for line_number, line in enumerate(_as_text(records).splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"record JSONL line {line_number} must be an object")
            values.append(value)

    states: list[State] = []
    seen_refs: set[str] = set()
    selected_fields = tuple(metadata_fields or ())
    for record_index, record in enumerate(values, start=1):
        if state_ref_field not in record:
            if _rejections is not None:
                _rejections.append(
                    StateRejection(
                        None,
                        "input_error",
                        "record is missing selected identity field "
                        f"{state_ref_field!r}",
                        f"stdin:byte=0,line={record_index}",
                    )
                )
                continue
            raise ValueError(
                f"record is missing selected identity field {state_ref_field!r}"
            )
        identity = record[state_ref_field]
        if type(identity) not in (str, int) or (
            isinstance(identity, str) and not identity
        ):
            message = (
                f"record identity field {state_ref_field!r} must be a non-empty "
                "string or integer"
            )
            if _rejections is not None:
                _rejections.append(
                    StateRejection(
                        None,
                        "input_error",
                        message,
                        f"stdin:byte=0,line={record_index}",
                    )
                )
                continue
            raise ValueError(message)
        state_ref = str(identity)
        if state_ref in seen_refs:
            message = f"duplicate record identity {state_ref!r}"
            if _rejections is not None:
                _rejections.append(
                    StateRejection(
                        state_ref,
                        "input_error",
                        message,
                        f"stdin:byte=0,line={record_index}",
                    )
                )
                continue
            raise ValueError(message)
        seen_refs.add(state_ref)
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
                source_ref=f"stdin:byte=0,line={record_index}",
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
