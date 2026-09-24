from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from .answers import JudgeResponse, answer_to_dict, parse_judge_response

LEGACY_CACHE_SCHEMA = "jm-answer/v1"
CACHE_SCHEMA = "jm-answer/v2"
PROTOCOL_VERSION = CACHE_SCHEMA
_CHUNKING_SHAPES = (
    frozenset({"by", "max_chunks", "limits"}),
    frozenset({"by", "context_paragraphs", "max_chunks", "limits"}),
    frozenset({"by", "context_lines", "max_chunks", "limits"}),
)
_CACHE_PREIMAGE_FIELDS = frozenset(
    {
        "cache_schema",
        "model",
        "preset",
        "preset_version",
        "chunking",
        "question_battery",
        "state",
    }
)
_CACHE_PREIMAGE_FIELDS_WITH_PRESET_SCHEMA = _CACHE_PREIMAGE_FIELDS | {
    "preset_schema"
}
_CACHE_ENTRY_REQUIRED_FIELDS = frozenset(
    {
        "cache_key",
        "cache_schema",
        "created_at",
        "model",
        "preset",
        "preset_version",
        "preimage",
        "answers",
        "protocol_version",
    }
)
_CACHE_ENTRY_OPTIONAL_FIELDS = frozenset({"usage", "served_model"})


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 representation used by cache keys."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def cache_key(preimage: Mapping[str, Any]) -> str:
    """Return the content address for one resolved answer request."""
    digest = hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()
    return f"sha256:{digest}"


compute_cache_key = cache_key


def build_cache_preimage(
    *,
    model: str,
    preset: str,
    preset_version: str,
    chunking: Mapping[str, Any],
    questions: Mapping[str, Any],
    state: Mapping[str, Any] | Any,
    limits: Mapping[str, int] | Any | None = None,
    cache_schema: str = CACHE_SCHEMA,
    preset_schema: str | None = None,
    include_uid: bool = False,
) -> dict[str, Any]:
    """Build the exact section 6.1 cache-key object."""
    resolved_chunking = _json_value(chunking)
    if not isinstance(resolved_chunking, dict):
        raise TypeError("chunking must be an object")
    if limits is not None:
        allowed_shapes = _CHUNKING_SHAPES
        if preset_schema == "jm.preset/v3":
            allowed_shapes = (*_CHUNKING_SHAPES, frozenset({"by", "limits"}))
        if frozenset((*resolved_chunking, "limits")) not in allowed_shapes:
            raise ValueError("chunking must contain exactly the resolved fields")
        resolved_chunking["limits"] = _limits_dict(limits)
    elif (
        frozenset(resolved_chunking) not in _CHUNKING_SHAPES
        and not (
            preset_schema == "jm.preset/v3"
            and frozenset(resolved_chunking) == frozenset({"by", "limits"})
        )
    ):
        raise ValueError("chunking must contain exactly the resolved fields")
    resolved_chunking["limits"] = _limits_dict(resolved_chunking["limits"])

    resolved_state = state.payload if hasattr(state, "payload") else state
    if preset_schema == "jm.preset/v3":
        resolved_state = _project_v3_state(resolved_state, questions, include_uid)
    preimage = {
        "cache_schema": cache_schema,
        "model": model,
        "preset": preset,
        "preset_version": preset_version,
        "chunking": resolved_chunking,
        "question_battery": _json_value(questions),
        "state": _json_value(resolved_state),
    }
    if preset_schema is not None:
        if not isinstance(preset_schema, str) or not preset_schema:
            raise ValueError("preset_schema must be a non-empty string")
        preimage["preset_schema"] = preset_schema
    return preimage


def _project_v3_state(
    state: Any, questions: Mapping[str, Any], include_uid: bool
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError("state must be an object")
    focus = state.get("focus")
    context = state.get("context", {})
    if not isinstance(context, Mapping):
        raise TypeError("state.context must be an object")
    named = v3_context_keys(questions, include_uid=include_uid)
    projected_context = {
        key: context[key]
        for key in sorted(named)
        if key in context
    }
    return {"focus": focus, "context": projected_context}


def v3_context_keys(
    questions: Mapping[str, Any], *, include_uid: bool = False
) -> frozenset[str]:
    named: set[str] = set()
    for question in questions.values():
        if not isinstance(question, Mapping):
            continue
        instructions = question.get("instructions")
        if not isinstance(instructions, Mapping):
            continue
        fields = instructions.get("state_fields")
        if not isinstance(fields, (list, tuple)):
            continue
        for field in fields:
            if isinstance(field, str) and field.startswith("context."):
                named.add(field.removeprefix("context."))
    if include_uid:
        named.add("uid")
    return frozenset(named)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key: str
    preimage: Mapping[str, Any]
    response: JudgeResponse
    usage: Any = None
    created_at: str = ""

    @property
    def model(self) -> str:
        return str(self.preimage["model"])

    @property
    def preset(self) -> str:
        return str(self.preimage["preset"])

    @property
    def preset_version(self) -> str:
        return str(self.preimage["preset_version"])

    @property
    def preset_schema(self) -> str | None:
        value = self.preimage.get("preset_schema")
        return str(value) if value is not None else None

    def to_dict(self) -> dict[str, Any]:
        schema = str(self.preimage.get("cache_schema", LEGACY_CACHE_SCHEMA))
        result: dict[str, Any] = {
            "cache_key": self.cache_key,
            "cache_schema": schema,
            "created_at": self.created_at,
            "model": self.model,
            "preset": self.preset,
            "preset_version": self.preset_version,
            "preimage": dict(self.preimage),
            "answers": {
                question_id: answer_to_dict(answer)
                for question_id, answer in self.response.answers.items()
            },
            "protocol_version": schema,
        }
        if self.usage is not None:
            result["usage"] = self.usage
        result["served_model"] = self.response.served_model or "unknown"
        return result


class CacheStore:
    """A local content-addressed store for complete typed answers."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        configured = os.environ.get("JM_CACHE_DIR")
        self.root = Path(root or configured or "~/.cache/jm").expanduser()

    def path_for(self, key: str) -> Path:
        digest = _digest(key)
        return self.root / "answers" / digest[:2] / digest[2:4] / f"{digest}.json"

    def get(
        self, key: str, questions: Mapping[str, Any] | None = None
    ) -> CacheEntry | None:
        try:
            path = self.path_for(key)
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _parse_entry(payload, key, questions)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    lookup = get

    def publish(
        self,
        preimage: Mapping[str, Any],
        response: JudgeResponse,
        *,
        usage: Any = None,
    ) -> CacheEntry:
        write_preimage = dict(preimage)
        write_preimage["cache_schema"] = CACHE_SCHEMA
        battery = write_preimage.get("question_battery")
        if not isinstance(battery, Mapping) or set(response.answers) != set(battery):
            raise ValueError("response answer IDs do not match question battery")
        if not response.complete:
            raise ValueError("only complete responses can be cached")
        if response.served_model is None:
            response = JudgeResponse(
                response.answers,
                response.missing_questions,
                served_model="unknown",
                usage=response.usage,
                latency_ms=response.latency_ms,
            )
        resolved_usage = response.usage if usage is None else usage
        key = cache_key(write_preimage)
        entry = CacheEntry(
            key,
            _json_value(write_preimage),
            response,
            _json_value(resolved_usage) if resolved_usage is not None else None,
            datetime.now(UTC).isoformat(),
        )
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(entry.to_dict(), stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return entry

    store = publish

    def entries(self, preset: str | None = None) -> Iterator[CacheEntry]:
        answers_root = self.root / "answers"
        if not answers_root.exists():
            return
        for path in sorted(answers_root.glob("*/*/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                entry = _parse_entry(payload, str(payload["cache_key"]))
            except (OSError, TypeError, KeyError, ValueError, json.JSONDecodeError):
                continue
            if preset is None or entry.preset == preset:
                yield entry

    def calibration_entries(self, preset: str) -> tuple[CacheEntry, ...]:
        """Read every valid case for calibration without skipping bad files."""
        result: list[CacheEntry] = []
        seen: set[str] = set()
        for path in self._calibration_paths():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("cache entry must be an object")
                expected_key = payload.get("cache_key")
                if not isinstance(expected_key, str):
                    raise ValueError("cache entry has no cache key")
                if path != self.path_for(expected_key):
                    raise ValueError("cache key does not match path")
                entry = _parse_entry(payload, expected_key)
                payload_preset = payload.get("preset")
                preimage_preset = entry.preimage.get("preset")
                if not (
                    isinstance(payload_preset, str)
                    and payload_preset
                    and isinstance(preimage_preset, str)
                    and preimage_preset
                    and payload_preset == preimage_preset
                ):
                    raise ValueError("cache entry has invalid preset metadata")
                if payload_preset != preset:
                    continue
                if entry.cache_key in seen:
                    raise ValueError("duplicate cache key")
                seen.add(entry.cache_key)
                result.append(entry)
            except (
                OSError,
                TypeError,
                KeyError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError(
                    f"invalid calibration cache entry {path}: {exc}"
                ) from exc
        return tuple(result)

    def _calibration_paths(self) -> tuple[Path, ...]:
        root_entries = _scan_directory(self.root, missing_ok=True)
        if not root_entries:
            return ()
        if len(root_entries) != 1 or root_entries[0].name != "answers":
            raise ValueError(f"unexpected path in calibration cache: {self.root}")

        answers_entry = root_entries[0]
        if not _is_directory(answers_entry):
            raise ValueError("calibration cache answers path is not a directory")
        first_level = _scan_directory(self.root / "answers")
        paths: list[Path] = []
        for first_entry in first_level:
            if not _is_directory(first_entry) or not _is_shard(first_entry.name):
                raise ValueError(
                    f"unexpected path in calibration cache: {first_entry.path}"
                )
            second_level = _scan_directory(Path(first_entry.path))
            for second_entry in second_level:
                if not _is_directory(second_entry) or not _is_shard(
                    second_entry.name
                ):
                    raise ValueError(
                        f"unexpected path in calibration cache: {second_entry.path}"
                    )
                for file_entry in _scan_directory(Path(second_entry.path)):
                    if not _is_file(file_entry) or not _is_digest_filename(
                        file_entry.name
                    ):
                        raise ValueError(
                            f"unexpected path in calibration cache: {file_entry.path}"
                        )
                    paths.append(Path(file_entry.path))
        return tuple(sorted(paths))

    read_calibration_entries = calibration_entries

    def clear(self, preset: str) -> int:
        removed = 0
        for path in sorted((self.root / "answers").glob("*/*/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("preset") != preset:
                    preimage = payload.get("preimage", {})
                    if (
                        not isinstance(preimage, Mapping)
                        or preimage.get("preset") != preset
                    ):
                        continue
                path.unlink()
                removed += 1
            except (OSError, TypeError, AttributeError, json.JSONDecodeError):
                continue
        return removed

    def export_triples(self, preset: str) -> Iterator[dict[str, Any]]:
        for entry in self.entries(preset):
            questions = entry.preimage["question_battery"]
            state = entry.preimage["state"]
            for question_id in sorted(entry.response.answers):
                yield {
                    "state": state,
                    "question_id": question_id,
                    "question": questions[question_id],
                    "answer": answer_to_dict(entry.response.answers[question_id]),
                    "model": entry.model,
                    "preset": entry.preset,
                    "preset_version": entry.preset_version,
                    "cache_key": entry.cache_key,
                }

    def export_jsonl(self, preset: str, stdout: TextIO) -> int:
        count = 0
        for triple in self.export_triples(preset):
            stdout.write(json.dumps(triple, ensure_ascii=False, sort_keys=True))
            stdout.write("\n")
            count += 1
        stdout.flush()
        return count


AnswerCache = CacheStore
Cache = CacheStore


def _parse_entry(
    payload: Any,
    expected_key: str,
    questions: Mapping[str, Any] | None = None,
) -> CacheEntry:
    if not isinstance(payload, Mapping):
        raise ValueError("cache entry must be an object")
    unknown_fields = set(payload) - (
        _CACHE_ENTRY_REQUIRED_FIELDS | _CACHE_ENTRY_OPTIONAL_FIELDS
    )
    if unknown_fields or not _CACHE_ENTRY_REQUIRED_FIELDS <= set(payload):
        raise ValueError("cache entry has an invalid field set")
    if payload.get("cache_key") != expected_key:
        raise ValueError("cache key does not match path")
    preimage = payload.get("preimage")
    if not isinstance(preimage, Mapping):
        raise ValueError("cache preimage must be an object")
    if set(preimage) not in {
        _CACHE_PREIMAGE_FIELDS,
        _CACHE_PREIMAGE_FIELDS_WITH_PRESET_SCHEMA,
    }:
        raise ValueError("cache preimage has an invalid field set")
    schema = payload.get("cache_schema")
    if schema not in {LEGACY_CACHE_SCHEMA, CACHE_SCHEMA}:
        raise ValueError("unknown cache schema")
    _validate_preimage(preimage, schema)
    if cache_key(preimage) != expected_key:
        raise ValueError("cache preimage does not match key")
    if payload["cache_schema"] != schema or preimage["cache_schema"] != schema:
        raise ValueError("cache schema does not match preimage")
    if payload.get("protocol_version") != schema:
        raise ValueError("unknown cache protocol")
    for field_name in ("model", "preset", "preset_version"):
        if payload[field_name] != preimage[field_name]:
            raise ValueError(f"cache {field_name} does not match preimage")
    battery = preimage.get("question_battery")
    if not isinstance(battery, Mapping):
        raise ValueError("cache entry has no question battery")
    expected_questions = questions or battery
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise ValueError("cache entry has no answers")
    response = parse_judge_response({"answers": raw_answers}, expected_questions)
    if not response.complete:
        raise ValueError("partial cache entry")
    usage = payload.get("usage")
    if usage is not None and not isinstance(usage, Mapping):
        raise ValueError("cache entry has invalid usage")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str):
        raise ValueError("cache entry has no creation time")
    served_model = payload.get("served_model", "unknown")
    if not isinstance(served_model, str) or not served_model:
        raise ValueError("cache entry has an invalid served model")
    response = JudgeResponse(
        response.answers,
        response.missing_questions,
        served_model=served_model,
    )
    response = JudgeResponse(
        response.answers,
        response.missing_questions,
        served_model=response.served_model,
        usage=usage,
    )
    return CacheEntry(expected_key, dict(preimage), response, usage, created_at)


def _validate_preimage(preimage: Mapping[str, Any], schema: str) -> None:
    if preimage["cache_schema"] != schema:
        raise ValueError("cache schema does not match preimage")
    for field_name in ("model", "preset", "preset_version"):
        value = preimage[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"cache preimage has an invalid {field_name}")
    if "preset_schema" in preimage and (
        not isinstance(preimage["preset_schema"], str)
        or not preimage["preset_schema"]
    ):
        raise ValueError("cache preimage has an invalid preset_schema")

    chunking = preimage["chunking"]
    is_v3 = preimage.get("preset_schema") == "jm.preset/v3"
    valid_chunking_shapes = (
        (*_CHUNKING_SHAPES, frozenset({"by", "limits"}))
        if is_v3
        else _CHUNKING_SHAPES
    )
    if (
        not isinstance(chunking, Mapping)
        or frozenset(chunking) not in valid_chunking_shapes
    ):
        raise ValueError("cache preimage has invalid chunking")
    if not isinstance(chunking["by"], str) or not chunking["by"]:
        raise ValueError("cache preimage has invalid chunking.by")
    for field_name in ("context_paragraphs", "context_lines", "max_chunks"):
        if field_name in chunking and not _nonnegative_int(chunking[field_name]):
            raise ValueError(f"cache preimage has invalid chunking.{field_name}")
    limits = chunking["limits"]
    if not isinstance(limits, Mapping) or set(limits) != {
        "focus_bytes",
        "context_field_bytes",
        "state_bytes",
    }:
        raise ValueError("cache preimage has invalid chunking limits")
    if any(not _positive_int(limits[name]) for name in limits):
        raise ValueError("cache preimage has invalid chunking limits")

    battery = preimage["question_battery"]
    if not isinstance(battery, Mapping) or any(
        not isinstance(question_id, str)
        or not isinstance(question, Mapping)
        or not isinstance(question.get("type"), str)
        for question_id, question in battery.items()
    ):
        raise ValueError("cache preimage has an invalid question battery")

    state = preimage["state"]
    if not isinstance(state, Mapping) or set(state) != {"focus", "context"}:
        raise ValueError("cache preimage has an invalid state")
    if not isinstance(state["focus"], str) or not isinstance(
        state["context"], Mapping
    ):
        raise ValueError("cache preimage has an invalid state")
    state_ref = state["context"].get("state_ref")
    if (
        not is_v3 or state_ref is not None
    ) and (not isinstance(state_ref, str) or not state_ref):
        raise ValueError("cache preimage has an invalid state reference")


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _scan_directory(
    path: Path, *, missing_ok: bool = False
) -> tuple[os.DirEntry[str], ...]:
    try:
        with os.scandir(path) as entries:
            return tuple(sorted(entries, key=lambda entry: entry.name))
    except FileNotFoundError:
        if missing_ok:
            return ()
        raise ValueError(f"calibration cache directory disappeared: {path}") from None
    except OSError as exc:
        raise ValueError(
            f"cannot read calibration cache directory {path}: {exc}"
        ) from exc


def _is_directory(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_dir(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(
            f"cannot inspect calibration cache path {entry.path}: {exc}"
        ) from exc


def _is_file(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_file(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(
            f"cannot inspect calibration cache path {entry.path}: {exc}"
        ) from exc


def _is_shard(name: str) -> bool:
    return len(name) == 2 and all(character in "0123456789abcdef" for character in name)


def _is_digest_filename(name: str) -> bool:
    return (
        len(name) == 69
        and name.endswith(".json")
        and all(character in "0123456789abcdef" for character in name[:-5])
    )


def _digest(key: str) -> str:
    if not isinstance(key, str) or not key.startswith("sha256:"):
        raise ValueError("cache key must start with sha256:")
    digest = key.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("cache key must contain a SHA-256 digest")
    return digest


def _limits_dict(limits: Mapping[str, int] | Any) -> dict[str, int]:
    fields = ("focus_bytes", "context_field_bytes", "state_bytes")
    if isinstance(limits, Mapping):
        if set(limits) != set(fields):
            raise ValueError("limits must contain exactly the resolved fields")
        return {
            name: int(limits[name]) for name in fields
        }
    return {
        name: int(getattr(limits, name)) for name in fields
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    return value
