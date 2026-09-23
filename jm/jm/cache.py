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
) -> dict[str, Any]:
    """Build the exact section 6.1 cache-key object."""
    resolved_chunking = _json_value(chunking)
    if not isinstance(resolved_chunking, dict):
        raise TypeError("chunking must be an object")
    if limits is not None:
        if frozenset((*resolved_chunking, "limits")) not in _CHUNKING_SHAPES:
            raise ValueError("chunking must contain exactly the resolved fields")
        resolved_chunking["limits"] = _limits_dict(limits)
    elif frozenset(resolved_chunking) not in _CHUNKING_SHAPES:
        raise ValueError("chunking must contain exactly the resolved fields")
    resolved_chunking["limits"] = _limits_dict(resolved_chunking["limits"])

    resolved_state = state.payload if hasattr(state, "payload") else state
    return {
        "cache_schema": cache_schema,
        "model": model,
        "preset": preset,
        "preset_version": preset_version,
        "chunking": resolved_chunking,
        "question_battery": _json_value(questions),
        "state": _json_value(resolved_state),
    }


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
        if self.response.served_model is not None:
            result["served_model"] = self.response.served_model
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
        answers_root = self.root / "answers"
        if not answers_root.exists():
            return ()

        result: list[CacheEntry] = []
        seen: set[str] = set()
        for path in sorted(answers_root.glob("*/*/*.json")):
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
    if payload.get("cache_key") != expected_key:
        raise ValueError("cache key does not match path")
    preimage = payload.get("preimage")
    if not isinstance(preimage, Mapping) or cache_key(preimage) != expected_key:
        raise ValueError("cache preimage does not match key")
    schema = payload.get("cache_schema")
    if schema not in {LEGACY_CACHE_SCHEMA, CACHE_SCHEMA}:
        raise ValueError("unknown cache schema")
    if preimage.get("cache_schema") != schema:
        raise ValueError("cache schema does not match preimage")
    if payload.get("protocol_version") != schema:
        raise ValueError("unknown cache protocol")
    for field_name in ("model", "preset", "preset_version"):
        if field_name in payload and payload[field_name] != preimage.get(field_name):
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
    served_model = payload.get("served_model")
    if served_model is not None and not isinstance(served_model, str):
        raise ValueError("cache entry has an invalid served model")
    if served_model is not None:
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
