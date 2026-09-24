"""The jm-answer/v3 cache and its one-time v1/v2 invalidation boundary."""

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

CACHE_SCHEMA = "jm-answer/v3"
PROTOCOL_VERSION = CACHE_SCHEMA
LEGACY_CACHE_SCHEMAS = frozenset({"jm-answer/v1", "jm-answer/v2"})
LEGACY_CACHE_SCHEMA = "jm-answer/v1"
_MIGRATION_MARKER = ".jm-answer-v3-migrated"
_ENTRY_FIELDS = frozenset(
    {
        "cache_key",
        "wire_state",
        "battery_hash",
        "transport_identity",
        "provenance",
        "configured_model",
        "served_model",
        "usage",
        "created_at",
        "response",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {"preset", "preset_version", "battery_hash", "state_refs"}
)
_TRANSPORT_FIELDS = frozenset(
    {
        "endpoint",
        "ai-evaluation-model-specification-version",
        "ai-gateway-auth-method",
        "ai-gateway-protocol-version",
        "ai-model-id",
    }
)
_PROVIDER_OPTIONS = {"gateway": {"zeroDataRetention": True}}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def cache_key(envelope: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


compute_cache_key = cache_key


def battery_hash(battery: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(battery)).hexdigest()


def build_cache_envelope(
    *,
    wire_state: Mapping[str, Any],
    questions: Mapping[str, Any],
    model: str,
) -> dict[str, Any]:
    from .client import build_canonical_request

    request = build_canonical_request(wire_state, questions, model=model)
    return {
        "cache_schema": CACHE_SCHEMA,
        "wire_request": request.payload,
        "transport_identity": request.transport_identity,
    }


def build_cache_preimage(
    *,
    model: str,
    preset: str | None = None,
    preset_version: str | None = None,
    chunking: Mapping[str, Any] | None = None,
    questions: Mapping[str, Any],
    state: Mapping[str, Any] | Any,
    limits: Mapping[str, int] | Any | None = None,
    cache_schema: str = CACHE_SCHEMA,
    preset_schema: str | None = None,
    include_uid: bool = False,
) -> dict[str, Any]:
    """Build the v3 hash envelope.

    The preset arguments remain accepted for callers migrating from v2. They
    are intentionally absent from the returned envelope.
    """
    del preset, preset_version, chunking, limits, preset_schema
    if cache_schema != CACHE_SCHEMA:
        raise ValueError("cache_schema must be jm-answer/v3")
    resolved_state = state.payload if hasattr(state, "payload") else state
    if not isinstance(resolved_state, Mapping):
        raise TypeError("state must be an object")
    wire_state = _project_state(resolved_state, questions, include_uid=include_uid)
    return build_cache_envelope(
        wire_state=wire_state,
        questions=questions,
        model=model,
    )


def _project_state(
    state: Mapping[str, Any],
    questions: Mapping[str, Any],
    *,
    include_uid: bool,
) -> dict[str, Any]:
    focus = state.get("focus")
    context = state.get("context", {})
    if not isinstance(focus, str) or not isinstance(context, Mapping):
        raise TypeError("state must contain a string focus and object context")
    named = v3_context_keys(questions, include_uid=include_uid)
    return {
        "focus": focus,
        "context": {key: context[key] for key in sorted(named) if key in context},
    }


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
class BatteryRecord:
    preset: str
    preset_version: str
    battery_hash: str
    battery: Mapping[str, Any]
    effective: Mapping[str, Any]


class BatteryStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, preset: str, preset_version: str, digest: str) -> Path:
        scope = hashlib.sha256(
            canonical_json_bytes({"preset": preset, "preset_version": preset_version})
        ).hexdigest()
        return self.root / "batteries" / scope[:2] / f"{scope[2:]}-{digest[7:]}.json"

    def put(
        self,
        preset: str,
        preset_version: str,
        battery: Mapping[str, Any],
        effective: Mapping[str, Any] | None = None,
    ) -> BatteryRecord:
        copied = _json_value(battery)
        if not isinstance(copied, dict):
            raise ValueError("battery must be an object")
        digest = battery_hash(copied)
        record = {
            "preset": preset,
            "preset_version": preset_version,
            "battery_hash": digest,
            "battery": copied,
            "effective": _json_value(effective or {}),
        }
        path = self._path(preset, preset_version, digest)
        _atomic_write(path, record)
        return BatteryRecord(
            preset, preset_version, digest, copied, record["effective"]
        )

    def get(
        self,
        preset: str,
        preset_version: str,
        digest: str,
    ) -> BatteryRecord | None:
        path = self._path(preset, preset_version, digest)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                return None
            if set(payload) != {
                "preset",
                "preset_version",
                "battery_hash",
                "battery",
                "effective",
            }:
                return None
            battery = payload["battery"]
            if (
                payload["preset"] != preset
                or payload["preset_version"] != preset_version
                or payload["battery_hash"] != digest
                or not isinstance(battery, Mapping)
                or battery_hash(battery) != digest
                or not isinstance(payload["effective"], Mapping)
            ):
                return None
            return BatteryRecord(
                preset,
                preset_version,
                digest,
                dict(battery),
                dict(payload["effective"]),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key: str
    wire_state: Mapping[str, Any]
    battery_hash: str
    transport_identity: Mapping[str, Any]
    provenance: tuple[Mapping[str, Any], ...]
    configured_model: str
    served_model: str
    usage: Mapping[str, Any] | None
    created_at: str
    response: JudgeResponse

    @property
    def model(self) -> str:
        return self.configured_model

    @property
    def preset(self) -> str:
        return str(self.provenance[0]["preset"])

    @property
    def preset_version(self) -> str:
        return str(self.provenance[0]["preset_version"])

    def provenance_for(
        self, preset: str, preset_version: str, digest: str
    ) -> Mapping[str, Any] | None:
        return next(
            (
                group
                for group in self.provenance
                if group["preset"] == preset
                and group["preset_version"] == preset_version
                and group["battery_hash"] == digest
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "wire_state": _json_value(self.wire_state),
            "battery_hash": self.battery_hash,
            "transport_identity": _json_value(self.transport_identity),
            "provenance": _json_value(self.provenance),
            "configured_model": self.configured_model,
            "served_model": self.served_model,
            "usage": _json_value(self.usage),
            "created_at": self.created_at,
            "response": {
                "answers": {
                    question_id: answer_to_dict(answer)
                    for question_id, answer in self.response.answers.items()
                }
            },
        }


class CacheStore:
    """A strict v3 content-addressed answer and battery store."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        configured = os.environ.get("JM_CACHE_DIR")
        self.root = Path(root or configured or "~/.cache/jm").expanduser()
        self.batteries = BatteryStore(self.root)
        self.battery_store = self.batteries
        self._migrate_legacy_once()

    def _migrate_legacy_once(self) -> None:
        marker = self.root / _MIGRATION_MARKER
        if marker.exists():
            return
        answers = self.root / "answers"
        if not answers.exists():
            return
        for path in answers.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping) or set(payload) != _ENTRY_FIELDS:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
        _atomic_write(
            marker,
            {
                "cache_schema": CACHE_SCHEMA,
                "migrated_at": datetime.now(UTC).isoformat(),
            },
        )

    def path_for(self, key: str) -> Path:
        digest = _digest(key)
        return self.root / "answers" / digest[:2] / digest[2:4] / f"{digest}.json"

    def _battery_for_entry(self, payload: Mapping[str, Any]) -> BatteryRecord | None:
        provenance = payload.get("provenance")
        if not isinstance(provenance, list) or not provenance:
            return None
        first = provenance[0]
        if not isinstance(first, Mapping):
            return None
        preset = first.get("preset")
        version = first.get("preset_version")
        digest = payload.get("battery_hash")
        if not all(
            isinstance(value, str) and value for value in (preset, version, digest)
        ):
            return None
        return self.batteries.get(preset, version, digest)

    def get(
        self, key: str, questions: Mapping[str, Any] | None = None
    ) -> CacheEntry | None:
        try:
            path = self.path_for(key)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                return None
            battery_record = self._battery_for_entry(payload)
            if questions is not None:
                digest = battery_hash(questions)
                if digest != payload.get("battery_hash"):
                    return None
                battery = questions
            elif battery_record is not None:
                battery = battery_record.battery
            else:
                return None
            return _parse_entry(payload, key, battery)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    lookup = get

    def publish(
        self,
        wire_state: Mapping[str, Any],
        response: JudgeResponse,
        *,
        battery: Mapping[str, Any],
        preset: str,
        preset_version: str,
        configured_model: str,
        transport_identity: Mapping[str, Any],
        state_ref: str,
        effective_preset: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> CacheEntry:
        if not response.complete:
            raise ValueError("only complete responses can be cached")
        if not _valid_usage(response.usage if usage is None else usage):
            raise ValueError("cache entry has invalid usage")
        battery_record = self.batteries.put(
            preset, preset_version, battery, effective_preset
        )
        from .client import build_canonical_request

        request = build_canonical_request(
            wire_state, battery_record.battery, model=configured_model
        )
        if request.transport_identity != dict(transport_identity):
            raise ValueError("transport identity does not match request")
        envelope = {
            "cache_schema": CACHE_SCHEMA,
            "wire_request": request.payload,
            "transport_identity": request.transport_identity,
        }
        key = cache_key(envelope)
        path = self.path_for(key)
        existing = self.get(key)
        group = {
            "preset": preset,
            "preset_version": preset_version,
            "battery_hash": battery_record.battery_hash,
            "state_refs": [state_ref],
        }
        if existing is not None:
            entry = CacheEntry(
                existing.cache_key,
                existing.wire_state,
                existing.battery_hash,
                existing.transport_identity,
                _merge_provenance(existing.provenance, group),
                existing.configured_model,
                existing.served_model,
                existing.usage,
                existing.created_at,
                existing.response,
            )
        else:
            entry = CacheEntry(
                key,
                request.payload["state"],
                battery_record.battery_hash,
                request.transport_identity,
                (_freeze_group(group),),
                configured_model,
                response.served_model or "unknown",
                _json_value(response.usage if usage is None else usage),
                datetime.now(UTC).isoformat(),
                response,
            )
        _atomic_write(path, entry.to_dict())
        return entry

    store = publish

    def add_provenance(
        self,
        key: str,
        *,
        preset: str,
        preset_version: str,
        battery_hash: str,
        state_ref: str,
    ) -> CacheEntry | None:
        entry = self.get(key)
        if entry is None:
            return None
        updated = CacheEntry(
            entry.cache_key,
            entry.wire_state,
            entry.battery_hash,
            entry.transport_identity,
            _merge_provenance(
                entry.provenance,
                {
                    "preset": preset,
                    "preset_version": preset_version,
                    "battery_hash": battery_hash,
                    "state_refs": [state_ref],
                },
            ),
            entry.configured_model,
            entry.served_model,
            entry.usage,
            entry.created_at,
            entry.response,
        )
        _atomic_write(self.path_for(key), updated.to_dict())
        return updated

    def entries(self, preset: str | None = None) -> Iterator[CacheEntry]:
        answers_root = self.root / "answers"
        if not answers_root.exists():
            return
        for path in sorted(answers_root.glob("*/*/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    continue
                entry = self.get(str(payload["cache_key"]))
                if entry is None:
                    continue
                if preset is None or any(
                    group["preset"] == preset for group in entry.provenance
                ):
                    yield entry
            except (OSError, TypeError, KeyError, ValueError, json.JSONDecodeError):
                continue

    def calibration_entries(
        self,
        preset: str,
        preset_version: str | None = None,
        digest: str | None = None,
    ) -> tuple[CacheEntry, ...]:
        result: list[CacheEntry] = []
        for path in self._calibration_paths():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("cache entry must be an object")
                expected_key = payload.get("cache_key")
                if not isinstance(expected_key, str) or path != self.path_for(
                    expected_key
                ):
                    raise ValueError("cache key does not match path")
                entry = self.get(expected_key)
                if entry is None:
                    raise ValueError("invalid v3 cache entry")
                if any(
                    group["preset"] == preset
                    and (
                        preset_version is None
                        or group["preset_version"] == preset_version
                    )
                    and (digest is None or group["battery_hash"] == digest)
                    for group in entry.provenance
                ):
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

    def _calibration_paths(self) -> tuple[Path, ...]:
        root = self.root / "answers"
        if not root.exists():
            return ()
        paths: list[Path] = []
        try:
            first_level = sorted(root.iterdir())
        except OSError as exc:
            raise ValueError(
                f"cannot read calibration cache directory: {root}"
            ) from exc
        for first in first_level:
            if not first.is_dir() or not _is_shard(first.name):
                raise ValueError(f"unexpected path in calibration cache: {first}")
            try:
                second_level = sorted(first.iterdir())
            except OSError as exc:
                raise ValueError(
                    f"cannot read calibration cache directory: {first}"
                ) from exc
            for second in second_level:
                if not second.is_dir() or not _is_shard(second.name):
                    raise ValueError(f"unexpected path in calibration cache: {second}")
                try:
                    paths_in_shard = sorted(second.iterdir())
                except OSError as exc:
                    raise ValueError(
                        f"cannot read calibration cache directory: {second}"
                    ) from exc
                for path in paths_in_shard:
                    if not path.is_file() or not _is_digest_filename(path.name):
                        raise ValueError(
                            f"unexpected path in calibration cache: {path}"
                        )
                    paths.append(path)
        return tuple(paths)

    def clear(self, preset: str) -> int:
        removed = 0
        for path in tuple((self.root / "answers").glob("*/*/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                entry = self.get(str(payload.get("cache_key")))
                if entry is not None and any(
                    group["preset"] == preset for group in entry.provenance
                ):
                    path.unlink()
                    removed += 1
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return removed

    def export_triples(self, preset: str) -> Iterator[dict[str, Any]]:
        for entry in self.entries(preset):
            group = next(
                group for group in entry.provenance if group["preset"] == preset
            )
            battery_record = self.batteries.get(
                group["preset"], group["preset_version"], group["battery_hash"]
            )
            if battery_record is None:
                continue
            for question_id in sorted(entry.response.answers):
                yield {
                    "state": _json_value(entry.wire_state),
                    "question_id": question_id,
                    "question": battery_record.battery[question_id],
                    "answer": answer_to_dict(entry.response.answers[question_id]),
                    "model": entry.configured_model,
                    "preset": preset,
                    "preset_version": group["preset_version"],
                    "cache_key": entry.cache_key,
                }

    def export_jsonl(self, preset: str, stdout: TextIO) -> int:
        count = 0
        for triple in self.export_triples(preset):
            stdout.write(json.dumps(triple, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        stdout.flush()
        return count


AnswerCache = CacheStore
Cache = CacheStore


def _parse_entry(
    payload: Any,
    expected_key: str,
    battery: Mapping[str, Any],
) -> CacheEntry:
    if not isinstance(payload, Mapping) or set(payload) != _ENTRY_FIELDS:
        raise ValueError("cache entry has an invalid field set")
    if payload["cache_key"] != expected_key:
        raise ValueError("cache key does not match path")
    wire_state = payload["wire_state"]
    if (
        not isinstance(wire_state, Mapping)
        or set(wire_state) != {"focus", "context"}
        or not isinstance(wire_state["focus"], str)
        or not isinstance(wire_state["context"], Mapping)
    ):
        raise ValueError("cache entry has an invalid wire state")
    context = wire_state["context"]
    names_state_ref = "state_ref" in v3_context_keys(battery)
    state_ref = context.get("state_ref")
    if names_state_ref:
        if not isinstance(state_ref, str) or not state_ref:
            raise ValueError("cache entry has an invalid state reference")
    elif "state_ref" in context:
        raise ValueError("cache entry has an invalid state reference")
    digest = payload["battery_hash"]
    if not isinstance(digest, str) or battery_hash(battery) != digest:
        raise ValueError("cache entry has an invalid battery hash")
    transport = payload["transport_identity"]
    if not isinstance(transport, Mapping) or set(transport) != _TRANSPORT_FIELDS:
        raise ValueError("cache entry has an invalid transport identity")
    if any(not isinstance(value, str) or not value for value in transport.values()):
        raise ValueError("cache entry has an invalid transport identity")
    provenance = payload["provenance"]
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("cache entry has no provenance")
    groups: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group in provenance:
        if not isinstance(group, Mapping) or set(group) != _PROVENANCE_FIELDS:
            raise ValueError("cache entry has invalid provenance")
        key = (group["preset"], group["preset_version"], group["battery_hash"])
        refs = group["state_refs"]
        if (
            any(not isinstance(value, str) or not value for value in key)
            or not isinstance(refs, list)
            or not refs
            or any(not isinstance(ref, str) or not ref for ref in refs)
            or refs != sorted(set(refs))
            or key in seen
        ):
            raise ValueError("cache entry has invalid provenance")
        if not groups and key[2] != digest:
            raise ValueError("cache entry provenance has the wrong battery hash")
        seen.add(key)
        groups.append(_freeze_group(group))
    configured = payload["configured_model"]
    served = payload["served_model"]
    if any(not isinstance(value, str) or not value for value in (configured, served)):
        raise ValueError("cache entry has invalid model provenance")
    if transport["ai-model-id"] != configured:
        raise ValueError("cache entry model does not match transport identity")
    usage = payload["usage"]
    if not _valid_usage(usage):
        raise ValueError("cache entry has invalid usage")
    created_at = payload["created_at"]
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("cache entry has no creation time")
    response_payload = payload["response"]
    if not isinstance(response_payload, Mapping) or set(response_payload) != {
        "answers"
    }:
        raise ValueError("cache entry has an invalid response")
    response = parse_judge_response(response_payload, battery)
    if not response.complete or set(response.answers) != set(battery):
        raise ValueError("partial cache entry")
    from .client import build_canonical_request

    questions_wire = build_canonical_request(
        wire_state, battery, model=configured
    ).payload["questions"]
    envelope = {
        "cache_schema": CACHE_SCHEMA,
        "wire_request": {
            "providerOptions": _PROVIDER_OPTIONS,
            "state": wire_state,
            "questions": questions_wire,
        },
        "transport_identity": dict(transport),
    }
    if cache_key(envelope) != expected_key:
        raise ValueError("cache entry does not match its request")
    response = JudgeResponse(
        response.answers,
        response.missing_questions,
        served_model=served,
        usage=usage,
    )
    return CacheEntry(
        expected_key,
        dict(wire_state),
        digest,
        dict(transport),
        tuple(groups),
        configured,
        served,
        dict(usage) if isinstance(usage, Mapping) else None,
        created_at,
        response,
    )


def _merge_provenance(
    existing: tuple[Mapping[str, Any], ...],
    addition: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    key = (addition["preset"], addition["preset_version"], addition["battery_hash"])
    groups = [dict(group) for group in existing]
    for group in groups:
        if (group["preset"], group["preset_version"], group["battery_hash"]) == key:
            group["state_refs"] = sorted(
                set(group["state_refs"]) | set(addition["state_refs"])
            )
            return tuple(_freeze_group(item) for item in groups)
    groups.append(dict(addition))
    return tuple(_freeze_group(item) for item in groups)


def _freeze_group(group: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "preset": group["preset"],
        "preset_version": group["preset_version"],
        "battery_hash": group["battery_hash"],
        "state_refs": sorted(set(group["state_refs"])),
    }


def _valid_usage(usage: Any) -> bool:
    return usage is None or (
        isinstance(usage, Mapping)
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in usage.values()
        )
    )


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _is_digest_filename(name: str) -> bool:
    return (
        len(name) == 69
        and name.endswith(".json")
        and all(character in "0123456789abcdef" for character in name[:-5])
    )


def _is_shard(name: str) -> bool:
    return len(name) == 2 and all(character in "0123456789abcdef" for character in name)


def _digest(key: str) -> str:
    if not isinstance(key, str) or not key.startswith("sha256:"):
        raise ValueError("cache key must start with sha256:")
    digest = key.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("cache key has an invalid digest")
    return digest
