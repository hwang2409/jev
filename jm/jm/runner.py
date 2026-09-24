from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import re
import string
import time as _time
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol, TextIO
from uuid import uuid4

from ._transport import _GATEWAY_MODEL as GATEWAY_MODEL
from .answers import (
    CacheStatus,
    CanonicalRecord,
    ChoiceAnswer,
    CoverageReason,
    CoverageRecord,
    Diagnostic,
    DiagnosticRecord,
    ErrorDetail,
    ErrorRecord,
    ErrorResponse,
    JudgeResponse,
    NoulAnswer,
    PartialResultRecord,
    RecordMeta,
    ResultRecord,
    ScoreAnswer,
    SkipSummary,
    TypedResponse,
)
from .cache import (
    CACHE_SCHEMA,
    CacheStore,
    build_cache_preimage,
    cache_key,
    v3_context_keys,
)
from .gates import (
    GateResult,
    Policy,
    PolicyError,
    compile_policy,
    evaluate_gate,
    evaluate_policy,
    parse_policy,
)
from .presets import (
    SCHEMA_V2,
    SCHEMA_V3,
    Preset,
    PresetUsageError,
    resolve_preset,
    validate_preset,
)


def _canonical_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )


def _value_bytes(value: Any) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return _canonical_bytes(value)


@dataclass(frozen=True, slots=True)
class StateLimits:
    focus_bytes: int = 16_384
    context_field_bytes: int = 4_096
    state_bytes: int = 32_768

    def __post_init__(self) -> None:
        if min(self.focus_bytes, self.context_field_bytes, self.state_bytes) <= 0:
            raise ValueError("state limits must be positive")


@dataclass(frozen=True, slots=True)
class State:
    state_ref: str
    focus: str
    context: Mapping[str, Any] = field(default_factory=dict)
    source_ref: str | None = None
    wire_context_keys: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.state_ref:
            raise ValueError("state_ref must not be empty")
        if not isinstance(self.focus, str):
            raise TypeError("focus must be a string")
        context = dict(self.context)
        existing_ref = context.get("state_ref")
        if existing_ref is not None and existing_ref != self.state_ref:
            raise ValueError("context state_ref does not match state_ref")
        context["state_ref"] = self.state_ref
        object.__setattr__(self, "context", context)

    @property
    def payload(self) -> dict[str, Any]:
        return {"focus": self.focus, "context": dict(self.context)}

    def to_payload(self) -> dict[str, Any]:
        return self.payload

    @property
    def api_payload(self) -> dict[str, Any]:
        if self.wire_context_keys is None:
            return self.payload
        return {
            "focus": self.focus,
            "context": {
                key: self.context[key]
                for key in sorted(self.wire_context_keys)
                if key in self.context
            },
        }


class JudgeFn(Protocol):
    def __call__(
        self, state: State, questions: Mapping[str, Any], model: str
    ) -> TypedResponse: ...


class StateLimitError(ValueError):
    """A formed state exceeds a configured byte limit."""


class StateInputError(ValueError):
    """A formed input cannot be judged safely."""

    exit_code = 2


class ConfigurationError(ValueError):
    """A preset or judgment option is invalid."""

    exit_code = 64


class InputError(ValueError):
    """Input data cannot be judged safely."""

    exit_code = 2

    def __init__(self, message: str, *, source_ref: str | None = None) -> None:
        super().__init__(message)
        self.source_ref = source_ref


@dataclass(frozen=True, slots=True)
class FormationEvent:
    kind: Literal["skip", "input_error"]
    reason: Literal["scan_cap", "context_limit", "prefiltered", "input_error"]
    message: str
    state_ref: str | None
    source_ref: str | None
    boundary: str | None

    def __post_init__(self) -> None:
        if self.kind not in {"skip", "input_error"}:
            raise ConfigurationError(f"unknown formation event kind: {self.kind}")
        if self.reason not in {
            "scan_cap",
            "context_limit",
            "prefiltered",
            "input_error",
        }:
            raise ConfigurationError(f"unknown formation event reason: {self.reason}")
        if self.kind == "skip":
            if self.state_ref is None or self.reason == "input_error":
                raise ConfigurationError("skip events require a state reference")
        elif self.reason != "input_error":
            raise ConfigurationError(
                "input_error events require reason input_error"
            )
        elif self.source_ref is None or self.state_ref is not None:
            raise ConfigurationError(
                "input_error events require a source reference only"
            )


@dataclass(frozen=True, slots=True)
class FormationReport:
    events: tuple[FormationEvent, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if any(not isinstance(event, FormationEvent) for event in self.events):
            raise ConfigurationError("formation events must be typed records")
        if any(not isinstance(item, Diagnostic) for item in self.diagnostics):
            raise ConfigurationError("formation diagnostics must be typed records")


@dataclass(frozen=True, slots=True)
class ResultFilter:
    kind: Literal["keep", "policy"]
    question_id: str | None = None
    operator: Literal["<", "<=", ">", ">=", "==", "!="] | None = None
    threshold: float | int | None = None
    expression: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "keep":
            if (
                not self.question_id
                or self.operator != ">="
                or isinstance(self.threshold, bool)
                or not isinstance(self.threshold, (int, float))
                or self.expression is not None
            ):
                raise ConfigurationError(
                    "keep filters require question_id, operator >=, and threshold"
                )
        elif self.kind == "policy":
            if not self.expression or any(
                value is not None
                for value in (self.question_id, self.operator, self.threshold)
            ):
                raise ConfigurationError(
                    "policy filters require expression and no threshold fields"
                )
        else:
            raise ConfigurationError(f"unknown result filter kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class InputSidecar:
    values: Mapping[str, Mapping[str, Any] | Path]


@dataclass(frozen=True, slots=True)
class EmitResult:
    records_written: int
    records_suppressed: int
    diagnostics_written: int
    coverage: CoverageRecord | None
    passthrough: bool
    broken_pipe: bool


@dataclass(frozen=True, slots=True)
class StateRejection:
    state_ref: str | None
    reason: str
    message: str
    source_ref: str | None = None
    boundary: str | None = None

    def __post_init__(self) -> None:
        if self.reason not in {
            "scan_cap",
            "context_limit",
            "input_error",
            "prefiltered",
        }:
            raise ValueError(f"unknown rejection reason: {self.reason}")
        if self.reason == "input_error":
            if not self.source_ref:
                raise ValueError("input errors require a source reference")
        elif self.state_ref is None:
            raise ValueError("formed-state rejections require a state reference")


def validate_state(state: State, limits: StateLimits = StateLimits()) -> None:
    identity_size = len(state.state_ref.encode("utf-8"))
    if identity_size > limits.context_field_bytes:
        raise StateLimitError(
            "context field 'state_ref' exceeds "
            f"context_field_bytes={limits.context_field_bytes}: {identity_size} bytes"
        )
    focus_size = _value_bytes(state.focus)
    if focus_size > limits.focus_bytes:
        raise StateLimitError(
            f"focus exceeds focus_bytes={limits.focus_bytes}: {focus_size} bytes"
        )

    for name, value in state.context.items():
        size = _value_bytes(value)
        if size > limits.context_field_bytes:
            raise StateLimitError(
                f"context field {name!r} exceeds "
                f"context_field_bytes={limits.context_field_bytes}: {size} bytes"
            )

    state_size = _canonical_bytes(state.payload)
    if state_size > limits.state_bytes:
        raise StateLimitError(
            f"state exceeds state_bytes={limits.state_bytes}: {state_size} bytes"
        )


@dataclass(frozen=True, slots=True)
class StateAdmission:
    formed: tuple[State, ...]
    admitted: tuple[State, ...]
    skipped: tuple[State, ...]
    max_chunks: int | None = None
    rejections: tuple[StateRejection, ...] = ()
    skip_rejections: tuple[StateRejection, ...] = ()

    @property
    def discovered(self) -> int:
        return len(self.formed) + sum(
            rejection.state_ref is not None and rejection.reason != "input_error"
            for rejection in self.rejections
        )

    @property
    def judged(self) -> int:
        return len(self.admitted)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped) + sum(
            rejection.state_ref is not None
            and rejection.reason in {"scan_cap", "context_limit"}
            for rejection in self.rejections
        )

    @property
    def skip_boundary(self) -> str | None:
        if not self.skipped:
            return None
        if self.skip_rejections:
            return self.skip_rejections[0].boundary
        if self.max_chunks is None:
            return None
        return f"max_chunks={self.max_chunks}"

    @property
    def input_errors(self) -> tuple[StateRejection, ...]:
        return tuple(
            rejection
            for rejection in self.rejections
            if rejection.reason == "input_error"
        )


def admit_states(
    states: Sequence[State],
    max_chunks: int | None = None,
    rejections: Sequence[StateRejection] = (),
) -> StateAdmission:
    if (
        max_chunks is not None
        and (isinstance(max_chunks, bool) or not isinstance(max_chunks, int)
             or max_chunks < 0)
    ):
        raise ValueError("max_chunks must be non-negative")
    formed = tuple(states)
    if max_chunks is None:
        admitted = formed
    else:
        admitted = formed[:max_chunks]
    return StateAdmission(
        formed,
        admitted,
        formed[len(admitted) :],
        max_chunks,
        tuple(rejections),
    )


def _admit_prefiltered_states(
    states: Sequence[State],
    *,
    top: int,
    max_chunks: int | None,
    rejections: Sequence[StateRejection],
    query: str,
    fields: Sequence[str],
) -> StateAdmission:
    ranked = bm25_rank(states, query, fields)
    shortlist = ranked[:top]
    admitted = shortlist if max_chunks is None else shortlist[:max_chunks]
    admitted_refs = {state.state_ref for state in admitted}
    skip_rejections: list[StateRejection] = []
    for state in ranked:
        if state.state_ref in admitted_refs:
            continue
        if state not in shortlist:
            skip_rejections.append(
                StateRejection(
                    state.state_ref,
                    "prefiltered",
                    "prefilter skipped before visit",
                    boundary=f"prefilter=bm25,top={top}",
                )
            )
        else:
            skip_rejections.append(
                StateRejection(
                    state.state_ref,
                    "scan_cap",
                    "scan cap reached before visit",
                    boundary=f"max_chunks={max_chunks}",
                )
            )
    return StateAdmission(
        tuple(states),
        tuple(admitted),
        tuple(state for state in states if state.state_ref not in admitted_refs),
        max_chunks,
        tuple(rejections),
        tuple(skip_rejections),
    )


@dataclass(frozen=True, slots=True)
class RunStats:
    discovered: int
    judged: int
    emitted: int
    skipped: int
    failed: int
    consistency_attempted_calls: int = 0
    consistency_cache_hits: int = 0
    consistency_live_calls: int = 0
    consistency_usage: Mapping[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    admission: StateAdmission
    responses: tuple[TypedResponse, ...]
    stats: RunStats
    records: tuple[CanonicalRecord, ...] = ()
    coverage_reasons: tuple[CoverageReason, ...] = ()
    exit_code: int = 0
    gate_result: GateResult | None = None
    broken_pipe: bool = False


_DEFAULT_MODEL = GATEWAY_MODEL

_V3_CONTEXT_KEYS: dict[str, frozenset[str] | None] = {
    "state": None,
    "file": frozenset({"language", "metadata", "path"}),
    "line": frozenset({"line", "source", "surrounding", "unit"}),
    "para": frozenset(
        {"heading", "paragraph", "source", "surrounding", "unit"}
    ),
    "record": frozenset({"metadata", "unit"}),
    "hunk": frozenset(
        {"changed_tests", "file", "hunk_header", "surrounding", "unit"}
    ),
}


class _Unset:
    __slots__ = ()


_UNSET = _Unset()


class Runner:
    def __init__(
        self,
        judge_fn: JudgeFn,
        model: str | _Unset = _UNSET,
        limits: StateLimits | _Unset = _UNSET,
        *,
        preset: Preset | str | PathLike[str] | None = None,
    ) -> None:
        if judge_fn is None:
            raise TypeError("judge_fn is required")
        self.judge_fn = judge_fn
        self._model_supplied = model is not _UNSET
        self._limits_supplied = limits is not _UNSET
        self.model = _DEFAULT_MODEL if model is _UNSET else model
        self.limits = StateLimits() if limits is _UNSET else limits
        self.preset = self._load_preset(preset)

    def judge(self, state: State, questions: Mapping[str, Any]) -> TypedResponse:
        validate_state(state, self.limits)
        return self.judge_fn(state, questions, self.model)

    def admit(
        self,
        states: Sequence[State],
        max_chunks: int | None = None,
        rejections: Sequence[StateRejection] = (),
    ) -> StateAdmission:
        for state in states:
            validate_state(state, self.limits)
        return admit_states(states, max_chunks, rejections)
    def run(
        self,
        states: Sequence[State],
        questions: Mapping[str, Any] | None | _Unset = _UNSET,
        max_chunks: int | None | _Unset = _UNSET,
        *,
        preset: Preset | str | PathLike[str] | None = None,
        preset_version: str | None | _Unset = _UNSET,
        chunker: str | _Unset = _UNSET,
        cache: CacheStatus = "not_applicable",
        cache_store: CacheStore | None = None,
        chunking: Mapping[str, Any] | None | _Unset = _UNSET,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        output_format: str = "jsonl",
        concurrency: int = 4,
        result_filter: Callable[[ResultRecord], bool] | None = None,
        rejections: Sequence[StateRejection] = (),
        policy: str | Policy | None = None,
        require_states: int = 1,
        prefilter: Mapping[str, Any] | None = None,
        prefilter_warning: bool = False,
        consistency: int | None = None,
        consistency_sigma: float = 2.0,
    ) -> RunResult:
        loaded_preset = self._load_preset(self.preset if preset is None else preset)
        if loaded_preset is None:
            if questions is _UNSET or questions is None:
                raise TypeError("questions or preset is required")
            runtime_chunker = "unknown" if chunker is _UNSET else chunker
            runtime_max_chunks = None if max_chunks is _UNSET else max_chunks
            runtime_chunking = (
                dict(chunking)
                if chunking is not _UNSET and chunking is not None
                else {
                    "by": runtime_chunker,
                    "max_chunks": runtime_max_chunks
                    if runtime_max_chunks is not None
                    else 512,
                    "limits": {
                        "focus_bytes": self.limits.focus_bytes,
                        "context_field_bytes": self.limits.context_field_bytes,
                        "state_bytes": self.limits.state_bytes,
                    },
                }
            )
            runtime_limits = self.limits
            runtime_chunking.setdefault(
                "limits",
                {
                    "focus_bytes": runtime_limits.focus_bytes,
                    "context_field_bytes": runtime_limits.context_field_bytes,
                    "state_bytes": runtime_limits.state_bytes,
                },
            )
            loaded_preset = Preset(
                {
                    "schema": "jm.preset/v1",
                    "name": str(preset) if preset is not None else "jm",
                    "version": "1" if preset_version is _UNSET else preset_version,
                    "model": self.model,
                    "chunking": runtime_chunking,
                    "compatible_chunkers": [runtime_chunker],
                    "questions": questions,
                    "thresholds": {},
                    "output": {
                        "default_format": "jsonl",
                        "pretty_template": None,
                        "fields": ["record_type", "state_ref", "answers", "meta"],
                    },
                },
                Path("<runtime>"),
            )
        else:
            if questions is not _UNSET:
                raise PresetUsageError(
                    "questions cannot be supplied with a preset; use the preset's "
                    "questions"
                )
            runtime_limits = StateLimits(**loaded_preset.chunking["limits"])
            if self._model_supplied and self.model != loaded_preset.model:
                raise PresetUsageError(
                    f"model {self.model!r} conflicts with preset {loaded_preset.name!r}"
                )
            if self._limits_supplied and self.limits != runtime_limits:
                raise PresetUsageError(
                    f"limits conflict with preset {loaded_preset.name!r}"
                )
            if preset_version is not _UNSET and preset_version != loaded_preset.version:
                raise PresetUsageError(
                    f"preset_version {preset_version!r} conflicts with preset "
                    f"{loaded_preset.name!r}"
                )
            if max_chunks is not _UNSET and max_chunks != loaded_preset.chunking.get(
                "max_chunks"
            ):
                raise PresetUsageError(
                    f"max_chunks {max_chunks!r} conflicts with preset "
                    f"{loaded_preset.name!r}"
                )
            runtime_chunker = loaded_preset.effective_chunker(
                None if chunker is _UNSET else chunker
            )
            runtime_chunking = dict(loaded_preset.chunking)
            runtime_chunking["by"] = runtime_chunker
            if chunking is not _UNSET:
                if chunking is None:
                    raise PresetUsageError(
                        f"chunking conflicts with preset {loaded_preset.name!r}"
                    )
                self._reject_chunking_conflicts(
                    chunking, runtime_chunking, loaded_preset.name
                )
            runtime_chunking.setdefault(
                "limits",
                {
                    "focus_bytes": runtime_limits.focus_bytes,
                    "context_field_bytes": runtime_limits.context_field_bytes,
                    "state_bytes": runtime_limits.state_bytes,
                },
            )
            if runtime_chunker != loaded_preset.default_chunker or (
                runtime_chunking != loaded_preset.chunking
            ):
                data = dict(loaded_preset.data)
                data["chunking"] = runtime_chunking
                loaded_preset = Preset(data, loaded_preset.path)
        runtime_max_chunks = (
            max_chunks
            if max_chunks is not _UNSET
            else loaded_preset.chunking.get("max_chunks")
        )
        if runtime_max_chunks is not None and (
            isinstance(runtime_max_chunks, bool)
            or not isinstance(runtime_max_chunks, int)
            or runtime_max_chunks < 0
        ):
            raise ValueError("max_chunks must be non-negative")
        if policy is not None and prefilter is not None:
            raise PresetUsageError("gate does not support prefiltering")
        _validate_consistency(consistency, consistency_sigma, loaded_preset.questions)
        if output_format not in {"jsonl", "pretty"}:
            raise ValueError("output format must be jsonl or pretty")
        capture = _PipelineCapture()
        outcome = _run_pipeline(
            loaded_preset,
            states,
            rejections=rejections,
            max_chunks=runtime_max_chunks,
            prefilter=prefilter,
            include_prefilter_warning=prefilter_warning,
            cache_store=cache_store,
            concurrency=concurrency,
            judge_fn=self.judge_fn,
            consistency=consistency,
            consistency_sigma=consistency_sigma,
            output_format=output_format,
            jsonl_stream=stdout,
            pretty_stream=stderr,
            result_filter=(
                result_filter if isinstance(result_filter, ResultFilter) else None
            ),
            legacy_filter=(
                result_filter if not isinstance(result_filter, ResultFilter) else None
            ),
            pretty_template=loaded_preset.data["output"]["pretty_template"],
            policy=policy,
            require_states=require_states,
            capture=capture,
        )
        if (
            cache_store is not None
            and any(
                isinstance(record, ErrorRecord)
                and record.error.kind == "malformed_answer"
                for record in capture.records
            )
        ):
            raise ValueError("malformed answer")
        return _run_result(
            outcome,
            capture,
            consistency=consistency,
            require_states=require_states,
        )

    def run_gate(
        self,
        states: Sequence[State],
        policy: str | Policy,
        *,
        require_states: int = 1,
        **kwargs: Any,
    ) -> RunResult:
        """Run a finite judgment and apply a typed failure-condition policy."""
        return self.run(
            states,
            policy=policy,
            require_states=require_states,
            **kwargs,
        )

    @staticmethod
    def _load_preset(
        preset: Preset | str | PathLike[str] | None,
    ) -> Preset | None:
        if preset is None:
            return None
        if isinstance(preset, Preset):
            return Preset(validate_preset(preset.data), preset.path)
        if isinstance(preset, PathLike) or "/" in preset:
            return resolve_preset("preset", explicit_path=preset)
        return resolve_preset(preset)

    @staticmethod
    def _reject_chunking_conflicts(
        requested: Mapping[str, Any],
        resolved: Mapping[str, Any],
        preset_name: str,
    ) -> None:
        for key, value in requested.items():
            if key not in resolved:
                raise PresetUsageError(
                    f"chunking.{key} conflicts with preset {preset_name!r}"
                )
            expected = resolved[key]
            if isinstance(value, Mapping):
                if not isinstance(expected, Mapping):
                    raise PresetUsageError(
                        f"chunking.{key} conflicts with preset {preset_name!r}"
                    )
                Runner._reject_chunking_conflicts(value, expected, preset_name)
            elif value != expected:
                raise PresetUsageError(
                    f"chunking.{key} conflicts with preset {preset_name!r}"
                )

    def run_jsonl(
        self,
        states: Sequence[State],
        questions: Mapping[str, Any] | None | _Unset = _UNSET,
        stdout: TextIO | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Run a finite judgment and write only canonical JSONL to stdout."""
        if stdout is None:
            raise TypeError("stdout is required")
        kwargs["stdout"] = stdout
        return self.run(states, questions, **kwargs)


@dataclass(slots=True)
class _PipelineCapture:
    records: list[CanonicalRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PipelineOutcome:
    admission: StateAdmission
    emitted: EmitResult
    gate_result: GateResult | None


def _capture_records(
    records: Iterable[CanonicalRecord], capture: _PipelineCapture
) -> Iterator[CanonicalRecord]:
    for record in records:
        capture.records.append(record)
        yield record


def _legacy_filter_records(
    records: Iterable[CanonicalRecord],
    result_filter: Callable[[ResultRecord], bool] | None,
) -> Iterator[CanonicalRecord]:
    for record in records:
        if (
            result_filter is not None
            and isinstance(record, ResultRecord)
            and not result_filter(record)
        ):
            continue
        yield record


def _gate_decision(
    policy: Policy,
    coverage: CoverageRecord,
    records: Iterable[CanonicalRecord],
    *,
    required_states: int,
    consistency_sigma: float,
) -> GateResult:
    gated_records = tuple(
        record for record in records if isinstance(record, ResultRecord)
    )
    return evaluate_gate(
        policy,
        gated_records,
        judged_states=coverage.coverage_counts["judged"],
        coverage_reasons=coverage.coverage_reasons,
        required_states=required_states,
        consistency_sigma=consistency_sigma,
    )


def _run_pipeline(
    preset: Preset,
    states: Sequence[State],
    *,
    rejections: Sequence[StateRejection] = (),
    max_chunks: int | None = None,
    prefilter: Mapping[str, Any] | None = None,
    include_prefilter_warning: bool = False,
    cache_store: CacheStore | None = None,
    concurrency: int = 4,
    judge_fn: JudgeFn | None = None,
    consistency: int | None = None,
    consistency_sigma: float = 2.0,
    output_format: str = "jsonl",
    jsonl_stream: TextIO | None = None,
    pretty_stream: TextIO | None = None,
    result_filter: ResultFilter | None = None,
    legacy_filter: Callable[[ResultRecord], bool] | None = None,
    pretty_template: str | None = None,
    policy: str | Policy | None = None,
    require_states: int = 1,
    formation_diagnostics: Sequence[Diagnostic] = (),
    empty_input_error: bool = True,
    capture: _PipelineCapture | None = None,
) -> _PipelineOutcome:
    compiled_policy = compile_policy(policy, preset) if policy is not None else None
    admission = admit_for_judgment(
        tuple(states),
        tuple(rejections),
        max_chunks=max_chunks,
        prefilter=prefilter,
    )
    formation = formation_report(
        admission,
        include_prefilter_warning=include_prefilter_warning,
        diagnostics=formation_diagnostics,
        empty_input_error=empty_input_error,
    )
    records: Iterable[CanonicalRecord] = _judge_core(
        preset,
        admission.admitted,
        formation_report=formation,
        cache_store=cache_store,
        concurrency=concurrency,
        judge_fn=judge_fn,
        consistency=consistency,
        consistency_sigma=consistency_sigma,
        validation_states=tuple(states),
    )
    if capture is not None:
        records = _capture_records(records, capture)
    gated_records: Iterable[CanonicalRecord] | None = None
    if compiled_policy is not None:
        records, gated_records = itertools.tee(records)
    emitted = _emit_core(
        _legacy_filter_records(records, legacy_filter),
        format=output_format,  # type: ignore[arg-type]
        jsonl_stream=jsonl_stream or io.StringIO(),
        pretty_stream=pretty_stream or io.StringIO(),
        result_filter=result_filter,
        pretty_template=pretty_template,
    )
    gate_result = None
    if compiled_policy is not None and not emitted.broken_pipe:
        if emitted.coverage is None or gated_records is None:
            raise ConfigurationError("judgment did not produce coverage")
        gate_result = _gate_decision(
            compiled_policy,
            emitted.coverage,
            gated_records,
            required_states=require_states,
            consistency_sigma=consistency_sigma,
        )
    return _PipelineOutcome(admission, emitted, gate_result)


def _run_result(
    outcome: _PipelineOutcome,
    capture: _PipelineCapture,
    *,
    consistency: int | None,
    require_states: int,
) -> RunResult:
    if outcome.emitted.coverage is None:
        return RunResult(
            outcome.admission,
            (),
            RunStats(0, 0, 0, outcome.admission.skipped_count, 0),
            tuple(capture.records),
            (),
            0,
            None,
            outcome.emitted.broken_pipe,
        )
    coverage = outcome.emitted.coverage
    responses = tuple(
        JudgeResponse(record.answers, usage=record.usage, latency_ms=record.latency_ms)
        for record in capture.records
        if isinstance(record, ResultRecord)
    )
    consistency_usage: dict[str, int | float] = {}
    for record in capture.records:
        if isinstance(record, ResultRecord):
            _add_usage(consistency_usage, record.usage)
    stats = RunStats(
        discovered=coverage.coverage_counts["discovered"],
        judged=coverage.coverage_counts["judged"],
        emitted=coverage.coverage_counts["emitted"],
        skipped=coverage.coverage_counts["skipped"],
        failed=coverage.coverage_counts["failed"],
        consistency_attempted_calls=(
            len(outcome.admission.admitted) * consistency
            if consistency is not None
            else 0
        ),
        consistency_usage=consistency_usage if consistency is not None else {},
    )
    exit_code = (
        outcome.gate_result.exit_code
        if outcome.gate_result is not None
        else 2
        if coverage.coverage == "partial"
        else 0
    )
    return RunResult(
        outcome.admission,
        responses,
        stats,
        tuple(capture.records),
        coverage.coverage_reasons,
        exit_code,
        outcome.gate_result,
        outcome.emitted.broken_pipe,
    )


def _judge_core(
    preset: Preset | str,
    states: Iterable[State],
    *,
    formation_report: FormationReport,
    cache_store: CacheStore | None = None,
    concurrency: int = 4,
    judge_fn: JudgeFn | None = None,
    consistency: int | None = None,
    consistency_sigma: float = 2.0,
    validation_states: Iterable[State] | None = None,
) -> Iterator[CanonicalRecord]:
    """Yield typed judgment records without writing to process streams."""

    loaded_preset = _public_preset(preset)
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency <= 0
    ):
        raise ConfigurationError("concurrency must be a positive integer")
    if not isinstance(formation_report, FormationReport):
        raise ConfigurationError("formation_report must be a FormationReport")

    # This function is intentionally a generator. The body starts only when
    # the caller consumes the iterator, which keeps construction side-effect free.
    state_values = tuple(states)
    validated_states = (
        state_values if validation_states is None else tuple(validation_states)
    )
    runtime_limits = StateLimits(**loaded_preset.chunking["limits"])
    seen_refs: set[str] = set()
    for state in validated_states:
        if not isinstance(state, State):
            raise InputError("states must contain State values")
        try:
            validate_state(state, runtime_limits)
        except (StateLimitError, TypeError, ValueError) as exc:
            raise InputError(str(exc)) from exc
        if state.state_ref in seen_refs:
            raise InputError(
                f"duplicate state reference {state.state_ref!r}",
                source_ref=state.source_ref,
            )
        seen_refs.add(state.state_ref)

    active_judge = judge_fn
    client = None
    if active_judge is None:
        from .client import JevClient

        client = JevClient()
        active_judge = client
    observer_target = active_judge
    if not callable(active_judge):
        async_evaluate = getattr(active_judge, "evaluate_async", None)
        if not callable(async_evaluate):
            raise ConfigurationError("judge_fn must be callable")

        def sync_evaluate(
            state: State, questions: Mapping[str, Any], model: str
        ) -> TypedResponse:
            import asyncio

            try:
                return asyncio.run(
                    async_evaluate(state, questions, model=model)
                )
            except TypeError:
                return asyncio.run(async_evaluate(state, questions))

        active_judge = sync_evaluate

    runtime_name = loaded_preset.name
    runtime_version = loaded_preset.version
    runtime_model = loaded_preset.model
    runtime_chunker = loaded_preset.default_chunker
    runtime_schema = (
        loaded_preset.schema
        if loaded_preset.schema in {SCHEMA_V2, SCHEMA_V3}
        else None
    )
    runtime_questions = loaded_preset.questions
    runtime_chunking = dict(loaded_preset.chunking)

    def cache_preimage(state: State) -> dict[str, Any]:
        return build_cache_preimage(
            model=runtime_model,
            preset=runtime_name,
            preset_version=runtime_version,
            chunking=runtime_chunking,
            questions=runtime_questions,
            state=state,
            cache_schema=CACHE_SCHEMA,
            preset_schema=runtime_schema,
            include_uid=consistency is not None,
        )

    def request_state(state: State) -> State:
        if runtime_schema != SCHEMA_V3:
            return state
        return State(
            state.state_ref,
            state.focus,
            state.context,
            source_ref=state.source_ref,
            wire_context_keys=v3_context_keys(
                runtime_questions, include_uid=consistency is not None
            ),
        )

    if cache_store is not None and validated_states:
        cache_preimage(validated_states[0])
    try:
        _validate_consistency(consistency, consistency_sigma, runtime_questions)
    except PresetUsageError as exc:
        raise ConfigurationError(str(exc)) from exc
    base_meta = RecordMeta(
        runtime_name,
        runtime_version,
        runtime_model,
        runtime_chunker,
        "not_applicable",
        preset_schema=runtime_schema,
    )
    diagnostics = list(formation_report.diagnostics)
    if runtime_schema == SCHEMA_V3:
        available = _V3_CONTEXT_KEYS[runtime_chunker]
        if available is not None:
            declared = set(loaded_preset.declared_parameters)
            for question_id, question in runtime_questions.items():
                for field in question["instructions"]["state_fields"]:
                    if not field.startswith("context."):
                        continue
                    key = field.removeprefix("context.")
                    if key in declared or key in available:
                        continue
                    diagnostics.append(
                        Diagnostic(
                            "warning",
                            "state_field_unavailable",
                            f"question '{question_id}' references unavailable field "
                            f"'{field}' for chunker '{runtime_chunker}'",
                            path=(
                                f"questions.{question_id}.instructions.state_fields"
                            ),
                        )
                    )
    effective_concurrency = min(concurrency, 8)
    if concurrency > 8:
        diagnostics.append(
            Diagnostic(
                "warning",
                "concurrency_capped",
                f"requested={concurrency} effective={effective_concurrency} cap=8",
            )
        )

    state_records: list[CanonicalRecord] = []
    formation_records = _formation_records(formation_report.events, base_meta)
    reasons: set[str] = {
        event.reason for event in formation_report.events
    }
    failed = 0

    consistency_usage: dict[str, int | float] = {}
    consistency_cache_hits = 0
    consistency_live_calls = 0
    consistency_stats_lock = Lock()
    concurrency_lock = Lock()
    current_concurrency = effective_concurrency
    consecutive_503 = 0
    last_503_at: float | None = None
    concurrency_diagnostics: list[DiagnosticRecord] = []

    def observe_response(status_code: int) -> None:
        nonlocal consecutive_503, current_concurrency, last_503_at
        with concurrency_lock:
            now = _time.monotonic()
            if status_code == 503:
                if consecutive_503 == 0:
                    last_503_at = now
                consecutive_503 += 1
                if consecutive_503 >= 2:
                    reduced = max(1, current_concurrency // 2)
                    if reduced < current_concurrency:
                        current_concurrency = reduced
                        concurrency_diagnostics.append(
                            DiagnosticRecord(
                                Diagnostic(
                                    "warning",
                                    "concurrency_backoff",
                                    "status=503 consecutive=2 "
                                    f"effective={current_concurrency}",
                                )
                            )
                        )
                    consecutive_503 = 0
                return
            if (
                last_503_at is not None
                and current_concurrency < effective_concurrency
                and now - last_503_at >= 60.0
            ):
                current_concurrency += 1
                concurrency_diagnostics.append(
                    DiagnosticRecord(
                        Diagnostic(
                            "info",
                            "concurrency_restored",
                            f"clean_seconds=60 effective={current_concurrency}",
                        )
                    )
                )
                last_503_at = now

    def take_concurrency_diagnostics() -> tuple[DiagnosticRecord, ...]:
        with concurrency_lock:
            records = tuple(concurrency_diagnostics)
            del concurrency_diagnostics[:]
            return records

    def records_for_state(
        state: State,
    ) -> ResultRecord | PartialResultRecord | ErrorRecord:
        def one_call(call_state: State) -> tuple[TypedResponse, bool]:
            preimage = None
            if cache_store is not None:
                preimage = cache_preimage(call_state)
                cached = cache_store.get(cache_key(preimage), runtime_questions)
                if cached is not None:
                    nonlocal consistency_cache_hits
                    if consistency is not None:
                        with consistency_stats_lock:
                            consistency_cache_hits += 1
                    return cached.response, True
            response = _call_public_judge(
                active_judge,
                request_state(call_state),
                runtime_questions,
                runtime_model,
            )
            if (
                isinstance(response, JudgeResponse)
                and response.complete
                and not _complete_for_questions(response, runtime_questions)
            ):
                response = ErrorResponse("malformed answer")
            nonlocal consistency_live_calls
            if consistency is not None:
                with consistency_stats_lock:
                    consistency_live_calls += 1
            if (
                cache_store is not None
                and preimage is not None
                and isinstance(response, JudgeResponse)
                and response.complete
            ):
                try:
                    cache_store.publish(preimage, response, usage=response.usage)
                except (OSError, TypeError, ValueError):
                    pass
            return response, False

        cache_state = "not_applicable"
        if consistency is None:
            response, cache_hit = one_call(state)
            cache_state = (
                "hit"
                if cache_hit
                else "miss"
                if cache_store
                else "not_applicable"
            )
        else:
            responses: list[JudgeResponse] = []
            failure: ErrorResponse | None = None
            hits = 0
            for _ in range(consistency):
                response, cache_hit = one_call(_repeat_state(state))
                hits += int(cache_hit)
                if isinstance(response, JudgeResponse):
                    _add_usage(consistency_usage, response.usage)
                if failure is not None:
                    continue
                if not isinstance(response, JudgeResponse):
                    failure = ErrorResponse("consistency repeat failed")
                elif not _complete_for_questions(response, runtime_questions):
                    failure = ErrorResponse(
                        "consistency repeat returned incomplete answers"
                    )
                else:
                    responses.append(response)
            cache_state = (
                "hit" if hits == consistency else "miss"
            ) if cache_store is not None else "not_applicable"
            if failure is not None:
                response = failure
            else:
                try:
                    response = _aggregate_responses(
                        responses, runtime_questions, consistency
                    )
                except (TypeError, ValueError):
                    response = ErrorResponse(
                        "consistency repeat returned malformed answers"
                    )
        meta = replace(base_meta, cache=cache_state)
        if isinstance(response, JudgeResponse) and response.served_model:
            meta = replace(meta, served_model=response.served_model)
        return _response_record(state.state_ref, response, meta)

    try:
        for diagnostic in diagnostics:
            yield DiagnosticRecord(diagnostic)

        set_response_observer = getattr(observer_target, "set_response_observer", None)
        uses_response_observer = callable(set_response_observer)
        if uses_response_observer:
            set_response_observer(observe_response)
        executor: ThreadPoolExecutor | None = None
        try:
            if state_values:
                executor = ThreadPoolExecutor(max_workers=effective_concurrency)
                futures: dict[Any, State] = {}
                next_state = 0

                def submit_available() -> None:
                    nonlocal next_state
                    with concurrency_lock:
                        limit = current_concurrency
                    while next_state < len(state_values) and len(futures) < limit:
                        state = state_values[next_state]
                        next_state += 1
                        futures[executor.submit(records_for_state, state)] = state

                submit_available()
                while futures:
                    future = next(as_completed(tuple(futures)))
                    state = futures.pop(future)
                    try:
                        record = future.result()
                    except Exception:
                        record = _response_record(
                            state.state_ref,
                            ErrorResponse("request failed"),
                            replace(base_meta, cache="not_applicable"),
                        )
                    state_records.append(record)
                    if isinstance(record, (ErrorRecord, PartialResultRecord)):
                        failed += 1
                    if isinstance(record, ErrorRecord):
                        reasons.add(record.error.kind)
                    elif isinstance(record, PartialResultRecord):
                        reasons.add("partial_answer")
                    if not uses_response_observer:
                        status_code = (
                            record.error.http_status
                            if isinstance(record, ErrorRecord)
                            and record.error.http_status is not None
                            else 200
                        )
                        observe_response(status_code)
                    for diagnostic in take_concurrency_diagnostics():
                        yield diagnostic
                    yield record
                    submit_available()
                if not uses_response_observer:
                    observe_response(200)
                    for diagnostic in take_concurrency_diagnostics():
                        yield diagnostic
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            if uses_response_observer:
                set_response_observer(None)

        yield from formation_records
        if consistency is not None:
            yield DiagnosticRecord(
                Diagnostic(
                    "info",
                    "consistency",
                    f"{len(state_values)} states * {consistency} = "
                    f"{len(state_values) * consistency} attempted calls; "
                    f"cache hits: {consistency_cache_hits}; "
                    f"live calls: {consistency_live_calls}; "
                    "total normalized token usage: "
                    f"{json.dumps(consistency_usage, sort_keys=True)}",
                )
            )

        skipped = sum(event.kind == "skip" for event in formation_report.events)
        discovered = len(state_values) + skipped
        judged = len(state_records)
        ordered_reasons = tuple(
            reason
            for reason in (
                "prefiltered",
                "scan_cap",
                "input_error",
                "context_limit",
                "api_error",
                "malformed_answer",
                "partial_answer",
            )
            if reason in reasons
        )
        coverage = "partial" if ordered_reasons else "complete"
        coverage_meta = replace(base_meta, cache="not_applicable")
        yield CoverageRecord(
            coverage=coverage,
            coverage_counts={
                "discovered": discovered,
                "judged": judged,
                "emitted": judged,
                "skipped": skipped,
                "failed": failed,
            },
            coverage_reasons=ordered_reasons,
            meta=coverage_meta,
        )
    finally:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close):
                    close()


def judge(
    preset: Preset | str,
    states: Iterable[State],
    *,
    formation_report: FormationReport,
    cache_store: CacheStore | None = None,
    concurrency: int = 4,
    judge_fn: JudgeFn | None = None,
) -> Iterator[CanonicalRecord]:
    return _judge_core(
        preset,
        states,
        formation_report=formation_report,
        cache_store=cache_store,
        concurrency=concurrency,
        judge_fn=judge_fn,
    )


def judge_async(
    preset: Preset | str,
    states: Iterable[State],
    *,
    formation_report: FormationReport,
    cache_store: CacheStore | None = None,
    concurrency: int = 4,
    judge_fn: JudgeFn | None = None,
) -> AsyncIterator[CanonicalRecord]:
    """Adapt the synchronous judgment iterator to async callers."""

    async def stream() -> AsyncIterator[CanonicalRecord]:
        import asyncio
        import inspect
        import queue
        import threading

        active_judge_fn = judge_fn
        if active_judge_fn is not None and inspect.iscoroutinefunction(
            active_judge_fn
        ):
            async_fn = active_judge_fn

            def sync_judge(
                state: State, questions: Mapping[str, Any], model: str
            ) -> TypedResponse:
                return asyncio.run(async_fn(state, questions, model))

            active_judge_fn = sync_judge

        sentinel = object()
        records: queue.Queue[object] = queue.Queue()

        def produce() -> None:
            try:
                for record in judge(
                    preset,
                    states,
                    formation_report=formation_report,
                    cache_store=cache_store,
                    concurrency=concurrency,
                    judge_fn=active_judge_fn,
                ):
                    records.put(record)
            except BaseException as exc:
                records.put(exc)
            finally:
                records.put(sentinel)

        threading.Thread(target=produce, daemon=True).start()
        while True:
            item = await asyncio.to_thread(records.get)
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]

    return stream()


def _emit_core(
    records: Iterable[CanonicalRecord],
    *,
    format: Literal["jsonl", "pretty"] = "jsonl",
    emit_mode: Literal["judgment", "input"] = "judgment",
    jsonl_stream: TextIO,
    pretty_stream: TextIO,
    output_path: Path | None = None,
    result_filter: ResultFilter | None = None,
    input_sidecar: InputSidecar | None = None,
    pretty_template: str | None = None,
) -> EmitResult:
    """Render canonical records. This is the only judgment output writer."""

    if format not in {"jsonl", "pretty"}:
        raise ConfigurationError("format must be jsonl or pretty")
    if emit_mode not in {"judgment", "input"}:
        raise ConfigurationError("emit_mode must be judgment or input")
    if emit_mode == "input" and input_sidecar is None:
        raise ConfigurationError("input_sidecar is required for input emission")
    if result_filter is not None and not isinstance(result_filter, ResultFilter):
        raise ConfigurationError("result_filter must be a ResultFilter")
    compiled_filter_policy = None
    if result_filter is not None and result_filter.kind == "policy":
        try:
            compiled_filter_policy = parse_policy(result_filter.expression or "")
        except PolicyError as exc:
            raise ConfigurationError(str(exc)) from exc

    sink = jsonl_stream
    close_sink = False
    if output_path is not None:
        sink = output_path.open("w", encoding="utf-8")
        close_sink = True
    records_written = 0
    records_suppressed = 0
    diagnostics_written = 0
    coverage: CoverageRecord | None = None
    pending_diagnostics: list[Diagnostic] = []
    has_prefilter_warning = False
    try:
        for record in records:
            try:
                if isinstance(record, DiagnosticRecord):
                    pending_diagnostics.append(record.diagnostic)
                    has_prefilter_warning |= (
                        record.diagnostic.code == "prefilter_recall"
                    )
                    continue
                if isinstance(record, CoverageRecord):
                    coverage = record
                visible = _record_visible(record, result_filter, compiled_filter_policy)
                if isinstance(record, ResultRecord) and not visible:
                    records_suppressed += 1
                    continue
                if emit_mode == "input" and isinstance(record, ResultRecord):
                    assert input_sidecar is not None
                    value = input_sidecar.values.get(record.state_ref)
                    if value is None:
                        raise ConfigurationError(
                            f"input sidecar has no state {record.state_ref!r}"
                        )
                    payload: Any = str(value) if isinstance(value, Path) else value
                    record_sink = sink
                    record_sink.write(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=isinstance(payload, Mapping),
                            separators=(",", ":")
                            if isinstance(payload, Mapping)
                            else None,
                        )
                        + "\n"
                    )
                else:
                    record_sink = (
                        pretty_stream
                        if emit_mode == "input"
                        and isinstance(record, (ErrorRecord, CoverageRecord))
                        else sink
                    )
                    record_sink.write(
                        json.dumps(
                            record.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                record_sink.flush()
                for diagnostic in pending_diagnostics:
                    _emit_diagnostic(diagnostic, pretty_stream)
                    diagnostics_written += 1
                pending_diagnostics.clear()
                records_written += 1
                if (
                    isinstance(record, ErrorRecord)
                    and record.state_ref is None
                    and record.error.kind != "prefiltered"
                ):
                    pretty_stream.write(f"jm: warning: {record.error.message}\n")
                    pretty_stream.flush()
                if (
                    isinstance(record, CoverageRecord)
                    and record.coverage == "partial"
                    and not has_prefilter_warning
                ):
                    pretty_stream.write(
                        "jm: warning: results are partial; coverage reasons: "
                        f"{', '.join(record.coverage_reasons)}\n"
                    )
                    pretty_stream.flush()
                if (
                    format == "pretty"
                    and isinstance(record, ResultRecord)
                    and emit_mode == "judgment"
                ):
                    emit_pretty(record, pretty_stream, pretty_template)
            except BrokenPipeError:
                try:
                    sink.close()
                except OSError:
                    pass
                return EmitResult(
                    records_written,
                    records_suppressed,
                    diagnostics_written,
                    None,
                    emit_mode == "input",
                    True,
                )
    finally:
        if close_sink:
            sink.close()
    for diagnostic in pending_diagnostics:
        _emit_diagnostic(diagnostic, pretty_stream)
        diagnostics_written += 1
    return EmitResult(
        records_written,
        records_suppressed,
        diagnostics_written,
        coverage,
        emit_mode == "input",
        False,
    )


def emit(
    records: Iterable[CanonicalRecord],
    *,
    format: Literal["jsonl", "pretty"] = "jsonl",
    emit_mode: Literal["judgment", "input"] = "judgment",
    jsonl_stream: TextIO,
    pretty_stream: TextIO,
    output_path: Path | None = None,
    result_filter: ResultFilter | None = None,
    input_sidecar: InputSidecar | None = None,
) -> EmitResult:
    return _emit_core(
        records,
        format=format,
        emit_mode=emit_mode,
        jsonl_stream=jsonl_stream,
        pretty_stream=pretty_stream,
        output_path=output_path,
        result_filter=result_filter,
        input_sidecar=input_sidecar,
    )


def _public_preset(preset: Preset | str) -> Preset:
    try:
        if isinstance(preset, Preset):
            if preset.path == Path("<runtime>"):
                return preset
            return Preset(validate_preset(preset.data), preset.path)
        if not isinstance(preset, str):
            raise ConfigurationError("preset must be a Preset or name")
        return Runner._load_preset(preset)  # type: ignore[return-value]
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc


def _call_public_judge(
    judge_fn: JudgeFn,
    state: State,
    questions: Mapping[str, Any],
    model: str,
) -> TypedResponse:
    try:
        response = judge_fn(state, questions, model)
    except Exception as exc:
        message = getattr(exc, "message", None)
        status = getattr(exc, "http_status", None)
        return ErrorResponse(
            message if isinstance(message, str) and message else "request failed",
            http_status=status if isinstance(status, int) else None,
        )
    if isinstance(response, (JudgeResponse, ErrorResponse)):
        return response
    return ErrorResponse("malformed answer")


def _formation_records(
    events: Sequence[FormationEvent], meta: RecordMeta
) -> tuple[ErrorRecord, ...]:
    grouped: dict[tuple[str, str], list[FormationEvent]] = {}
    order: list[tuple[str, str] | FormationEvent] = []
    for event in events:
        if event.kind == "input_error":
            order.append(event)
            continue
        key = (event.reason, event.boundary or event.reason)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(event)
    error_meta = replace(meta, cache="not_applicable")
    result: list[ErrorRecord] = []
    for item in order:
        if isinstance(item, FormationEvent):
            result.append(
                ErrorRecord(
                    None,
                    ErrorDetail("input_error", item.message),
                    error_meta,
                    source_ref=item.source_ref,
                )
            )
            continue
        reason, boundary = item
        grouped_events = grouped[item]
        result.append(
            ErrorRecord(
                None,
                ErrorDetail(
                    reason,  # type: ignore[arg-type]
                    grouped_events[0].message,
                    skip_summary=SkipSummary(
                        boundary,
                        len(grouped_events),
                        tuple(
                            event.state_ref
                            for event in grouped_events
                            if event.state_ref is not None
                        ),
                    ),
                ),
                error_meta,
            )
        )
    return tuple(result)


def admit_for_judgment(
    states: tuple[State, ...],
    rejections: tuple[StateRejection, ...],
    *,
    max_chunks: int | None,
    prefilter: Mapping[str, object] | None,
) -> StateAdmission:
    if prefilter is None:
        return admit_states(states, max_chunks, rejections)
    _validate_runtime_prefilter(prefilter)
    return _admit_prefiltered_states(
        states,
        top=prefilter["top"],  # type: ignore[arg-type]
        max_chunks=max_chunks,
        rejections=rejections,
        query=prefilter["query"],  # type: ignore[arg-type]
        fields=prefilter["fields"],  # type: ignore[arg-type]
    )


def formation_report(
    admission: StateAdmission,
    *,
    include_prefilter_warning: bool,
    diagnostics: Sequence[Diagnostic] = (),
    empty_input_error: bool = True,
) -> FormationReport:
    events: list[FormationEvent] = []
    for rejection in admission.rejections:
        if rejection.reason == "input_error":
            events.append(
                FormationEvent(
                    "input_error",
                    "input_error",
                    rejection.message,
                    None,
                    rejection.source_ref,
                    rejection.boundary,
                )
            )
        elif rejection.state_ref is not None:
            events.append(
                FormationEvent(
                    "skip",
                    rejection.reason,
                    rejection.message,
                    rejection.state_ref,
                    rejection.source_ref,
                    rejection.boundary,
                )
            )
    if admission.skip_rejections:
        for rejection in admission.skip_rejections:
            events.append(
                FormationEvent(
                    "skip",
                    rejection.reason,
                    rejection.message,
                    rejection.state_ref,
                    rejection.source_ref,
                    rejection.boundary,
                )
            )
    elif admission.skipped and admission.max_chunks is not None:
        events.extend(
            FormationEvent(
                "skip",
                "scan_cap",
                "scan cap reached before visit",
                state.state_ref,
                None,
                f"max_chunks={admission.max_chunks}",
            )
            for state in admission.skipped
        )
    result_diagnostics = list(diagnostics)
    if include_prefilter_warning:
        count = sum(
            rejection.reason == "prefiltered"
            for rejection in admission.skip_rejections
        )
        if count:
            result_diagnostics.append(
                Diagnostic(
                    "warning",
                    "prefilter_recall",
                    f"BM25 prefilter skipped {count} of {admission.discovered} states; "
                    "recall is bounded by the shortlist; rerun without "
                    "--prefilter for full recall",
                )
            )
    if empty_input_error and not admission.formed and not events:
        events.append(
            FormationEvent(
                "input_error",
                "input_error",
                "input is empty",
                None,
                "stdin:byte=0,line=1",
                None,
            )
        )
    return FormationReport(tuple(events), tuple(result_diagnostics))


def _record_visible(
    record: CanonicalRecord,
    result_filter: ResultFilter | None,
    policy: Policy | None = None,
) -> bool:
    if not isinstance(record, ResultRecord) or result_filter is None:
        return True
    if result_filter.kind == "policy":
        if policy is None:
            return False
        try:
            return evaluate_policy(policy, [record])
        except PolicyError as exc:
            raise ConfigurationError(str(exc)) from exc
    answer = record.answers.get(result_filter.question_id or "")
    value = getattr(answer, "noul", None)
    if value is None:
        value = getattr(answer, "score", None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return value >= result_filter.threshold  # type: ignore[operator]


def _emit_diagnostic(diagnostic: Diagnostic, stream: TextIO) -> None:
    if diagnostic.code == "prefilter_recall":
        stream.write(f"jm: warning: {diagnostic.message}\n")
    elif diagnostic.code == "consistency":
        stream.write(f"jm: consistency: {diagnostic.message}\n")
    else:
        stream.write(
            "jm: diagnostic: "
            f"code={diagnostic.code} severity={diagnostic.severity} "
            f"{diagnostic.message}\n"
        )
    stream.flush()


def _validate_consistency(
    consistency: int | None,
    sigma: float,
    questions: Mapping[str, Any],
) -> None:
    if any(
        "context.uid" in _question_state_fields(question)
        for question in questions.values()
    ):
        raise PresetUsageError("question state_fields must not refer to context.uid")
    if consistency is not None and (
        isinstance(consistency, bool)
        or not isinstance(consistency, int)
        or consistency < 2
    ):
        raise PresetUsageError("--consistency must be an integer of at least 2")
    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(float(sigma))
        or sigma < 0
    ):
        raise PresetUsageError("--consistency-sigma must be finite and non-negative")
    if consistency is None and sigma != 2.0:
        raise PresetUsageError("--consistency-sigma requires --consistency")
    if consistency is not None and not any(
        _question_type(question) == "noul" for question in questions.values()
    ):
        raise PresetUsageError("--consistency requires at least one Noul question")


def _question_type(question: Any) -> str | None:
    if isinstance(question, Mapping):
        value = question.get("type")
    else:
        value = getattr(question, "type", None)
    return value if isinstance(value, str) else None


def _question_state_fields(question: Any) -> tuple[str, ...]:
    if isinstance(question, Mapping):
        instructions = question.get("instructions")
    else:
        instructions = getattr(question, "instructions", None)
    if not isinstance(instructions, Mapping):
        return ()
    fields = instructions.get("state_fields")
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        return ()
    return tuple(field for field in fields if isinstance(field, str))


def _repeat_state(state: State) -> State:
    context = dict(state.context)
    context["uid"] = uuid4().hex
    return State(state.state_ref, state.focus, context)


def _complete_for_questions(
    response: JudgeResponse,
    questions: Mapping[str, Any],
) -> bool:
    if not response.complete or set(response.answers) != set(questions):
        return False
    return all(
        getattr(response.answers[question_id], "type", None)
        == _question_type(question)
        for question_id, question in questions.items()
    )


def _aggregate_responses(
    responses: Sequence[JudgeResponse],
    questions: Mapping[str, Any],
    samples: int,
) -> JudgeResponse:
    first = responses[0]
    answers: dict[str, Any] = {}
    for question_id, question in questions.items():
        answer = first.answers[question_id]
        if _question_type(question) != "noul":
            answers[question_id] = answer
            continue
        values = [
            response.answers[question_id].noul
            for response in responses
            if isinstance(response.answers[question_id], NoulAnswer)
        ]
        if len(values) != samples:
            raise ValueError("consistency responses have an invalid Noul answer")
        mean = math.fsum(values) / samples
        variance = math.fsum((value - mean) ** 2 for value in values) / samples
        answers[question_id] = NoulAnswer(
            mean,
            consistency={
                "samples": samples,
                "mean": mean,
                "stddev": math.sqrt(variance),
            },
        )
    usage: dict[str, int | float] = {}
    for response in responses:
        _add_usage(usage, response.usage)
    return JudgeResponse(
        answers=answers,
        served_model=first.served_model,
        usage=usage or None,
    )


def _add_usage(
    total: dict[str, int | float],
    usage: Mapping[str, Any] | None,
) -> None:
    if not isinstance(usage, Mapping):
        return
    for key, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        total[key] = total.get(key, 0) + value


class FakeJudge:
    """Deterministic offline judge for unit tests and evals."""

    def __init__(self, mode: str = "complete") -> None:
        if mode not in {"complete", "incomplete", "error"}:
            raise ValueError(f"unknown fake judge mode: {mode}")
        self.mode = mode

    def __call__(
        self, state: State, questions: Mapping[str, Any], model: str
    ) -> TypedResponse:
        del model
        if self.mode == "error":
            return ErrorResponse("fake operational error")

        answers: dict[str, Any] = {}
        for question_id, question in questions.items():
            digest = hashlib.sha256(
                f"{state.state_ref}\0{question_id}".encode()
            ).digest()
            fraction = int.from_bytes(digest[:2], "big") / 65_535
            question_type = (
                question["type"] if isinstance(question, Mapping) else question.type
            )
            if question_type == "noul":
                answers[question_id] = NoulAnswer(round(fraction, 4))
            elif question_type == "choice":
                answers[question_id] = ChoiceAnswer(
                    "true" if fraction >= 0.5 else "false",
                    probabilities={
                        "false": round(1 - fraction, 4),
                        "true": round(fraction, 4),
                    },
                    confidence=round(abs(fraction - 0.5) * 2, 4),
                )
            elif question_type == "score":
                answers[question_id] = ScoreAnswer(
                    round(fraction * 3, 4),
                    legend={"0": "low", "1": "moderate", "2": "high", "3": "critical"},
                    probabilities={
                        "0": round(1 - fraction, 4),
                        "3": round(fraction, 4),
                    },
                    confidence=round(abs(fraction - 0.5) * 2, 4),
                )
            else:
                raise ValueError(f"unknown question type: {question_type}")

        missing: tuple[str, ...] = ()
        if self.mode == "incomplete" and answers:
            missing_id = next(reversed(answers))
            del answers[missing_id]
            missing = (missing_id,)
        return JudgeResponse(answers=answers, missing_questions=missing)


DeterministicFakeJudge = FakeJudge
FormedState = State


_WORD_TOKEN = re.compile(r"\w+", re.UNICODE)
_BM25_K1 = 1.2
_BM25_B = 0.75


@dataclass(frozen=True, slots=True)
class BM25CorpusStats:
    document_count: int
    average_length: float
    document_frequency: Mapping[str, int]


def tokenize(value: str) -> tuple[str, ...]:
    """Return lowercase Unicode word tokens in stable order."""
    return tuple(_WORD_TOKEN.findall(value.lower()))


def _searchable_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _state_tokens(state: State, fields: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field_name in fields:
        if field_name == "focus":
            value: Any = state.focus
        elif field_name.startswith("context."):
            value = state.context.get(field_name.removeprefix("context."), "")
        else:
            value = ""
        values.append(_searchable_value(value))
    return tokenize(" ".join(values))


def bm25_score(
    query_tokens: Sequence[str],
    document_tokens: Sequence[str],
    corpus_stats: BM25CorpusStats,
) -> float:
    """Score one document with the pinned BM25 parameters."""
    length = len(document_tokens)
    counts: dict[str, int] = {}
    for token in document_tokens:
        counts[token] = counts.get(token, 0) + 1
    score = 0.0
    for token in query_tokens:
        frequency = counts.get(token, 0)
        if not frequency:
            continue
        df = corpus_stats.document_frequency[token]
        idf = math.log(
            1
            + (corpus_stats.document_count - df + 0.5)
            / (df + 0.5)
        )
        denominator = frequency + _BM25_K1 * (
            1
            - _BM25_B
            + _BM25_B * length / corpus_stats.average_length
            if corpus_stats.average_length
            else 1
        )
        score += idf * ((frequency * (_BM25_K1 + 1)) / denominator)
    return score


def bm25_rank(
    states: Sequence[State], query: str, fields: Sequence[str]
) -> tuple[State, ...]:
    """Rank all states with deterministic BM25 and stable state references."""
    formed = tuple(states)
    if not formed:
        return ()
    query_tokens = tuple(dict.fromkeys(tokenize(query)))
    documents = tuple(_state_tokens(state, fields) for state in formed)
    document_frequency = {
        token: sum(token in document for document in documents)
        for token in set(query_tokens)
    }
    average_length = sum(len(document) for document in documents) / len(documents)
    corpus_stats = BM25CorpusStats(
        len(formed),
        average_length,
        document_frequency,
    )
    scored: list[tuple[float, str, int, State]] = []
    for index, (state, document) in enumerate(zip(formed, documents)):
        score = bm25_score(query_tokens, document, corpus_stats)
        scored.append((score, state.state_ref, index, state))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple(item[3] for item in scored)


def _validate_runtime_prefilter(prefilter: Mapping[str, Any]) -> None:
    required = {"ranker", "top", "fields", "query"}
    if not required <= set(prefilter):
        raise ValueError("prefilter is missing required fields")
    if prefilter["ranker"] != "bm25":
        raise ValueError("prefilter.ranker must be bm25")
    top = prefilter["top"]
    if isinstance(top, bool) or not isinstance(top, int) or top <= 0:
        raise ValueError("prefilter.top must be a positive integer")
    fields = prefilter["fields"]
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        raise ValueError("prefilter.fields must be a list")
    if not fields or any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("prefilter.fields must not be empty")
    if len(set(fields)) != len(fields):
        raise ValueError("prefilter.fields must not contain duplicates")
    query = prefilter["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("prefilter.query must not be empty")


def _response_record(
    state_ref: str, response: TypedResponse, meta: RecordMeta
) -> ResultRecord | PartialResultRecord | ErrorRecord:
    if isinstance(response, ErrorResponse):
        kind: str = (
            "malformed_answer" if response.error == "malformed answer" else "api_error"
        )
        return ErrorRecord(
            state_ref,
            ErrorDetail(
                kind, response.error, response.http_status, response.attempts
            ),
            meta,
        )
    if response.complete:
        return ResultRecord(
            state_ref,
            response.answers,
            meta,
            usage=response.usage,
            latency_ms=response.latency_ms,
        )
    return PartialResultRecord(
        state_ref,
        response.answers,
        response.missing_questions,
        meta,
        usage=response.usage,
        latency_ms=response.latency_ms,
    )


def emit_pretty(
    record: ResultRecord, stderr: TextIO, template: str | None = None
) -> None:
    if template is None:
        rendered = "\t".join(
            (
                record.state_ref,
                json.dumps(
                    record.to_dict()["answers"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        )
    else:
        rendered = _render_pretty_template(record.to_dict(), template)
    stderr.write(rendered + "\n")
    stderr.flush()


def _render_pretty_template(payload: Mapping[str, Any], template: str) -> str:
    formatter = string.Formatter()
    output: list[str] = []
    for literal, field_name, format_spec, conversion in formatter.parse(template):
        output.append(literal)
        if field_name is None:
            continue
        value: Any = payload
        for part in field_name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ValueError(
                    f"pretty template references unknown field {field_name!r}"
                )
            value = value[part]
        if conversion:
            value = formatter.convert_field(value, conversion)
        output.append(format(value, format_spec))
    return "".join(output).replace(r"\t", "\t")
