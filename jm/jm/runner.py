from __future__ import annotations

import hashlib
import json
import math
import re
import string
import time as _time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from os import PathLike
from threading import Lock
from typing import Any, Protocol, TextIO
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
from .cache import CACHE_SCHEMA, CacheStore, build_cache_preimage, cache_key
from .gates import GateResult, Policy, PolicyError, compile_policy, evaluate_gate
from .presets import (
    SCHEMA_V2,
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
        return self.payload


class JudgeFn(Protocol):
    def __call__(
        self, state: State, questions: Mapping[str, Any], model: str
    ) -> TypedResponse: ...


class StateLimitError(ValueError):
    """A formed state exceeds a configured byte limit."""


class StateInputError(ValueError):
    """A formed input cannot be judged safely."""

    exit_code = 2


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
        if policy is not None and require_states < 0:
            raise PolicyError("require_states must be non-negative")
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency <= 0
        ):
            raise ValueError("concurrency must be a positive integer")
        loaded_preset = self._load_preset(self.preset if preset is None else preset)
        if policy is not None and prefilter is not None:
            raise PresetUsageError("gate does not support prefiltering")
        if loaded_preset is None:
            if questions is _UNSET or questions is None:
                raise TypeError("questions or preset is required")
            runtime_questions = questions
            runtime_model = self.model
            runtime_limits = self.limits
            runtime_name = str(preset) if preset is not None else "jm"
            runtime_version = "1" if preset_version is _UNSET else preset_version
            runtime_schema = None
            runtime_chunker = "unknown" if chunker is _UNSET else chunker
            runtime_max_chunks = None if max_chunks is _UNSET else max_chunks
            resolved_chunking = (
                dict(chunking)
                if chunking is not _UNSET and chunking is not None
                else {
                    "by": runtime_chunker,
                    "max_chunks": runtime_max_chunks,
                }
            )
        else:
            if questions is not _UNSET:
                raise PresetUsageError(
                    "questions cannot be supplied with a preset; "
                    "use the preset's questions"
                )
            runtime_chunker = loaded_preset.effective_chunker(
                None if chunker is _UNSET else chunker
            )
            preset_limits = StateLimits(**loaded_preset.chunking["limits"])
            if self._model_supplied and self.model != loaded_preset.model:
                raise PresetUsageError(
                    f"model {self.model!r} conflicts with preset "
                    f"{loaded_preset.name!r}"
                )
            if self._limits_supplied and self.limits != preset_limits:
                raise PresetUsageError(
                    f"limits conflict with preset {loaded_preset.name!r}"
                )
            if (
                preset_version is not _UNSET
                and preset_version != loaded_preset.version
            ):
                raise PresetUsageError(
                    f"preset_version {preset_version!r} conflicts with preset "
                    f"{loaded_preset.name!r}"
                )
            preset_max_chunks = loaded_preset.chunking.get("max_chunks")
            if max_chunks is not _UNSET and max_chunks != preset_max_chunks:
                raise PresetUsageError(
                    f"max_chunks {max_chunks!r} conflicts with preset "
                    f"{loaded_preset.name!r}"
                )
            runtime_questions = loaded_preset.questions
            runtime_model = loaded_preset.model
            runtime_limits = preset_limits
            runtime_name = loaded_preset.name
            runtime_version = loaded_preset.version
            runtime_schema = (
                loaded_preset.schema if loaded_preset.schema == SCHEMA_V2 else None
            )
            resolved_chunking = dict(loaded_preset.chunking)
            resolved_chunking["by"] = runtime_chunker
            if chunking is not _UNSET:
                if chunking is None:
                    raise PresetUsageError(
                        f"chunking conflicts with preset {loaded_preset.name!r}"
                    )
                self._reject_chunking_conflicts(
                    chunking, resolved_chunking, loaded_preset.name
                )
            runtime_max_chunks = preset_max_chunks

        _validate_consistency(
            consistency,
            consistency_sigma,
            runtime_questions,
        )

        compiled_policy = None
        if policy is not None:
            if loaded_preset is None:
                raise PresetUsageError("a gate policy requires a preset")
            compiled_policy = compile_policy(policy, loaded_preset)

        if output_format not in {"jsonl", "pretty"}:
            raise ValueError("output format must be jsonl or pretty")
        if not states and not rejections:
            rejections = (
                StateRejection(
                    None,
                    "input_error",
                    "input is empty",
                    "stdin:byte=0,line=1",
                ),
            )
        if (
            runtime_max_chunks is not None
            and (
                isinstance(runtime_max_chunks, bool)
                or not isinstance(runtime_max_chunks, int)
                or runtime_max_chunks < 0
            )
        ):
            raise ValueError("max_chunks must be non-negative")
        for state in states:
            validate_state(state, runtime_limits)
        seen_refs: set[str] = set()
        for state in states:
            if state.state_ref in seen_refs:
                raise StateInputError(
                    f"duplicate state reference {state.state_ref!r}"
                )
            seen_refs.add(state.state_ref)
        if prefilter is None:
            admission = admit_states(states, runtime_max_chunks, rejections)
        else:
            _validate_runtime_prefilter(prefilter)
            admission = _admit_prefiltered_states(
                states,
                top=prefilter["top"],
                max_chunks=runtime_max_chunks,
                rejections=rejections,
                query=prefilter["query"],
                fields=prefilter["fields"],
            )
        meta = RecordMeta(
            runtime_name,
            runtime_version,
            runtime_model,
            runtime_chunker,
            cache,
            preset_schema=runtime_schema,
        )
        responses: list[TypedResponse] = []
        records: list[CanonicalRecord] = []
        effective_concurrency = min(concurrency, 8)
        if concurrency > 8:
            records.append(
                DiagnosticRecord(
                    Diagnostic(
                        "warning",
                        "concurrency_capped",
                        f"requested={concurrency} "
                        f"effective={effective_concurrency} cap=8",
                    )
                )
            )
        consistency_cache_hits = 0
        consistency_live_calls = 0
        consistency_usage: dict[str, int | float] = {}
        consistency_lock = Lock()
        consistency_call_lock = Lock()
        concurrency_lock = Lock()

        def judge_state(
            state: State,
        ) -> tuple[TypedResponse, RecordMeta]:
            nonlocal consistency_cache_hits, consistency_live_calls
            state_meta = meta

            def one_call(call_state: State) -> tuple[TypedResponse, bool]:
                nonlocal consistency_cache_hits, consistency_live_calls
                preimage = None
                if cache_store is not None:
                    preimage = build_cache_preimage(
                        model=runtime_model,
                        preset=runtime_name,
                        preset_version=runtime_version,
                        chunking=resolved_chunking,
                        questions=runtime_questions,
                        state=call_state,
                        limits=runtime_limits if loaded_preset is None else None,
                        cache_schema=CACHE_SCHEMA,
                        preset_schema=runtime_schema,
                    )
                    cached = cache_store.get(cache_key(preimage), runtime_questions)
                    if cached is not None:
                        if consistency is not None:
                            with consistency_lock:
                                consistency_cache_hits += 1
                        return cached.response, True

                try:
                    if consistency is None:
                        response = self.judge_fn(
                            call_state, runtime_questions, runtime_model
                        )
                    else:
                        with consistency_call_lock:
                            response = self.judge_fn(
                                call_state, runtime_questions, runtime_model
                            )
                except Exception:
                    response = ErrorResponse("request failed")
                if consistency is not None:
                    with consistency_lock:
                        consistency_live_calls += 1
                if (
                    cache_store is not None
                    and preimage is not None
                    and isinstance(response, JudgeResponse)
                    and response.complete
                    and (
                        consistency is None
                        or _complete_for_questions(response, runtime_questions)
                    )
                ):
                    cache_store.publish(preimage, response, usage=response.usage)
                return response, False

            if consistency is None:
                response, cache_hit = one_call(state)
                if cache_store is not None:
                    state_meta = replace(
                        state_meta,
                        cache="hit" if cache_hit else "miss",
                    )
                if isinstance(response, JudgeResponse) and response.served_model:
                    state_meta = replace(
                        state_meta, served_model=response.served_model
                    )
                return response, state_meta

            responses_for_state: list[JudgeResponse] = []
            failure: ErrorResponse | None = None
            state_cache_hits = 0
            for _ in range(consistency):
                repeat_state = _repeat_state(state)
                response, cache_hit = one_call(repeat_state)
                state_cache_hits += int(cache_hit)
                if isinstance(response, JudgeResponse):
                    with consistency_lock:
                        _add_usage(consistency_usage, response.usage)
                if failure is not None:
                    continue
                if not isinstance(response, JudgeResponse):
                    failure = ErrorResponse(
                        "consistency repeat failed: "
                        + (
                            response.error
                            if isinstance(response, ErrorResponse)
                            else "invalid response"
                        )
                    )
                elif not _complete_for_questions(response, runtime_questions):
                    failure = ErrorResponse(
                        "consistency repeat returned incomplete answers"
                    )
                else:
                    responses_for_state.append(response)

            state_meta = replace(
                state_meta,
                cache=(
                    "hit"
                    if state_cache_hits == consistency
                    else "miss"
                )
                if cache_store is not None
                else "not_applicable",
            )
            if failure is not None:
                return failure, state_meta
            try:
                aggregate = _aggregate_responses(
                    responses_for_state,
                    runtime_questions,
                    consistency,
                )
            except (TypeError, ValueError):
                return (
                    ErrorResponse("consistency repeat returned malformed answers"),
                    state_meta,
                )
            if aggregate.served_model:
                state_meta = replace(
                    state_meta, served_model=aggregate.served_model
                )
            return aggregate, state_meta

        broken_pipe = False

        def write(record: CanonicalRecord, visible: bool = True) -> None:
            nonlocal broken_pipe
            if broken_pipe:
                return
            records.append(record)
            if (
                stdout is None
                or not visible
                or isinstance(record, DiagnosticRecord)
            ):
                return
            try:
                emit_jsonl(record, stdout)
            except BrokenPipeError:
                broken_pipe = True
                try:
                    stdout.close()
                except OSError:
                    pass

        admitted = admission.admitted
        judged: list[tuple[TypedResponse, RecordMeta]] = []
        current_concurrency = effective_concurrency
        consecutive_503 = 0
        last_503_at: float | None = None
        concurrency_diagnostics: list[DiagnosticRecord] = []

        def observe_response(status_code: int) -> None:
            nonlocal consecutive_503, current_concurrency, last_503_at
            with concurrency_lock:
                now = _time.monotonic()
                if status_code == 503:
                    consecutive_503 += 1
                    last_503_at = now
                    if consecutive_503 >= 2:
                        reduced = max(1, current_concurrency // 2)
                        if reduced < current_concurrency:
                            current_concurrency = reduced
                            concurrency_diagnostics.append(
                                DiagnosticRecord(
                                    Diagnostic(
                                        "warning",
                                        "concurrency_backoff",
                                        "status=503 "
                                        "consecutive=2 "
                                        f"effective={current_concurrency}",
                                    )
                                )
                            )
                        consecutive_503 = 0
                    return

                consecutive_503 = 0
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

        set_response_observer = getattr(
            self.judge_fn, "set_response_observer", None
        )
        uses_transport_observer = callable(set_response_observer)
        if uses_transport_observer:
            set_response_observer(observe_response)
        offset = 0
        diagnostics_published = 0
        try:
            while offset < len(admitted):
                with concurrency_lock:
                    batch_size = current_concurrency
                batch = admitted[offset : offset + batch_size]
                if batch_size == 1 or len(batch) <= 1:
                    batch_results = [judge_state(state) for state in batch]
                else:
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        batch_results = list(executor.map(judge_state, batch))
                judged.extend(batch_results)
                if not uses_transport_observer:
                    for response, _state_meta in batch_results:
                        status_code = (
                            response.http_status
                            if isinstance(response, ErrorResponse)
                            and response.http_status is not None
                            else 200
                        )
                        observe_response(status_code)
                with concurrency_lock:
                    new_diagnostics = concurrency_diagnostics[
                        diagnostics_published:
                    ]
                    diagnostics_published = len(concurrency_diagnostics)
                records.extend(new_diagnostics)
                offset += len(batch)
        finally:
            if uses_transport_observer:
                set_response_observer(None)

        responses.extend(response for response, _state_meta in judged)

        pretty_template = (
            loaded_preset.data["output"]["pretty_template"]
            if loaded_preset is not None
            else None
        )
        for state, (response, state_meta) in zip(admitted, judged):
            record = _response_record(state.state_ref, response, state_meta)
            visible = not isinstance(record, ResultRecord) or result_filter is None
            if isinstance(record, ResultRecord) and result_filter is not None:
                visible = result_filter(record)
            write(record, visible)
            if broken_pipe:
                break
            if (
                output_format == "pretty"
                and isinstance(record, ResultRecord)
                and visible
                and stderr is not None
            ):
                emit_pretty(record, stderr, pretty_template)

        for record in _rejection_records(admission, meta):
            write(record)
            if broken_pipe:
                break
            if stderr is not None and record.error.kind != "prefiltered":
                stderr.write(f"jm: warning: {record.error.message}\n")
                stderr.flush()

        failed = sum(not response.complete for response in responses)
        stats = RunStats(
            discovered=admission.discovered,
            judged=len(responses),
            emitted=len(responses),
            skipped=admission.skipped_count,
            failed=failed,
            consistency_attempted_calls=(len(admitted) * consistency)
            if consistency is not None
            else 0,
            consistency_cache_hits=consistency_cache_hits,
            consistency_live_calls=consistency_live_calls,
            consistency_usage=dict(consistency_usage),
        )
        reasons = _coverage_reasons(admission, responses)
        coverage = "partial" if reasons else "complete"
        coverage_meta = RecordMeta(
            runtime_name,
            runtime_version,
            runtime_model,
            runtime_chunker,
            "not_applicable",
            preset_schema=runtime_schema,
        )
        coverage_record = CoverageRecord(
            coverage=coverage,
            coverage_counts={
                "discovered": stats.discovered,
                "judged": stats.judged,
                "emitted": stats.emitted,
                "skipped": stats.skipped,
                "failed": stats.failed,
            },
            coverage_reasons=reasons,
            meta=coverage_meta,
        )
        write(coverage_record)
        if broken_pipe:
            return RunResult(
                admission,
                tuple(responses),
                stats,
                tuple(records),
                reasons,
                0,
                None,
                True,
            )
        if stderr is not None:
            if coverage == "partial":
                prefiltered_count = sum(
                    rejection.reason == "prefiltered"
                    for rejection in admission.skip_rejections
                )
                if prefilter_warning and prefiltered_count:
                    stderr.write(
                        "jm: warning: BM25 prefilter skipped "
                        f"{prefiltered_count} of {stats.discovered} states; "
                        "recall is bounded by the shortlist; rerun without "
                        "--prefilter for full recall\n"
                    )
                else:
                    stderr.write(
                        "jm: warning: results are partial; "
                        f"coverage reasons: {', '.join(reasons)}\n"
                    )
            if consistency is not None:
                stderr.write(
                    "jm: consistency: "
                    f"{len(admitted)} states * {consistency} = "
                    f"{stats.consistency_attempted_calls} attempted calls; "
                    f"cache hits: {stats.consistency_cache_hits}; "
                    f"live calls: {stats.consistency_live_calls}; "
                    "total normalized token usage: "
                    f"{json.dumps(stats.consistency_usage, sort_keys=True)}\n"
                )
            stderr.flush()
        gate_result = None
        exit_code = 2 if coverage == "partial" else 0
        if compiled_policy is not None:
            result_records = tuple(
                record for record in records if isinstance(record, ResultRecord)
            )
            gate_result = evaluate_gate(
                compiled_policy,
                result_records,
                judged_states=stats.judged,
                coverage_reasons=reasons,
                required_states=require_states,
                consistency_sigma=consistency_sigma,
            )
            exit_code = gate_result.exit_code
        return RunResult(
            admission,
            tuple(responses),
            stats,
            tuple(records),
            reasons,
            exit_code,
            gate_result,
            False,
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
        return ResultRecord(state_ref, response.answers, meta)
    return PartialResultRecord(
        state_ref, response.answers, response.missing_questions, meta
    )


def _rejection_records(
    admission: StateAdmission, meta: RecordMeta
) -> tuple[ErrorRecord, ...]:
    grouped: dict[tuple[str, str], list[StateRejection]] = {}
    events: list[StateRejection | tuple[str, str]] = []

    def add_skip(rejection: StateRejection) -> None:
        boundary = rejection.boundary or rejection.reason
        key = (rejection.reason, boundary)
        if key not in grouped:
            grouped[key] = []
            events.append(key)
        grouped[key].append(rejection)

    if admission.skip_rejections:
        for rejection in admission.skip_rejections:
            add_skip(rejection)
    elif admission.skipped and admission.max_chunks is not None:
        for state in admission.skipped:
            add_skip(
                StateRejection(
                    state.state_ref,
                    "scan_cap",
                    "scan cap reached before visit",
                    boundary=f"max_chunks={admission.max_chunks}",
                )
            )
    for rejection in admission.rejections:
        if rejection.reason in {"scan_cap", "context_limit", "prefiltered"} and (
            rejection.state_ref is not None
        ):
            add_skip(rejection)
        elif rejection.reason == "input_error":
            events.append(rejection)

    skip_meta = RecordMeta(
        meta.preset,
        meta.preset_version,
        meta.model,
        meta.chunker,
        "not_applicable",
        preset_schema=meta.preset_schema,
    )
    records: list[ErrorRecord] = []
    for event in events:
        if isinstance(event, StateRejection):
            records.append(
                ErrorRecord(
                    None,
                    ErrorDetail("input_error", event.message),
                    skip_meta,
                    source_ref=event.source_ref,
                )
            )
            continue
        reason, boundary = event
        skipped = grouped[event]
        records.append(
            ErrorRecord(
                None,
                ErrorDetail(
                    reason,  # type: ignore[arg-type]
                    skipped[0].message,
                    skip_summary=SkipSummary(
                        boundary,
                        len(skipped),
                        tuple(item.state_ref for item in skipped if item.state_ref),
                    ),
                ),
                skip_meta,
            )
        )
    return tuple(records)


def _coverage_reasons(
    admission: StateAdmission, responses: Sequence[TypedResponse]
) -> tuple[CoverageReason, ...]:
    found: set[str] = set()
    for rejection in admission.skip_rejections:
        found.add(rejection.reason)
    if admission.skipped and not admission.skip_rejections:
        found.add("scan_cap")
    for rejection in admission.rejections:
        if rejection.reason in {"scan_cap", "input_error", "context_limit"}:
            found.add(rejection.reason)
    for response in responses:
        if isinstance(response, ErrorResponse):
            found.add(
                "malformed_answer"
                if response.error == "malformed answer"
                else "api_error"
            )
        elif not response.complete:
            found.add("partial_answer")
    order = (
        "prefiltered",
        "scan_cap",
        "input_error",
        "context_limit",
        "api_error",
        "malformed_answer",
        "partial_answer",
    )
    return tuple(reason for reason in order if reason in found)  # type: ignore[return-value]


def emit_jsonl(record: CanonicalRecord, stdout: TextIO) -> None:
    stdout.write(
        json.dumps(
            record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    )
    stdout.flush()


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
