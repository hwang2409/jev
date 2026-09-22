from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TextIO

from .answers import (
    CacheStatus,
    CanonicalRecord,
    ChoiceAnswer,
    CoverageReason,
    CoverageRecord,
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
from .cache import CacheStore, build_cache_preimage, cache_key


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


@dataclass(frozen=True, slots=True)
class StateRejection:
    state_ref: str | None
    reason: str
    message: str
    source_ref: str | None = None
    boundary: str | None = None

    def __post_init__(self) -> None:
        if self.reason not in {"scan_cap", "context_limit", "input_error"}:
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
        if not self.skipped or self.max_chunks is None:
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
    if max_chunks is not None and max_chunks < 0:
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


@dataclass(frozen=True, slots=True)
class RunStats:
    discovered: int
    judged: int
    emitted: int
    skipped: int
    failed: int


@dataclass(frozen=True, slots=True)
class RunResult:
    admission: StateAdmission
    responses: tuple[TypedResponse, ...]
    stats: RunStats
    records: tuple[CanonicalRecord, ...] = ()
    coverage_reasons: tuple[CoverageReason, ...] = ()
    exit_code: int = 0


class Runner:
    def __init__(
        self,
        judge_fn: JudgeFn,
        model: str = "jev-1.13.0",
        limits: StateLimits = StateLimits(),
    ) -> None:
        if judge_fn is None:
            raise TypeError("judge_fn is required")
        self.judge_fn = judge_fn
        self.model = model
        self.limits = limits

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
        questions: Mapping[str, Any],
        max_chunks: int | None = None,
        *,
        preset: str = "jmap",
        preset_version: str = "1",
        chunker: str = "unknown",
        cache: CacheStatus = "not_applicable",
        cache_store: CacheStore | None = None,
        chunking: Mapping[str, Any] | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        output_format: str = "jsonl",
        result_filter: Callable[[ResultRecord], bool] | None = None,
        rejections: Sequence[StateRejection] = (),
    ) -> RunResult:
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
        admission = self.admit(states, max_chunks, rejections)
        meta = RecordMeta(preset, preset_version, self.model, chunker, cache)
        resolved_chunking = (
            dict(chunking) if chunking is not None else {"by": chunker}
        )
        responses: list[TypedResponse] = []
        records: list[CanonicalRecord] = []

        def write(record: CanonicalRecord, visible: bool = True) -> None:
            records.append(record)
            if stdout is not None and visible:
                emit_jsonl(record, stdout)

        for state in admission.admitted:
            state_meta = meta
            preimage = None
            if cache_store is not None:
                preimage = build_cache_preimage(
                    model=self.model,
                    preset=preset,
                    preset_version=preset_version,
                    chunking=resolved_chunking,
                    questions=questions,
                    state=state,
                    limits=self.limits,
                )
                cached = cache_store.get(cache_key(preimage), questions)
                if cached is not None:
                    response = cached.response
                    state_meta = RecordMeta(
                        preset, preset_version, self.model, chunker, "hit"
                    )
                else:
                    state_meta = RecordMeta(
                        preset, preset_version, self.model, chunker, "miss"
                    )
                    response = None
            else:
                response = None

            if response is None:
                try:
                    response = self.judge(state, questions)
                except Exception:
                    response = ErrorResponse("request failed")
                if (
                    cache_store is not None
                    and preimage is not None
                    and isinstance(response, JudgeResponse)
                    and response.complete
                ):
                    cache_store.publish(preimage, response)
            responses.append(response)
            record = _response_record(state.state_ref, response, state_meta)
            visible = not isinstance(record, ResultRecord) or result_filter is None
            if isinstance(record, ResultRecord) and result_filter is not None:
                visible = result_filter(record)
            write(record, visible)
            if (
                output_format == "pretty"
                and isinstance(record, ResultRecord)
                and visible
                and stderr is not None
            ):
                emit_pretty(record, stderr)

        for record in _rejection_records(admission, meta):
            write(record)
            if stderr is not None:
                stderr.write(f"jmap: warning: {record.error.message}\n")
                stderr.flush()

        failed = sum(not response.complete for response in responses)
        stats = RunStats(
            discovered=admission.discovered,
            judged=len(responses),
            emitted=len(responses),
            skipped=admission.skipped_count,
            failed=failed,
        )
        reasons = _coverage_reasons(admission, responses)
        coverage = "partial" if reasons else "complete"
        coverage_meta = RecordMeta(
            preset, preset_version, self.model, chunker, "not_applicable"
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
        if stderr is not None:
            if coverage == "partial":
                stderr.write(
                    "jmap: warning: results are partial; "
                    f"coverage reasons: {', '.join(reasons)}\n"
                )
            stderr.flush()
        exit_code = 2 if coverage == "partial" else 0
        return RunResult(
            admission,
            tuple(responses),
            stats,
            tuple(records),
            reasons,
            exit_code,
        )

    def run_jsonl(
        self,
        states: Sequence[State],
        questions: Mapping[str, Any],
        stdout: TextIO,
        **kwargs: Any,
    ) -> RunResult:
        """Run a finite judgment and write only canonical JSONL to stdout."""
        kwargs["stdout"] = stdout
        return self.run(states, questions, **kwargs)


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

    if admission.skipped and admission.max_chunks is not None:
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
        if rejection.reason in {"scan_cap", "context_limit"} and (
            rejection.state_ref is not None
        ):
            add_skip(rejection)
        elif rejection.reason == "input_error":
            events.append(rejection)

    skip_meta = RecordMeta(
        meta.preset, meta.preset_version, meta.model, meta.chunker, "not_applicable"
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
    if admission.skipped:
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


def emit_pretty(record: ResultRecord, stderr: TextIO) -> None:
    answers = json.dumps(
        record.to_dict()["answers"], ensure_ascii=False, sort_keys=True
    )
    stderr.write(f"{record.state_ref}\t{answers}\n")
    stderr.flush()
