from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .answers import (
    ChoiceAnswer,
    ErrorResponse,
    JudgeResponse,
    NoulAnswer,
    ScoreAnswer,
    TypedResponse,
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


@dataclass(frozen=True, slots=True)
class StateRejection:
    state_ref: str | None
    reason: str
    message: str
    source_ref: str | None = None


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
            rejection.state_ref is not None for rejection in self.rejections
        )

    @property
    def judged(self) -> int:
        return len(self.admitted)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped) + sum(
            rejection.state_ref is not None for rejection in self.rejections
        )

    @property
    def skip_boundary(self) -> str | None:
        if not self.skipped or self.max_chunks is None:
            return None
        return f"max_chunks={self.max_chunks}"


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


class Runner:
    def __init__(
        self,
        judge_fn: JudgeFn | None = None,
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
        self, states: Sequence[State], max_chunks: int | None = None
    ) -> StateAdmission:
        for state in states:
            validate_state(state, self.limits)
        return admit_states(states, max_chunks)

    def run(
        self,
        states: Sequence[State],
        questions: Mapping[str, Any],
        max_chunks: int | None = None,
    ) -> RunResult:
        admission = self.admit(states, max_chunks)
        responses = tuple(self.judge(state, questions) for state in admission.admitted)
        failed = sum(isinstance(response, ErrorResponse) for response in responses)
        stats = RunStats(
            discovered=admission.discovered,
            judged=len(responses),
            emitted=len(responses),
            skipped=admission.skipped_count,
            failed=failed,
        )
        return RunResult(admission, responses, stats)


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
