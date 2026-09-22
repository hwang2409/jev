from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

CacheStatus = Literal["hit", "miss", "not_applicable"]
ErrorKind = Literal[
    "api_error", "malformed_answer", "scan_cap", "context_limit", "input_error"
]
CoverageStatus = Literal["complete", "partial"]
CoverageReason = Literal[
    "scan_cap",
    "input_error",
    "context_limit",
    "api_error",
    "malformed_answer",
    "partial_answer",
]


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    noul: float
    type: Literal["noul"] = "noul"


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    type: Literal["choice"] = "choice"


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    score: float
    legend: dict[str, str] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    type: Literal["score"] = "score"


type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True, slots=True)
class JudgeResponse:
    answers: dict[str, Answer] = field(default_factory=dict)
    missing_questions: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing_questions


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    error: str
    answers: dict[str, Answer] = field(default_factory=dict)
    http_status: int | None = None
    attempts: int = 0

    @property
    def message(self) -> str:
        return self.error

    @property
    def complete(self) -> bool:
        return False


type TypedResponse = JudgeResponse | ErrorResponse


@dataclass(frozen=True, slots=True)
class RecordMeta:
    preset: str
    preset_version: str
    model: str
    chunker: str
    cache: CacheStatus
    partial: bool = False

    def __post_init__(self) -> None:
        if self.cache not in {"hit", "miss", "not_applicable"}:
            raise ValueError(f"unknown cache status: {self.cache}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "preset": self.preset,
            "preset_version": self.preset_version,
            "model": self.model,
            "chunker": self.chunker,
            "cache": self.cache,
        }
        if self.partial:
            result["partial"] = True
        return result


@dataclass(frozen=True, slots=True)
class SkipSummary:
    boundary: str
    count: int
    sample_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("skip count must not be negative")
        object.__setattr__(self, "sample_refs", tuple(self.sample_refs[:8]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "count": self.count,
            "sample_refs": list(self.sample_refs),
        }


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    kind: ErrorKind
    message: str
    http_status: int | None = None
    attempts: int = 0
    skip_summary: SkipSummary | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "api_error",
            "malformed_answer",
            "scan_cap",
            "context_limit",
            "input_error",
        }:
            raise ValueError(f"unknown error kind: {self.kind}")
        if self.attempts < 0:
            raise ValueError("attempts must not be negative")
        is_skip = self.kind in {"scan_cap", "context_limit"}
        if is_skip != (self.skip_summary is not None):
            raise ValueError("skip summaries are required only for skip errors")
        if is_skip and (self.http_status is not None or self.attempts != 0):
            raise ValueError("skip summaries cannot have http status or attempts")
        if self.kind == "input_error" and (
            self.http_status is not None or self.attempts != 0
        ):
            raise ValueError("input errors cannot have http status or attempts")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "message": self.message,
            "http_status": self.http_status,
            "attempts": self.attempts,
        }
        if self.skip_summary is not None:
            result["skip_summary"] = self.skip_summary.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class ResultRecord:
    state_ref: str
    answers: Mapping[str, Answer]
    meta: RecordMeta

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "result",
            "state_ref": self.state_ref,
            "answers": answers_to_dict(self.answers),
            "meta": self.meta.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PartialResultRecord:
    state_ref: str
    answers: Mapping[str, Answer]
    missing_questions: tuple[str, ...]
    meta: RecordMeta

    def __post_init__(self) -> None:
        if not self.missing_questions:
            raise ValueError("partial results require missing questions")
        if not self.meta.partial:
            object.__setattr__(
                self,
                "meta",
                RecordMeta(
                    self.meta.preset,
                    self.meta.preset_version,
                    self.meta.model,
                    self.meta.chunker,
                    self.meta.cache,
                    partial=True,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "partial_result",
            "state_ref": self.state_ref,
            "answers": answers_to_dict(self.answers),
            "missing_questions": list(self.missing_questions),
            "meta": self.meta.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    state_ref: str | None
    error: ErrorDetail
    meta: RecordMeta
    source_ref: str | None = None

    def __post_init__(self) -> None:
        if self.error.kind in {"api_error", "malformed_answer"}:
            if self.state_ref is None or self.source_ref is not None:
                raise ValueError("judgment errors require only a state reference")
        elif self.error.kind == "input_error":
            if self.state_ref is not None or not self.source_ref:
                raise ValueError("input errors require a source reference")
            if not isinstance(self.source_ref, str) or not re.fullmatch(
                r".+:byte=\d+,line=\d+", self.source_ref
            ):
                raise ValueError("input errors require a canonical source reference")
        elif self.state_ref is not None or self.source_ref is not None:
            raise ValueError("skip summaries cannot identify a state")
        if self.error.kind in {"scan_cap", "context_limit", "input_error"} and (
            self.meta.cache != "not_applicable"
        ):
            raise ValueError("skip and input errors are not cacheable")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "record_type": "error",
            "state_ref": self.state_ref,
        }
        if self.state_ref is None:
            result["source_ref"] = self.source_ref
        result["error"] = self.error.to_dict()
        result["meta"] = self.meta.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    coverage: CoverageStatus
    coverage_counts: Mapping[str, int]
    coverage_reasons: tuple[CoverageReason, ...]
    meta: RecordMeta

    def __post_init__(self) -> None:
        if self.coverage not in {"complete", "partial"}:
            raise ValueError(f"unknown coverage status: {self.coverage}")
        required = {"discovered", "judged", "emitted", "skipped", "failed"}
        if set(self.coverage_counts) != required:
            raise ValueError("coverage counts must have the five required fields")
        if any(count < 0 for count in self.coverage_counts.values()):
            raise ValueError("coverage counts must not be negative")
        if any(
            reason
            not in {
                "scan_cap",
                "input_error",
                "context_limit",
                "api_error",
                "malformed_answer",
                "partial_answer",
            }
            for reason in self.coverage_reasons
        ):
            raise ValueError("unknown coverage reason")
        if self.coverage == "complete" and self.coverage_reasons:
            raise ValueError("complete coverage cannot have reasons")
        if self.coverage == "complete" and self.coverage_counts["failed"]:
            raise ValueError("complete coverage cannot have failed states")
        if self.coverage == "complete" and self.coverage_counts["skipped"]:
            raise ValueError("complete coverage cannot have skipped states")
        if self.coverage_counts["discovered"] != (
            self.coverage_counts["judged"] + self.coverage_counts["skipped"]
        ):
            raise ValueError("coverage counts do not account for every state")
        if self.coverage_counts["failed"] > self.coverage_counts["judged"]:
            raise ValueError("failed states cannot exceed judged states")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "coverage",
            "coverage": self.coverage,
            "coverage_counts": dict(self.coverage_counts),
            "coverage_reasons": list(self.coverage_reasons),
            "meta": self.meta.to_dict(),
        }


type CanonicalRecord = (
    ResultRecord | PartialResultRecord | ErrorRecord | CoverageRecord
)


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    if isinstance(answer, NoulAnswer):
        return {"type": "noul", "noul": answer.noul}
    if isinstance(answer, ChoiceAnswer):
        return {
            "type": "choice",
            "choice": answer.choice,
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
        }
    if isinstance(answer, ScoreAnswer):
        return {
            "type": "score",
            "score": answer.score,
            "legend": dict(answer.legend),
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
        }
    raise TypeError(f"unsupported answer type: {type(answer).__name__}")


def answers_to_dict(answers: Mapping[str, Answer]) -> dict[str, dict[str, Any]]:
    return {
        question_id: answer_to_dict(answer) for question_id, answer in answers.items()
    }


def record_to_dict(record: CanonicalRecord) -> dict[str, Any]:
    return record.to_dict()


def parse_judge_response(
    payload: Any, questions: Mapping[str, Any]
) -> JudgeResponse:
    """Validate one complete TypeSafe response against its requested battery."""
    if not isinstance(payload, Mapping):
        raise ValueError("response must be an object")
    answers_payload = payload.get("answers")
    if not isinstance(answers_payload, Mapping):
        raise ValueError("response requires an answers object")

    answers: dict[str, Answer] = {}
    for question_id, raw_answer in answers_payload.items():
        if question_id not in questions:
            raise ValueError(f"unknown question id: {question_id}")
        answers[question_id] = _parse_answer(
            raw_answer, _question_type(questions[question_id]), question_id
        )

    missing = tuple(
        question_id for question_id in questions if question_id not in answers
    )
    return JudgeResponse(answers=answers, missing_questions=missing)


def _question_type(question: Any) -> str:
    if isinstance(question, Mapping):
        question_type = question.get("type")
    else:
        question_type = getattr(question, "type", None)
    if not isinstance(question_type, str):
        raise ValueError("question type must be a string")
    return question_type


def _parse_answer(raw_answer: Any, question_type: str, question_id: str) -> Answer:
    if not isinstance(raw_answer, Mapping):
        raise ValueError(f"answer for {question_id} must be an object")
    answer_type = raw_answer.get("type")
    if answer_type != question_type:
        raise ValueError(
            f"answer for {question_id} has type {answer_type!r}, "
            f"expected {question_type!r}"
        )

    if question_type == "noul":
        _require_keys(raw_answer, {"type", "noul"}, question_id)
        _require_number(raw_answer.get("noul"), f"noul answer for {question_id}")
        return NoulAnswer(float(raw_answer["noul"]))
    if question_type == "choice":
        _require_keys(
            raw_answer,
            {"type", "choice", "probabilities", "confidence"},
            question_id,
        )
        choice = raw_answer.get("choice")
        if not isinstance(choice, str):
            raise ValueError(
                f"choice answer for {question_id} requires a string choice"
            )
        probabilities = _probabilities(raw_answer.get("probabilities"), question_id)
        confidence = _number(raw_answer.get("confidence"), question_id, "confidence")
        return ChoiceAnswer(choice, probabilities, confidence)
    if question_type == "score":
        _require_keys(
            raw_answer,
            {"type", "score", "legend", "probabilities", "confidence"},
            question_id,
        )
        score = _number(raw_answer.get("score"), question_id, "score")
        legend = raw_answer.get("legend")
        if not isinstance(legend, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in legend.items()
        ):
            raise ValueError(f"score answer for {question_id} requires a string legend")
        probabilities = _probabilities(raw_answer.get("probabilities"), question_id)
        confidence = _number(raw_answer.get("confidence"), question_id, "confidence")
        return ScoreAnswer(score, dict(legend), probabilities, confidence)
    raise ValueError(f"unknown question type for {question_id}: {question_type!r}")


def _require_keys(
    answer: Mapping[str, Any], expected: set[str], question_id: str
) -> None:
    actual = set(answer)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"answer for {question_id} missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(
            f"answer for {question_id} has unknown fields: {sorted(extra)}"
        )


def _number(value: Any, question_id: str, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} for {question_id} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} for {question_id} must be finite")
    return converted


def _require_number(value: Any, description: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{description} must be finite")


def _probabilities(value: Any, question_id: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"probabilities for {question_id} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"probability keys for {question_id} must be strings")
    return {
        key: _number(probability, question_id, "probability")
        for key, probability in value.items()
    }
