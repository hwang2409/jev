from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


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
