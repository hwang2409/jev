from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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

    @property
    def message(self) -> str:
        return self.error

    @property
    def complete(self) -> bool:
        return False


type TypedResponse = JudgeResponse | ErrorResponse
