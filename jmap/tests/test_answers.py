from __future__ import annotations

import pytest

from jmap.answers import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    parse_judge_response,
)

QUESTIONS = {
    "matches": {"type": "noul"},
    "kind": {"type": "choice"},
    "risk": {"type": "score"},
}


def test_parse_valid_typed_answers() -> None:
    response = parse_judge_response(
        {
            "answers": {
                "matches": {"type": "noul", "noul": 0.75},
                "kind": {
                    "type": "choice",
                    "choice": "yes",
                    "probabilities": {"yes": 0.8, "no": 0.2},
                    "confidence": 0.6,
                },
                "risk": {
                    "type": "score",
                    "score": 2.5,
                    "legend": {"0": "low", "3": "high"},
                    "probabilities": {"0": 0.1, "3": 0.9},
                    "confidence": 0.8,
                },
            }
        },
        QUESTIONS,
    )

    assert response.answers == {
        "matches": NoulAnswer(0.75),
        "kind": ChoiceAnswer("yes", {"yes": 0.8, "no": 0.2}, 0.6),
        "risk": ScoreAnswer(
            2.5,
            {"0": "low", "3": "high"},
            {"0": 0.1, "3": 0.9},
            0.8,
        ),
    }


@pytest.mark.parametrize(
    "question_id,answer",
    [
        ("matches", {"type": "noul"}),
        ("kind", {"type": "choice", "choice": "yes", "confidence": 0.5}),
        ("risk", {"type": "score", "score": 1, "legend": {}, "probabilities": {}}),
    ],
)
def test_parse_rejects_missing_fields(
    question_id: str, answer: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="missing|requires"):
        parse_judge_response(
            {"answers": {question_id: answer}},
            {question_id: QUESTIONS[question_id]},
        )


def test_parse_rejects_unknown_question_id() -> None:
    with pytest.raises(ValueError, match="unknown question"):
        parse_judge_response(
            {"answers": {"other": {"type": "noul", "noul": 0.5}}},
            QUESTIONS,
        )


def test_parse_rejects_bad_primitive_types() -> None:
    with pytest.raises(ValueError, match="noul"):
        parse_judge_response(
            {"answers": {"matches": {"type": "noul", "noul": "0.5"}}},
            QUESTIONS,
        )


def test_parse_rejects_confidence_on_noul() -> None:
    with pytest.raises(ValueError, match="confidence"):
        parse_judge_response(
            {
                "answers": {
                    "matches": {"type": "noul", "noul": 0.5, "confidence": 0.9}
                }
            },
            QUESTIONS,
        )
