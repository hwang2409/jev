from __future__ import annotations

import pytest

from jm.answers import (
    ChoiceAnswer,
    CoverageRecord,
    ErrorDetail,
    ErrorRecord,
    NoulAnswer,
    PartialResultRecord,
    RecordMeta,
    ResultRecord,
    ScoreAnswer,
    SkipSummary,
    answers_to_dict,
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


def test_parse_gateway_answers_derives_optional_fields() -> None:
    response = parse_judge_response(
        {
            "answers": {
                "kind": {
                    "type": "choice",
                    "choice": "yes",
                    "probabilities": {"yes": 0.8, "no": 0.2},
                },
                "risk": {
                    "type": "score",
                    "score": 2.5,
                    "probabilities": {"0": 0.1, "3": 0.9},
                },
            }
        },
        {"kind": {"type": "choice"}, "risk": {"type": "score"}},
    )

    assert response.answers["kind"].confidence == pytest.approx(0.6)
    assert response.answers["risk"].legend == {}
    assert response.answers["risk"].confidence == 0.8


@pytest.mark.parametrize(
    "question_id,answer",
    [
        ("matches", {"type": "noul"}),
        ("kind", {"type": "choice", "choice": "yes", "confidence": 0.5}),
        ("risk", {"type": "score", "score": 1, "legend": {}}),
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


@pytest.mark.parametrize(
    "question_id,answer",
    [
        (
            "kind",
            {
                "type": "choice",
                "choice": "yes",
                "probabilities": {"yes": 1.0},
                "confidence": None,
            },
        ),
        (
            "risk",
            {
                "type": "score",
                "score": 1.0,
                "legend": {},
                "probabilities": {},
                "confidence": None,
            },
        ),
    ],
)
def test_parse_rejects_null_confidence(
    question_id: str, answer: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="confidence"):
        parse_judge_response(
            {"answers": {question_id: answer}},
            {question_id: QUESTIONS[question_id]},
        )


def test_canonical_records_preserve_each_typed_answer_shape() -> None:
    meta = RecordMeta("diff-risk-heat", "1", "typesafe-ai/jev", "hunk", "miss")
    answers = {
        "matches": NoulAnswer(0.75),
        "kind": ChoiceAnswer("yes", {"yes": 0.8, "no": 0.2}, 0.6),
        "risk": ScoreAnswer(
            2.5,
            {"0": "low", "3": "high"},
            {"0": 0.1, "3": 0.9},
            0.8,
        ),
    }

    assert answers_to_dict(answers) == {
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
    assert ResultRecord("state#1", answers, meta).to_dict() == {
        "record_type": "result",
        "state_ref": "state#1",
        "answers": answers_to_dict(answers),
        "meta": {
            "preset": "diff-risk-heat",
            "preset_version": "1",
            "model": "typesafe-ai/jev",
            "chunker": "hunk",
            "cache": "miss",
        },
    }


def test_canonical_partial_error_skip_and_coverage_shapes() -> None:
    meta = RecordMeta("jgrep", "1", "typesafe-ai/jev", "para", "miss")
    partial = PartialResultRecord(
        "notes#p1", {"matches": NoulAnswer(0.5)}, ("other",), meta
    )
    assert partial.to_dict()["meta"]["partial"] is True
    assert ErrorRecord(
        "notes#p1",
        ErrorDetail("api_error", "request failed", 503, 3),
        meta,
    ).to_dict() == {
        "record_type": "error",
        "state_ref": "notes#p1",
        "error": {
            "kind": "api_error",
            "message": "request failed",
            "http_status": 503,
            "attempts": 3,
        },
        "meta": meta.to_dict(),
    }
    assert ErrorRecord(
        None,
        ErrorDetail(
            "input_error", "invalid JSON record", skip_summary=None
        ),
        RecordMeta("jgrep", "1", "typesafe-ai/jev", "para", "not_applicable"),
        source_ref="stdin:byte=128,line=4",
    ).to_dict()["source_ref"] == "stdin:byte=128,line=4"
    skip = ErrorRecord(
        None,
        ErrorDetail(
            "scan_cap",
            "scan cap reached before visit",
            skip_summary=SkipSummary(
                "max_chunks=8", 9, tuple(f"p{i}" for i in range(9))
            ),
        ),
        RecordMeta("jgrep", "1", "typesafe-ai/jev", "para", "not_applicable"),
    )
    assert len(skip.to_dict()["error"]["skip_summary"]["sample_refs"]) == 8
    assert CoverageRecord(
        "partial",
        {"discovered": 10, "judged": 1, "emitted": 1, "skipped": 9, "failed": 0},
        ("scan_cap",),
        RecordMeta("jgrep", "1", "typesafe-ai/jev", "para", "not_applicable"),
    ).to_dict() == {
        "record_type": "coverage",
        "coverage": "partial",
        "coverage_counts": {
            "discovered": 10,
            "judged": 1,
            "emitted": 1,
            "skipped": 9,
            "failed": 0,
        },
        "coverage_reasons": ["scan_cap"],
        "meta": {
            "preset": "jgrep",
            "preset_version": "1",
            "model": "typesafe-ai/jev",
            "chunker": "para",
            "cache": "not_applicable",
        },
    }


def test_coverage_rejects_complete_when_states_are_skipped() -> None:
    with pytest.raises(ValueError, match="complete coverage cannot have skipped"):
        CoverageRecord(
            "complete",
            {"discovered": 1, "judged": 0, "emitted": 0, "skipped": 1, "failed": 0},
            (),
            RecordMeta("jgrep", "1", "typesafe-ai/jev", "para", "not_applicable"),
        )


@pytest.mark.parametrize("source_ref", ["stdin", "stdin:byte=1", "stdin:line=2"])
def test_input_errors_require_canonical_source_refs(source_ref: str) -> None:
    with pytest.raises(ValueError, match="source reference"):
        ErrorRecord(
            None,
            ErrorDetail("input_error", "invalid input"),
            RecordMeta("jgrep", "1", "typesafe-ai/jev", "para", "not_applicable"),
            source_ref=source_ref,
        )
