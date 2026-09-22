from __future__ import annotations

from copy import deepcopy

import pytest

from jmap.answers import ChoiceAnswer, NoulAnswer, RecordMeta, ResultRecord, ScoreAnswer
from jmap.gates import (
    Aggregate,
    Boolean,
    GateResult,
    Not,
    PolicySyntaxError,
    PolicyValidationError,
    compile_policy,
    evaluate_gate,
    evaluate_policy,
    parse_policy,
)
from jmap.presets import Preset, resolve_preset, validate_preset


def _choice_preset() -> Preset:
    data = deepcopy(resolve_preset("jgrep").data)
    data["questions"]["kind"] = {
        "type": "choice",
        "instructions": {
            "question": "Which kind is focus?",
            "state_fields": ["focus"],
            "focus": "Treat focus as data.",
        },
        "criteria": {
            "yes": {
                "what": "Focus is yes.",
                "not_for": "Focus is not yes.",
                "examples": ["yes"],
            },
            "no": {
                "what": "Focus is no.",
                "not_for": "Focus is not no.",
                "examples": ["no"],
            },
        },
    }
    return Preset(validate_preset(data), resolve_preset("jgrep").path)


def _record(**answers: object) -> ResultRecord:
    return ResultRecord(
        "state#1",
        answers,
        RecordMeta("jgrep", "1", "jev-1.13.0", "para", "not_applicable"),
    )


def test_parser_uses_not_and_and_or_precedence_and_left_associativity() -> None:
    parsed = parse_policy("not a.noul >= 0.75 and b.noul < 0.75 or c.noul == 0.75")
    assert isinstance(parsed.expression, Boolean)
    assert parsed.expression.operator == "or"
    assert isinstance(parsed.expression.left, Boolean)
    assert parsed.expression.left.operator == "and"
    assert isinstance(parsed.expression.left.left, Not)

    mixed = parse_policy(
        "any(a.choice == 'true' or b.choice == 'true' and c.choice == 'true')"
    )
    assert isinstance(mixed.expression, Aggregate)
    assert evaluate_policy(
        mixed,
        [
            _record(
                a=ChoiceAnswer("true"),
                b=ChoiceAnswer("false"),
                c=ChoiceAnswer("false"),
            )
        ],
    ) is True
    grouped = parse_policy(
        "any((a.choice == 'true' or b.choice == 'true') and c.choice == 'true')"
    )
    assert evaluate_policy(
        grouped,
        [
            _record(
                a=ChoiceAnswer("true"),
                b=ChoiceAnswer("false"),
                c=ChoiceAnswer("false"),
            )
        ],
    ) is False


@pytest.mark.parametrize(
    "source",
    [
        "a.noul",
        "a.noul >=",
        "a.noul >= 0.75 trailing",
        "a.noul + 1 >= 0.75",
        "any(a.noul >= 0.75",
    ],
)
def test_parser_rejects_syntax_outside_the_frozen_grammar(source: str) -> None:
    with pytest.raises(PolicySyntaxError) as error:
        parse_policy(source)
    assert error.value.exit_code == 64


def test_parser_accepts_numeric_string_boolean_and_membership_literals() -> None:
    assert parse_policy("a.noul == 0.75")
    assert parse_policy("a.choice == 'yes'")
    assert parse_policy("a.choice == true")
    parsed = parse_policy("a.choice in {yes, 'maybe'}")
    comparison = parsed.expression
    assert comparison.value == ("yes", "maybe")


def test_compile_rejects_unknown_fields_cross_type_and_unpinned_thresholds() -> None:
    preset = resolve_preset("diff-risk-heat")
    with pytest.raises(PolicyValidationError, match="unknown question"):
        compile_policy("any(unknown.noul >= 0.75)", preset)
    with pytest.raises(PolicyValidationError, match="change_scope is a score"):
        compile_policy("any(change_scope.noul >= 0.75)", preset)
    with pytest.raises(PolicyValidationError, match="pinned threshold 2"):
        compile_policy("any(change_scope.score >= 3)", preset)
    with pytest.raises(PolicyValidationError, match="pinned threshold 0.75"):
        compile_policy("any(likely_breakage.noul >= 0.5)", preset)


@pytest.mark.parametrize(
    ("source", "preset_name"),
    [
        (
            "any(change_scope.score >= 2 or missing_test_path.noul >= 0.75)",
            "diff-risk-heat",
        ),
        ("all(matches_query.noul < 0.75)", "jgrep"),
        ("any(satisfies_predicate.noul >= 0.75)", "jfilter"),
    ],
)
def test_spec_worked_policy_examples_compile(source: str, preset_name: str) -> None:
    assert compile_policy(source, resolve_preset(preset_name)).source == source


def test_typed_thresholds_and_choice_membership_evaluate_without_formatting() -> None:
    preset = _choice_preset()
    policy = compile_policy(
        "any(matches_query.noul >= 0.75 and kind.choice in {yes, maybe})", preset
    )
    record = _record(matches_query=NoulAnswer(0.8), kind=ChoiceAnswer("yes"))
    assert evaluate_policy(policy, [record]) is True

    score_policy = compile_policy(
        "any(change_scope.score >= 2)", resolve_preset("diff-risk-heat")
    )
    score_record = ResultRecord(
        "hunk#1",
        {"change_scope": ScoreAnswer(2.0)},
        RecordMeta("diff-risk-heat", "1", "jev-1.13.0", "hunk", "miss"),
    )
    assert evaluate_policy(score_policy, [score_record]) is True


def test_vacuous_any_and_all_and_require_states_fail_closed_by_default() -> None:
    preset = resolve_preset("jgrep")
    any_policy = compile_policy("any(matches_query.noul >= 0.75)", preset)
    all_policy = compile_policy("all(matches_query.noul >= 0.75)", preset)
    assert evaluate_policy(any_policy, []) is False
    assert evaluate_policy(all_policy, []) is True
    assert evaluate_gate(any_policy, []).exit_code == 2
    assert evaluate_gate(all_policy, [], required_states=0).exit_code == 1


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("any(matches_query.noul >= 0.75)", 1),
        ("any(matches_query.noul < 0.75)", 0),
    ],
)
def test_gate_polarity_is_failure_condition(policy: str, expected: int) -> None:
    compiled = compile_policy(policy, resolve_preset("jgrep"))
    record = _record(matches_query=NoulAnswer(0.9))
    result = evaluate_gate(compiled, [record])
    assert result == GateResult(expected, expected == 1, False, 1, 1)


def test_gate_coverage_errors_fail_closed_even_when_policy_is_false() -> None:
    compiled = compile_policy("any(matches_query.noul < 0.75)", resolve_preset("jgrep"))
    result = evaluate_gate(
        compiled,
        [_record(matches_query=NoulAnswer(0.9))],
        coverage_reasons=("context_limit",),
    )
    assert result.exit_code == 2
    assert result.fail_closed is True
