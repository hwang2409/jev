from __future__ import annotations

from copy import deepcopy

import pytest

from jmap.answers import ChoiceAnswer, NoulAnswer, RecordMeta, ResultRecord, ScoreAnswer
from jmap.gates import (
    MAX_POLICY_DEPTH,
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
        RecordMeta("jgrep", "1", "typesafe-ai/jev", "para", "not_applicable"),
    )


def test_parser_uses_not_and_and_or_precedence_and_left_associativity() -> None:
    parsed = parse_policy(
        "any(not a.noul >= 0.75 and b.noul < 0.75 or c.noul == 0.75)"
    )
    assert isinstance(parsed.expression, Aggregate)
    assert isinstance(parsed.expression.expression, Boolean)
    assert parsed.expression.expression.operator == "or"
    assert isinstance(parsed.expression.expression.left, Boolean)
    assert parsed.expression.expression.left.operator == "and"
    assert isinstance(parsed.expression.expression.left.left, Not)

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

    left_associative = parse_policy(
        "any(a.noul >= 0.75 or b.noul >= 0.75 or c.noul >= 0.75)"
    )
    assert isinstance(left_associative.expression, Aggregate)
    assert isinstance(left_associative.expression.expression, Boolean)
    assert left_associative.expression.expression.operator == "or"
    assert isinstance(left_associative.expression.expression.left, Boolean)
    assert left_associative.expression.expression.left.operator == "or"


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


def test_parser_accepts_numeric_string_and_boolean_literals() -> None:
    assert parse_policy("any(a.noul == 0.75)")
    assert parse_policy("any(a.choice == 'yes')")
    assert parse_policy("any(a.choice == true)")


@pytest.mark.parametrize(
    "source",
    [
        "a.noul >= 0.75",
        "not any(a.noul >= 0.75)",
        "any(any(a.noul >= 0.75))",
        "any(a.noul >= 0.75) and a.noul >= 0.75",
        "(any(a.noul >= 0.75))",
    ],
)
def test_parser_requires_one_outer_aggregate(source: str) -> None:
    with pytest.raises(PolicySyntaxError, match="outer|exactly one"):
        parse_policy(source)


@pytest.mark.parametrize(
    "source",
    [
        "any(" + "not " * (MAX_POLICY_DEPTH + 1) + "a.noul >= 0.75)",
        "any("
        + "(" * (MAX_POLICY_DEPTH + 1)
        + "a.noul >= 0.75"
        + ")" * (MAX_POLICY_DEPTH + 1)
        + ")",
    ],
)
def test_parser_rejects_overdeep_input(source: str) -> None:
    with pytest.raises(PolicySyntaxError, match="maximum depth"):
        parse_policy(source)


def test_parser_rejects_an_overlarge_boolean_chain() -> None:
    comparisons = ["a.noul >= 0.75"] * (MAX_POLICY_DEPTH + 1)

    with pytest.raises(PolicySyntaxError, match="maximum depth"):
        parse_policy("any(" + " and ".join(comparisons) + ")")


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


def test_typed_thresholds_and_choice_equality_evaluate_without_formatting() -> None:
    preset = _choice_preset()
    policy = compile_policy(
        "any(matches_query.noul >= 0.75 and kind.choice == 'yes')", preset
    )
    record = _record(matches_query=NoulAnswer(0.8), kind=ChoiceAnswer("yes"))
    assert evaluate_policy(policy, [record]) is True

    score_policy = compile_policy(
        "any(change_scope.score >= 2)", resolve_preset("diff-risk-heat")
    )
    score_record = ResultRecord(
        "hunk#1",
        {"change_scope": ScoreAnswer(2.0)},
        RecordMeta("diff-risk-heat", "1", "typesafe-ai/jev", "hunk", "miss"),
    )
    assert evaluate_policy(score_policy, [score_record]) is True


@pytest.mark.parametrize(
    ("wrapper", "required_states", "expected_exit"),
    [("any", 0, 0), ("any", 1, 2), ("all", 0, 1), ("all", 1, 2)],
)
def test_vacuous_aggregate_results_respect_required_states(
    wrapper: str, required_states: int, expected_exit: int
) -> None:
    policy = compile_policy(
        f"{wrapper}(matches_query.noul >= 0.75)", resolve_preset("jgrep")
    )
    result = evaluate_gate(policy, [], required_states=required_states)
    assert result.exit_code == expected_exit


def test_choice_threshold_comparison_is_a_policy_type_error() -> None:
    with pytest.raises(PolicyValidationError, match="choice fields"):
        compile_policy("any(kind.choice >= 1)", _choice_preset())


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
