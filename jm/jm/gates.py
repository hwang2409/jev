from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .answers import (
    ChoiceAnswer,
    NoulAnswer,
    ResultRecord,
    ScoreAnswer,
    score_argmax,
)
from .presets import Preset, validate_preset


class PolicyError(ValueError):
    """A policy is invalid or cannot be evaluated."""

    exit_code = 64


class PolicySyntaxError(PolicyError):
    """A policy does not match the v1 grammar."""


class PolicyValidationError(PolicyError):
    """A policy does not match the preset's typed fields or thresholds."""


class IndeterminateGate(PolicyError):
    """A consistency interval overlaps a gate threshold."""


LiteralValue = float | int | str | bool
ComparisonOperator = Literal[">=", ">", "<=", "<", "==", "!="]


@dataclass(frozen=True, slots=True)
class Comparison:
    question_id: str
    answer_field: str
    operator: ComparisonOperator
    value: LiteralValue


@dataclass(frozen=True, slots=True)
class Not:
    expression: Expression


@dataclass(frozen=True, slots=True)
class Boolean:
    operator: Literal["and", "or"]
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Aggregate:
    operator: Literal["any", "all"]
    expression: Expression


Expression = Comparison | Not | Boolean | Aggregate


@dataclass(frozen=True, slots=True)
class Policy:
    source: str
    expression: Expression


@dataclass(frozen=True, slots=True)
class GateResult:
    exit_code: Literal[0, 1, 2]
    failed: bool
    fail_closed: bool
    judged_states: int
    required_states: int
    reason: str | None = None


_TOKEN = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<number>(?:0|[1-9][0-9]*)(?:\.[0-9]+)?)"
    r"|(?P<string>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
    r"|(?P<operator>>=|<=|==|!=|>|<)"
    r"|(?P<punct>[().])"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*)"
)

# Maximum recursive nesting for `not` and parenthesized predicates.
MAX_POLICY_DEPTH = 64


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    value: str
    position: int


def parse_policy(source: str) -> Policy:
    if not isinstance(source, str) or not source.strip():
        raise PolicySyntaxError("policy must be a non-empty string")
    tokens = _tokenize(source)
    parser = _Parser(tokens, source)
    if not (parser._peek_word("any") or parser._peek_word("all")):
        parser._error(
            "policy must have exactly one outer any(...) or all(...) wrapper"
        )
    operator = parser._take().value
    parser._expect_punct("(")
    expression = parser.parse_expression()
    parser._expect_punct(")")
    if parser.index != len(tokens):
        parser._error(
            "policy must contain exactly one outer aggregate wrapper",
            tokens[parser.index],
        )
    return Policy(source, Aggregate(operator, expression))  # type: ignore[arg-type]


def compile_policy(source: str | Policy, preset: Preset | Mapping[str, Any]) -> Policy:
    policy = parse_policy(source) if isinstance(source, str) else source
    if not isinstance(policy, Policy):
        raise TypeError("policy must be a policy expression or Policy")
    _validate_policy_shape(policy.expression)
    data = preset.data if isinstance(preset, Preset) else validate_preset(preset)
    _validate_expression(policy.expression, data)
    return policy


def evaluate_policy(
    policy: Policy | str,
    records: Sequence[ResultRecord],
    *,
    consistency_sigma: float = 2.0,
) -> bool:
    if isinstance(policy, str):
        policy = parse_policy(policy)
    if not isinstance(policy, Policy):
        raise TypeError("policy must be a policy expression or Policy")
    _validate_policy_shape(policy.expression)
    return _evaluate(policy.expression, records, consistency_sigma)


def evaluate_gate(
    policy: Policy | str,
    records: Sequence[ResultRecord],
    *,
    judged_states: int | None = None,
    coverage_reasons: Sequence[str] = (),
    required_states: int = 1,
    consistency_sigma: float = 2.0,
) -> GateResult:
    if (
        isinstance(consistency_sigma, bool)
        or not isinstance(consistency_sigma, (int, float))
        or not math.isfinite(float(consistency_sigma))
        or consistency_sigma < 0
    ):
        raise PolicyError("consistency sigma must be finite and non-negative")
    if required_states < 0:
        raise PolicyError("require_states must be non-negative")
    judged = len(records) if judged_states is None else judged_states
    if judged < 0:
        raise PolicyError("judged_states must be non-negative")
    if coverage_reasons:
        return GateResult(
            2, False, True, judged, required_states, "incomplete coverage"
        )
    if judged < required_states:
        return GateResult(
            2, False, True, judged, required_states, "too few judged states"
        )
    try:
        failed = evaluate_policy(
            policy,
            records,
            consistency_sigma=float(consistency_sigma),
        )
    except IndeterminateGate:
        return GateResult(
            2, False, True, judged, required_states, "indeterminate consistency"
        )
    return GateResult(1 if failed else 0, failed, False, judged, required_states)


class _Parser:
    def __init__(self, tokens: Sequence[_Token], source: str) -> None:
        self.tokens = tokens
        self.source = source
        self.index = 0
        self.depth = 0
        self.complexity = 0

    def parse_expression(self) -> Expression:
        return self.parse_or()

    def parse_or(self) -> Expression:
        expression = self.parse_and()
        while self._accept_word("or"):
            expression = Boolean("or", expression, self.parse_and())
        return expression

    def parse_and(self) -> Expression:
        expression = self.parse_not()
        while self._accept_word("and"):
            expression = Boolean("and", expression, self.parse_not())
        return expression

    def parse_not(self) -> Expression:
        if self._accept_word("not"):
            self._enter_depth()
            try:
                return Not(self.parse_not())
            finally:
                self.depth -= 1
        return self.parse_primary()

    def parse_primary(self) -> Expression:
        if self._accept_punct("("):
            self._enter_depth()
            try:
                expression = self.parse_expression()
                self._expect_punct(")")
                return expression
            finally:
                self.depth -= 1
        if self._peek_word("any") or self._peek_word("all"):
            self._error("aggregate wrappers must be the outer policy expression")
        return self.parse_comparison()

    def parse_comparison(self) -> Comparison:
        self._consume_complexity()
        question_id = self._expect_kind("word").value
        self._expect_punct(".")
        answer_field = self._expect_kind("word").value
        operator_token = self._take()
        if operator_token.kind == "operator" and operator_token.value in {
            ">=",
            ">",
            "<=",
            "<",
            "==",
            "!=",
        }:
            operator = operator_token.value  # type: ignore[assignment]
            value = self.parse_literal()
        else:
            self._error("expected a comparison operator")
        return Comparison(question_id, answer_field, operator, value)

    def parse_literal(self) -> LiteralValue:
        token = self._take()
        if token.kind == "number":
            return float(token.value) if "." in token.value else int(token.value)
        if token.kind == "string":
            try:
                value = ast.literal_eval(token.value)
            except (SyntaxError, ValueError) as exc:
                raise PolicySyntaxError(
                    self._message("invalid quoted string", token)
                ) from exc
            if isinstance(value, str):
                return value
        if token.kind == "word" and token.value in {"true", "false"}:
            return token.value == "true"
        self._error("expected a number, quoted string, true, or false", token)

    def _take(self) -> _Token:
        if self.index == len(self.tokens):
            raise PolicySyntaxError("unexpected end of policy")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _expect_kind(self, kind: str) -> _Token:
        token = self._take()
        if token.kind != kind:
            self._error(f"expected {kind}", token)
        return token

    def _expect_punct(self, value: str) -> None:
        token = self._take()
        if token.kind != "punct" or token.value != value:
            self._error(f"expected {value!r}", token)

    def _accept_punct(self, value: str) -> bool:
        if self.index < len(self.tokens):
            token = self.tokens[self.index]
            if token.kind == "punct" and token.value == value:
                self.index += 1
                return True
        return False

    def _accept_word(self, value: str) -> bool:
        if self._peek_word(value):
            self.index += 1
            return True
        return False

    def _peek_word(self, value: str) -> bool:
        return (
            self.index < len(self.tokens)
            and self.tokens[self.index].kind == "word"
            and self.tokens[self.index].value == value
        )

    def _enter_depth(self) -> None:
        self.depth += 1
        self._consume_complexity()
        if self.depth > MAX_POLICY_DEPTH:
            self._error(f"policy nesting exceeds maximum depth {MAX_POLICY_DEPTH}")

    def _consume_complexity(self) -> None:
        self.complexity += 1
        if self.complexity > MAX_POLICY_DEPTH:
            self._error(
                f"policy expression exceeds maximum depth {MAX_POLICY_DEPTH}"
            )

    def _error(self, message: str, token: _Token | None = None) -> None:
        raise PolicySyntaxError(self._message(message, token))

    def _message(self, message: str, token: _Token | None = None) -> str:
        position = len(self.source) if token is None else token.position
        return f"{message} at position {position}"


def _tokenize(source: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    position = 0
    while position < len(source):
        match = _TOKEN.match(source, position)
        if match is None:
            raise PolicySyntaxError(f"unexpected character at position {position}")
        kind = match.lastgroup
        if kind != "space":
            tokens.append(_Token(kind, match.group(), position))
        position = match.end()
    return tuple(tokens)


def _validate_expression(expression: Expression, preset: Mapping[str, Any]) -> None:
    pending = [expression]
    while pending:
        current = pending.pop()
        if isinstance(current, Comparison):
            _validate_comparison(current, preset)
        elif isinstance(current, Not):
            pending.append(current.expression)
        elif isinstance(current, Boolean):
            pending.extend((current.left, current.right))
        elif isinstance(current, Aggregate):
            pending.append(current.expression)


def _validate_comparison(comparison: Comparison, preset: Mapping[str, Any]) -> None:
    questions = preset["questions"]
    if comparison.question_id not in questions:
        raise PolicyValidationError(
            f"policy references unknown question {comparison.question_id!r}"
        )
    question_type = questions[comparison.question_id]["type"]
    expected_field = {
        "noul": "noul",
        "score": "score",
        "choice": "choice",
    }[question_type]
    if comparison.answer_field != expected_field:
        raise PolicyValidationError(
            f"{comparison.question_id} is a {question_type} field; "
            f"use {comparison.question_id}.{expected_field}"
        )
    if question_type == "choice":
        _validate_choice_comparison(comparison)
        return
    if (
        isinstance(comparison.value, bool)
        or not isinstance(comparison.value, (int, float))
    ):
        raise PolicyValidationError(
            f"{question_type} comparisons require a numeric pinned threshold"
        )
    threshold = preset["thresholds"].get(comparison.question_id)
    if threshold is None:
        raise PolicyValidationError(
            f"{comparison.question_id} has no pinned threshold"
        )
    pinned = threshold.get("keep_at_least", threshold.get("fail_at_least"))
    if comparison.value != pinned:
        raise PolicyValidationError(
            f"{comparison.question_id} must use pinned threshold {pinned!r}"
        )


def _validate_choice_comparison(comparison: Comparison) -> None:
    if comparison.operator not in {"==", "!="} or not isinstance(
        comparison.value, str
    ):
        raise PolicyValidationError(
            "choice fields support string equality or inequality"
        )


def _validate_policy_shape(expression: Expression) -> None:
    if not isinstance(expression, Aggregate):
        raise PolicySyntaxError(
            "policy must have exactly one outer any(...) or all(...) wrapper"
        )
    if _contains_aggregate(expression.expression):
        raise PolicySyntaxError(
            "aggregate wrappers must be the outer policy expression"
        )


def _contains_aggregate(expression: Expression) -> bool:
    pending = [expression]
    while pending:
        current = pending.pop()
        if isinstance(current, Aggregate):
            return True
        if isinstance(current, Not):
            pending.append(current.expression)
        elif isinstance(current, Boolean):
            pending.extend((current.left, current.right))
    return False


def _evaluate(
    expression: Expression,
    records: Sequence[ResultRecord],
    consistency_sigma: float,
) -> bool:
    if not isinstance(expression, Aggregate):
        raise PolicyError("aggregate must be the outer policy expression")
    values = [
        _evaluate_one(expression.expression, record, consistency_sigma)
        for record in records
    ]
    return any(values) if expression.operator == "any" else all(values)


def _evaluate_one(
    expression: Expression,
    record: ResultRecord,
    consistency_sigma: float,
) -> bool:
    if isinstance(expression, Comparison):
        answer = record.answers.get(expression.question_id)
        if answer is None:
            raise PolicyError(f"missing answer {expression.question_id!r}")
        actual = _answer_value(answer, expression.value, consistency_sigma)
        return _compare(actual, expression.operator, expression.value)
    if isinstance(expression, Not):
        return not _evaluate_one(expression.expression, record, consistency_sigma)
    if isinstance(expression, Boolean):
        left = _evaluate_one(expression.left, record, consistency_sigma)
        right = _evaluate_one(expression.right, record, consistency_sigma)
        if expression.operator == "and":
            return left and right
        return left or right
    raise PolicyError("aggregate must be the outer policy expression")


def _answer_value(
    answer: NoulAnswer | ChoiceAnswer | ScoreAnswer,
    threshold: LiteralValue,
    consistency_sigma: float,
) -> float | str:
    if isinstance(answer, NoulAnswer):
        consistency = answer.consistency
        if consistency is not None:
            if not isinstance(consistency, Mapping) or set(consistency) != {
                "samples",
                "mean",
                "stddev",
            }:
                raise PolicyError("invalid consistency metadata")
            mean = consistency["mean"]
            stddev = consistency["stddev"]
            samples = consistency["samples"]
            if (
                isinstance(samples, bool)
                or not isinstance(samples, int)
                or samples < 2
                or isinstance(mean, bool)
                or not isinstance(mean, (int, float))
                or not math.isfinite(float(mean))
                or isinstance(stddev, bool)
                or not isinstance(stddev, (int, float))
                or not math.isfinite(float(stddev))
                or stddev < 0
            ):
                raise PolicyError("invalid consistency metadata")
            if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
                lower = float(mean) - consistency_sigma * float(stddev)
                upper = float(mean) + consistency_sigma * float(stddev)
                threshold_value = float(threshold)
                if lower <= threshold_value <= upper:
                    raise IndeterminateGate(
                        "consistency interval overlaps the threshold"
                    )
            return float(mean)
        return answer.noul
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    if isinstance(answer, ScoreAnswer):
        return score_argmax(answer)
    raise PolicyError(f"unsupported typed answer {type(answer).__name__}")


def _compare(
    actual: float | str, operator: ComparisonOperator, expected: LiteralValue
) -> bool:
    if operator in {"==", "!=", ">=", ">", "<=", "<"}:
        if operator == "==":
            return actual == expected
        if operator == "!=":
            return actual != expected
        if not isinstance(actual, (int, float)) or not isinstance(
            expected, (int, float)
        ):
            raise PolicyError("ordered comparison requires numeric typed answers")
        return {
            ">=": actual >= expected,
            ">": actual > expected,
            "<=": actual <= expected,
            "<": actual < expected,
        }[operator]
    raise PolicyError(f"unsupported comparison operator {operator!r}")
