from __future__ import annotations

import math
import os
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ._transport import _GATEWAY_MODEL as GATEWAY_MODEL

SCHEMA = "jm.preset/v1"
SCHEMA_V2 = "jm.preset/v2"
CHUNKERS = frozenset({"line", "para", "hunk", "file", "record"})
QUESTION_TYPES = frozenset({"noul", "choice", "score"})
RESERVED_QUESTION_IDS = frozenset({"any", "all", "not"})
_QUESTION_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_PRESET_NAME = re.compile(r"^[a-z][a-z0-9-]*(?:\.(?:yml|yaml))?$")
_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "name",
        "version",
        "model",
        "chunking",
        "compatible_chunkers",
        "questions",
        "thresholds",
        "output",
    }
)
_OPTIONAL_FIELDS = frozenset({"description", "calibration"})
_PREFILTER_FIELDS = frozenset(
    {"ranker", "top", "query_source", "fields", "query"}
)
_OUTPUT_FIELDS = frozenset(
    {
        "record_type",
        "state_ref",
        "source_ref",
        "answers",
        "error",
        "missing_questions",
        "coverage",
        "coverage_counts",
        "coverage_reasons",
        "meta",
    }
)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.Loader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if isinstance(key, bool):
            key = str(key).lower()
        if key in mapping:
            raise yaml.YAMLError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class PresetError(ValueError):
    """Base error for invalid or unusable presets."""


class PresetValidationError(PresetError):
    """A preset does not match the v1 schema."""


class PresetUsageError(PresetError):
    """A valid preset cannot be used with the requested invocation options."""

    exit_code = 64


class PresetNotFoundError(FileNotFoundError):
    """No preset matched the requested lookup name or path."""


@dataclass(frozen=True, slots=True)
class Preset:
    data: Mapping[str, Any]
    path: Path

    @property
    def name(self) -> str:
        return str(self.data["name"])

    @property
    def schema(self) -> str:
        return str(self.data["schema"])

    @property
    def version(self) -> str:
        return str(self.data["version"])

    @property
    def model(self) -> str:
        return str(self.data["model"])

    @property
    def chunking(self) -> Mapping[str, Any]:
        return self.data["chunking"]

    @property
    def questions(self) -> Mapping[str, Any]:
        return self.data["questions"]

    @property
    def prefilter(self) -> Mapping[str, Any] | None:
        value = self.data.get("prefilter")
        return value if isinstance(value, Mapping) else None

    @property
    def compatible_chunkers(self) -> tuple[str, ...]:
        return tuple(self.data["compatible_chunkers"])

    @property
    def default_chunker(self) -> str:
        return str(self.chunking["by"])

    def effective_chunker(self, by: str | None = None) -> str:
        return resolve_chunker(self, by)


def validate_preset(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return one decoded preset mapping."""
    root = _mapping(data, "preset")
    schema = root.get("schema")
    allowed_fields = _REQUIRED_FIELDS | _OPTIONAL_FIELDS
    if schema == SCHEMA_V2:
        allowed_fields |= {"prefilter"}
    _reject_unknown(root, allowed_fields, "preset")
    _require_fields(root, _REQUIRED_FIELDS, "preset")

    if root["schema"] not in {SCHEMA, SCHEMA_V2}:
        raise PresetValidationError(
            f"schema must be {SCHEMA!r} or {SCHEMA_V2!r}"
        )
    _string(root["name"], "name")
    _string(root["version"], "version")
    model = _string(root["model"], "model")
    if model != GATEWAY_MODEL:
        raise PresetValidationError(f"model must be {GATEWAY_MODEL!r}")
    if "description" in root:
        _string(root["description"], "description")
    if "calibration" in root:
        _validate_calibration(root["calibration"])

    chunking = _mapping(root["chunking"], "chunking")
    _reject_unknown(
        chunking,
        {"by", "context_paragraphs", "context_lines", "max_chunks", "limits"},
        "chunking",
    )
    _require_fields(chunking, {"by", "limits"}, "chunking")
    by = _string(chunking["by"], "chunking.by")
    if by not in CHUNKERS:
        raise PresetValidationError(f"chunking.by must be one of {sorted(CHUNKERS)}")
    for field_name in ("context_paragraphs", "context_lines", "max_chunks"):
        if field_name in chunking:
            _nonnegative_integer(chunking[field_name], f"chunking.{field_name}")
    limits = _mapping(chunking["limits"], "chunking.limits")
    _reject_unknown(
        limits,
        {"focus_bytes", "context_field_bytes", "state_bytes"},
        "chunking.limits",
    )
    _require_fields(
        limits,
        {"focus_bytes", "context_field_bytes", "state_bytes"},
        "chunking.limits",
    )
    for field_name, value in limits.items():
        _positive_integer(value, f"chunking.limits.{field_name}")

    compatible = _string_list(root["compatible_chunkers"], "compatible_chunkers")
    if not compatible:
        raise PresetValidationError("compatible_chunkers must not be empty")
    if len(set(compatible)) != len(compatible):
        raise PresetValidationError("compatible_chunkers must not contain duplicates")
    unknown_chunkers = set(compatible) - CHUNKERS
    if unknown_chunkers:
        raise PresetValidationError(
            f"compatible_chunkers contains unknown values: {sorted(unknown_chunkers)}"
        )
    if by not in compatible:
        raise PresetValidationError("chunking.by must be in compatible_chunkers")

    questions = _mapping(root["questions"], "questions")
    if not questions:
        raise PresetValidationError("questions must not be empty")
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not _QUESTION_ID.fullmatch(question_id):
            raise PresetValidationError(
                f"question ID {question_id!r} must be lowercase and stable"
            )
        if question_id in RESERVED_QUESTION_IDS:
            raise PresetValidationError(
                f"question ID {question_id!r} is reserved for policy keywords"
            )
        _validate_question(question, question_id)

    if "prefilter" in root:
        if schema != SCHEMA_V2:
            raise PresetValidationError(
                f"prefilter requires schema {SCHEMA_V2!r}"
            )
        _validate_prefilter(root["prefilter"], questions)

    thresholds = _mapping(root["thresholds"], "thresholds")
    for question_id, threshold in thresholds.items():
        if question_id not in questions:
            raise PresetValidationError(
                f"threshold references unknown question {question_id!r}"
            )
        _validate_threshold(threshold, question_id, questions[question_id])

    output = _mapping(root["output"], "output")
    _reject_unknown(output, {"default_format", "pretty_template", "fields"}, "output")
    _require_fields(output, {"default_format", "pretty_template", "fields"}, "output")
    if output["default_format"] not in {"jsonl", "pretty"}:
        raise PresetValidationError("output.default_format must be jsonl or pretty")
    _string(output["pretty_template"], "output.pretty_template")
    _validate_pretty_template(output["pretty_template"], questions)
    output_fields = _string_list(output["fields"], "output.fields")
    unknown_output_fields = set(output_fields) - _OUTPUT_FIELDS
    if unknown_output_fields:
        raise PresetValidationError(
            f"output.fields contains unknown values: {sorted(unknown_output_fields)}"
        )

    return root


def _validate_prefilter(
    value: Any, questions: Mapping[str, Any]
) -> None:
    prefilter = _mapping(value, "prefilter")
    _reject_unknown(prefilter, _PREFILTER_FIELDS, "prefilter")
    _require_fields(
        prefilter,
        {"ranker", "top", "query_source", "fields"},
        "prefilter",
    )
    if prefilter["ranker"] != "bm25":
        raise PresetValidationError("prefilter.ranker must be 'bm25'")
    _positive_integer(prefilter["top"], "prefilter.top")
    query_source = _string(prefilter["query_source"], "prefilter.query_source")
    if query_source not in {
        "cli",
        "context.query",
        "context.predicate",
        "literal",
    }:
        raise PresetValidationError(
            "prefilter.query_source must be one of cli, context.query, "
            "context.predicate, literal"
        )
    fields = _string_list(prefilter["fields"], "prefilter.fields")
    if not fields:
        raise PresetValidationError("prefilter.fields must not be empty")
    if len(set(fields)) != len(fields):
        raise PresetValidationError("prefilter.fields must not contain duplicates")
    battery_fields = {
        field
        for question in questions.values()
        for field in question["instructions"]["state_fields"]
    }
    unknown_fields = set(fields) - battery_fields
    if unknown_fields:
        raise PresetValidationError(
            "prefilter.fields contains fields not in state_fields: "
            f"{sorted(unknown_fields)}"
        )
    if query_source == "literal":
        if "query" not in prefilter:
            raise PresetValidationError(
                "prefilter.query is required for literal query_source"
            )
        _string(prefilter["query"], "prefilter.query")
    elif "query" in prefilter:
        raise PresetValidationError(
            "prefilter.query is only valid for literal query_source"
        )


def _validate_pretty_template(
    template: str, questions: Mapping[str, Any]
) -> None:
    values: dict[str, Any] = {
        "record_type": "result",
        "state_ref": "state_ref",
        "answers": {},
        "meta": {
            "preset": "preset",
            "preset_version": "1",
            "model": GATEWAY_MODEL,
            "chunker": "para",
            "cache": "not_applicable",
        },
    }
    for question_id, question in questions.items():
        question_type = question["type"]
        answer: dict[str, Any] = {"type": question_type}
        if question_type == "noul":
            answer["noul"] = 0.5
        elif question_type == "choice":
            answer.update(
                choice="value", probabilities={}, confidence=0.5
            )
        elif question_type == "score":
            answer.update(score=1, legend={}, probabilities={}, confidence=0.5)
        values["answers"][question_id] = answer

    formatter = string.Formatter()
    try:
        fields = list(formatter.parse(template))
    except ValueError as exc:
        raise PresetValidationError(
            f"output.pretty_template is invalid: {exc}"
        ) from exc

    for _, field_name, format_spec, conversion in fields:
        if field_name is None:
            continue
        if not field_name:
            raise PresetValidationError(
                "output.pretty_template does not support positional fields"
            )
        value: Any = values
        for part in field_name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise PresetValidationError(
                    "output.pretty_template references unknown field "
                    f"{field_name!r}"
                )
            value = value[part]
        if "{" in format_spec or "}" in format_spec:
            raise PresetValidationError(
                "output.pretty_template does not support nested format fields"
            )
        if conversion:
            value = formatter.convert_field(value, conversion)
        try:
            format(value, format_spec)
        except (TypeError, ValueError) as exc:
            raise PresetValidationError(
                "output.pretty_template has an invalid format specifier "
                f"for field {field_name!r}: {exc}"
            ) from exc


def load_preset(path: str | os.PathLike[str]) -> Preset:
    """Read and validate a preset at an exact path."""
    resolved_path = Path(path).expanduser()
    try:
        data = yaml.load(
            resolved_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except OSError as exc:
        raise PresetNotFoundError(str(resolved_path)) from exc
    except yaml.YAMLError as exc:
        raise PresetValidationError(f"invalid YAML in {resolved_path}: {exc}") from exc
    try:
        validated = validate_preset(data)
    except PresetValidationError as exc:
        raise PresetValidationError(f"{resolved_path}: {exc}") from exc
    return Preset(validated, resolved_path.resolve())


def resolve_preset(
    name: str | os.PathLike[str],
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    package_dir: str | os.PathLike[str] | None = None,
    user_dir: str | os.PathLike[str] | None = None,
) -> Preset:
    """Resolve a preset in the specified v1 lookup order."""
    if explicit_path is not None:
        return load_preset(explicit_path)
    identifier = os.fspath(name)
    if isinstance(identifier, bytes) or not _PRESET_NAME.fullmatch(identifier):
        raise PresetNotFoundError(
            f"preset name must be a safe preset name, not a path: {name!s}"
        )

    search_cwd = Path(cwd) if cwd is not None else Path.cwd()
    builtins = (
        Path(package_dir)
        if package_dir is not None
        else Path(__file__).with_name("presets")
    )
    configured_user_dir = os.environ.get("JM_PRESETS")
    if user_dir is not None:
        user = Path(user_dir)
    elif configured_user_dir:
        user = Path(configured_user_dir).expanduser()
    else:
        user = Path("~/.config/jm/presets").expanduser()

    locations = (search_cwd, builtins, user)
    for directory in locations:
        for candidate in _candidate_paths(directory, identifier):
            if candidate.is_file():
                return load_preset(candidate)
    searched = ", ".join(str(directory) for directory in locations)
    raise PresetNotFoundError(f"preset {name!s} was not found in: {searched}")


def lookup_preset(name: str | os.PathLike[str], **kwargs: Any) -> Preset:
    """Compatibility alias for resolve_preset."""
    return resolve_preset(name, **kwargs)


def resolve_chunker(preset: Preset | Mapping[str, Any], by: str | None = None) -> str:
    """Return the effective chunker or raise a command usage error."""
    data = preset.data if isinstance(preset, Preset) else validate_preset(preset)
    allowed = tuple(data["compatible_chunkers"])
    effective = by if by is not None else str(data["chunking"]["by"])
    if effective not in allowed:
        allowed_text = ", ".join(allowed)
        raise PresetUsageError(
            f"chunker {effective!r} is incompatible with preset {data['name']!r}; "
            f"allowed set: [{allowed_text}]"
        )
    return effective


def check_chunker_compatibility(
    preset: Preset | Mapping[str, Any], by: str | None = None
) -> str:
    """Compatibility alias for resolve_chunker."""
    return resolve_chunker(preset, by)


def resolve_prefilter(
    preset: Preset,
    *,
    command: str,
    ranker: str | None = None,
    top: int | None = None,
    fields: str | None = None,
    query: str | None = None,
    invocation_query: str | None = None,
    invocation_predicate: str | None = None,
) -> dict[str, Any] | None:
    """Resolve explicit prefilter CLI values against a validated preset."""
    supplied = any(value is not None for value in (ranker, top, fields, query))
    if ranker is None:
        if supplied:
            raise PresetUsageError(
                "prefilter options require --prefilter bm25"
            )
        return None
    if ranker != "bm25":
        raise PresetUsageError("--prefilter must be bm25")
    if command == "gate":
        raise PresetUsageError("gate does not support prefiltering")

    declared = preset.prefilter
    if declared is None and preset.name == "diff-risk-heat":
        raise PresetUsageError(
            "diff-risk-heat requires a preset-declared review query"
        )
    if declared is None and fields is None:
        raise PresetUsageError(
            "--prefilter-fields is required without a preset prefilter"
        )

    resolved_top = top if top is not None else declared["top"] if declared else None
    if (
        resolved_top is None
        or isinstance(resolved_top, bool)
        or not isinstance(resolved_top, int)
        or resolved_top <= 0
    ):
        raise PresetUsageError("prefilter top must be a positive integer")

    if fields is not None:
        resolved_fields = [item.strip() for item in fields.split(",")]
        if not resolved_fields or any(not item for item in resolved_fields):
            raise PresetUsageError("prefilter fields must not be empty")
        if len(set(resolved_fields)) != len(resolved_fields):
            raise PresetUsageError("prefilter fields must not contain duplicates")
    else:
        resolved_fields = list(declared["fields"]) if declared else []

    battery_fields = {
        field
        for question in preset.questions.values()
        for field in question["instructions"]["state_fields"]
    }
    unknown_fields = set(resolved_fields) - battery_fields
    if unknown_fields:
        raise PresetUsageError(
            "prefilter fields are not in state_fields: "
            f"{sorted(unknown_fields)}"
        )
    if not resolved_fields:
        raise PresetUsageError("prefilter fields must not be empty")

    if query is not None:
        if not query.strip():
            raise PresetUsageError("--prefilter-query must not be empty")
        query_source = "cli"
        resolved_query = query
    elif declared is not None:
        query_source = declared["query_source"]
        resolved_query = declared.get("query")
    else:
        if command == "jgrep":
            query_source = "context.query"
            resolved_query = None
        elif command == "jfilter":
            query_source = "context.predicate"
            resolved_query = None
        else:
            raise PresetUsageError(
                "--prefilter-query is required without a preset query"
            )

    if query_source == "cli":
        if not isinstance(resolved_query, str) or not resolved_query.strip():
            raise PresetUsageError("--prefilter-query is required")
    elif query_source == "context.query":
        if not isinstance(invocation_query, str) or not invocation_query.strip():
            raise PresetUsageError(
                "prefilter query_source context.query requires --query"
            )
        resolved_query = invocation_query
    elif query_source == "context.predicate":
        if (
            not isinstance(invocation_predicate, str)
            or not invocation_predicate.strip()
        ):
            raise PresetUsageError(
                "prefilter query_source context.predicate requires --predicate"
            )
        resolved_query = invocation_predicate
    elif query_source == "literal":
        if not isinstance(resolved_query, str) or not resolved_query.strip():
            raise PresetUsageError("literal prefilter query must not be empty")
    else:
        raise PresetUsageError(f"unknown prefilter query source: {query_source}")

    if preset.name == "diff-risk-heat" and (
        declared is None or declared.get("query_source") != "literal"
    ):
        raise PresetUsageError(
            "diff-risk-heat requires a preset-declared review query"
        )

    return {
        "ranker": "bm25",
        "top": resolved_top,
        "query_source": query_source,
        "fields": tuple(resolved_fields),
        "query": resolved_query,
    }


def _candidate_paths(directory: Path, identifier: str) -> tuple[Path, ...]:
    path = directory / identifier
    if path.suffix in {".yml", ".yaml"}:
        return (path,)
    return (path, directory / f"{identifier}.yml", directory / f"{identifier}.yaml")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PresetValidationError(f"{field_name} must be an object")
    return value


def _require_fields(
    value: Mapping[str, Any], required: set[str] | frozenset[str], field_name: str
) -> None:
    missing = required - set(value)
    if missing:
        raise PresetValidationError(
            f"{field_name} is missing required fields: {sorted(missing)}"
        )


def _reject_unknown(
    value: Mapping[str, Any], allowed: set[str] | frozenset[str], field_name: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise PresetValidationError(
            f"{field_name} contains unknown fields: {sorted(unknown)}"
        )


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PresetValidationError(f"{field_name} must be a non-empty string")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PresetValidationError(f"{field_name} must be a list")
    result = [_string(item, f"{field_name}[]") for item in value]
    return result


def _positive_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PresetValidationError(f"{field_name} must be a positive integer")


def _nonnegative_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PresetValidationError(f"{field_name} must be a non-negative integer")


def _validate_question(value: Any, question_id: str) -> None:
    question = _mapping(value, f"questions.{question_id}")
    _reject_unknown(
        question,
        {"type", "instructions", "criteria"},
        f"questions.{question_id}",
    )
    _require_fields(
        question,
        {"type", "instructions", "criteria"},
        f"questions.{question_id}",
    )
    question_type = _string(question["type"], f"questions.{question_id}.type")
    if question_type not in QUESTION_TYPES:
        raise PresetValidationError(
            f"questions.{question_id}.type must be one of {sorted(QUESTION_TYPES)}"
        )
    instructions = _mapping(
        question["instructions"], f"questions.{question_id}.instructions"
    )
    _reject_unknown(
        instructions,
        {"question", "state_fields", "focus"},
        f"questions.{question_id}.instructions",
    )
    _require_fields(
        instructions,
        {"question", "state_fields", "focus"},
        f"questions.{question_id}.instructions",
    )
    _string(instructions["question"], f"questions.{question_id}.instructions.question")
    state_fields = _string_list(
        instructions["state_fields"],
        f"questions.{question_id}.instructions.state_fields",
    )
    if "focus" not in state_fields:
        raise PresetValidationError(
            f"questions.{question_id}.instructions.state_fields must name focus"
        )
    if any(
        field != "focus" and not field.startswith("context.")
        for field in state_fields
    ):
        raise PresetValidationError(
            f"questions.{question_id}.instructions.state_fields must use "
            "literal context fields"
        )
    _string(instructions["focus"], f"questions.{question_id}.instructions.focus")
    _validate_criteria(question["criteria"], question_type, question_id)


def _validate_criteria(value: Any, question_type: str, question_id: str) -> None:
    field_name = f"questions.{question_id}.criteria"
    if question_type == "score":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise PresetValidationError(f"{field_name} must be a list for score")
        if len(value) != 4:
            raise PresetValidationError(f"{field_name} must have four score levels")
        for index, criterion in enumerate(value):
            _validate_criterion(criterion, f"{field_name}[{index}]")
        return
    criteria = _mapping(value, field_name)
    if question_type == "noul":
        criterion_keys = set(criteria)
        if criterion_keys == {True, False}:
            criteria = {
                str(key).lower(): criterion for key, criterion in criteria.items()
            }
        if set(criteria) != {"true", "false"}:
            raise PresetValidationError(f"{field_name} must contain true and false")
    if not criteria:
        raise PresetValidationError(f"{field_name} must not be empty")
    for polarity, criterion in criteria.items():
        _validate_criterion(criterion, f"{field_name}.{polarity}")


def _validate_criterion(value: Any, field_name: str) -> None:
    criterion = _mapping(value, field_name)
    _reject_unknown(criterion, {"what", "not_for", "examples"}, field_name)
    _require_fields(criterion, {"what", "not_for", "examples"}, field_name)
    _string(criterion["what"], f"{field_name}.what")
    _string(criterion["not_for"], f"{field_name}.not_for")
    examples = _string_list(criterion["examples"], f"{field_name}.examples")
    if not examples:
        raise PresetValidationError(f"{field_name}.examples must not be empty")


def _validate_threshold(value: Any, question_id: str, question: Any) -> None:
    field_name = f"thresholds.{question_id}"
    threshold = _mapping(value, field_name)
    _reject_unknown(threshold, {"type", "keep_at_least", "fail_at_least"}, field_name)
    _require_fields(threshold, {"type"}, field_name)
    question_type = _mapping(question, f"questions.{question_id}")["type"]
    if threshold["type"] != question_type:
        raise PresetValidationError(f"{field_name}.type must match question type")
    threshold_fields = set(threshold) - {"type"}
    if threshold_fields != {"keep_at_least"} and threshold_fields != {"fail_at_least"}:
        raise PresetValidationError(
            f"{field_name} must have exactly one of keep_at_least or fail_at_least"
        )
    field = next(iter(threshold_fields))
    amount = threshold[field]
    if question_type == "choice":
        raise PresetValidationError(
            "choice thresholds use equality, not numeric values"
        )
    if question_type == "score":
        if field != "fail_at_least":
            raise PresetValidationError("score thresholds require fail_at_least")
        if (
            isinstance(amount, bool)
            or not isinstance(amount, int)
            or amount not in range(4)
        ):
            raise PresetValidationError(
                f"{field_name}.fail_at_least must be an integer from 0 to 3"
            )
        return
    if (
        isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not 0 <= amount <= 1
    ):
        raise PresetValidationError(
            f"{field_name}.{field} must be a number from 0 to 1"
        )


def _validate_calibration(value: Any) -> None:
    field_name = "calibration"
    calibration = _mapping(value, field_name)
    allowed = {
        "schema",
        "threshold_margin",
        "max_choice_flips",
        "max_probability_delta",
        "max_score_delta",
        "max_noul_delta",
        "max_threshold_crossings",
        "repeats",
    }
    _reject_unknown(calibration, allowed, field_name)
    _require_fields(calibration, {"schema"}, field_name)
    if calibration["schema"] != "jm.calibration/v1":
        raise PresetValidationError("calibration.schema must be 'jm.calibration/v1'")
    for name in (
        "max_choice_flips",
        "max_threshold_crossings",
        "repeats",
    ):
        if name in calibration:
            if name == "repeats":
                _positive_integer(calibration[name], f"{field_name}.{name}")
            else:
                _nonnegative_integer(calibration[name], f"{field_name}.{name}")
    for name in (
        "threshold_margin",
        "max_probability_delta",
        "max_score_delta",
        "max_noul_delta",
    ):
        if name in calibration:
            amount = calibration[name]
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(float(amount))
                or amount < 0
            ):
                raise PresetValidationError(
                    f"{field_name}.{name} must be a finite non-negative number"
                )
