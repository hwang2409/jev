from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

from ._transport import _resolve_gateway_key as resolve_gateway_key
from .answers import CoverageRecord, Diagnostic, ErrorDetail, ErrorRecord, RecordMeta
from .cache import CacheStore
from .calibrate import (
    CalibrationTolerances,
    run_calibration,
    tolerances_for_preset,
)
from .chunkers import chunk_file, chunk_input
from .client import make_judge
from .presets import (
    CHUNKER_SETTINGS,
    CHUNKING_COMMON_SETTINGS,
    SCHEMA_V2,
    SCHEMA_V3,
    Preset,
    PresetError,
    PresetNotFoundError,
    load_preset,
    resolve_chunker,
    resolve_prefilter,
    resolve_preset,
    validate_preset,
)
from .runner import (
    InputError,
    ResultFilter,
    State,
    StateLimits,
    StateRejection,
    _run_pipeline,
    emit,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


class _UsageError(ValueError):
    exit_code = 64


class _OperationalError(RuntimeError):
    exit_code = 2


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="jm",
        description="Apply typed Jev questions to finite input states.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="judge finite input with a preset")
    run.add_argument("--preset", required=True, help="preset name or path")
    _add_judgment_options(run)
    run.add_argument("paths", nargs="*", help="file paths when --by file is used")

    jgrep = commands.add_parser("jgrep", help="run the jgrep preset")
    jgrep.set_defaults(short_preset="jgrep")
    _add_judgment_options(jgrep, include_predicate=False)
    jgrep.add_argument("paths", nargs="*", help="file paths when --by file is used")

    jfilter = commands.add_parser("jfilter", help="run the jfilter preset")
    jfilter.add_argument("predicate", nargs="?", help="natural-language predicate")
    jfilter.add_argument(
        "--predicate", dest="predicate_option", action="append"
    )
    jfilter.set_defaults(short_preset="jfilter")
    _add_judgment_options(jfilter, include_query=False, include_predicate=False)
    jfilter.add_argument("paths", nargs="*", help="file paths when --by file is used")

    gate = commands.add_parser("gate", help="judge input and apply a policy")
    gate.add_argument("--preset", required=True, help="preset name or path")
    gate.add_argument("--policy", required=True, help="typed failure policy")
    gate.add_argument(
        "--require-states",
        type=_nonnegative_int,
        default=1,
        help="minimum judged states required for a passing gate",
    )
    _add_judgment_options(gate)
    gate.add_argument("paths", nargs="*", help="file paths when --by file is used")

    preset = commands.add_parser("preset", help="inspect or validate presets")
    preset_commands = preset.add_subparsers(dest="preset_command", required=True)
    preset_commands.add_parser("list", help="list available presets")
    for name in ("show", "validate"):
        command = preset_commands.add_parser(name, help=f"{name} a preset")
        command.add_argument("target", nargs="?", help="preset name or path")

    cache = commands.add_parser("cache", help="manage the local answer cache")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    for name in ("export", "clear"):
        command = cache_commands.add_parser(name, help=f"{name} cached answers")
        command.add_argument("--preset", required=True, help="preset name")

    calibrate = commands.add_parser(
        "calibrate", help="compare cached answers with the live model"
    )
    calibrate.add_argument("--preset", required=True, help="preset name or path")
    calibrate.add_argument("--cache-dir", type=Path)
    calibrate.add_argument("--threshold-margin", type=_nonnegative_float)
    calibrate.add_argument("--max-choice-flips", type=_nonnegative_int)
    calibrate.add_argument("--max-probability-delta", type=_nonnegative_float)
    calibrate.add_argument("--max-score-delta", type=_nonnegative_float)
    calibrate.add_argument("--max-noul-delta", type=_nonnegative_float)
    calibrate.add_argument("--repeats", type=_positive_int)
    calibrate.add_argument("--concurrency", type=_positive_int, default=4)
    calibrate.add_argument("--max-threshold-crossings", type=_nonnegative_int)
    calibrate.add_argument("--format", choices=("jsonl",), default="jsonl")

    return parser


def _add_judgment_options(
    parser: argparse.ArgumentParser,
    *,
    include_query: bool = True,
    include_predicate: bool = True,
) -> None:
    parser.add_argument("--input", type=Path, help="read finite input from PATH")
    parser.add_argument(
        "--by",
        choices=("line", "para", "hunk", "file", "record", "state"),
        help=(
            "form states by line, paragraph, hunk, file, record, or state; "
            "file mode skips oversized files"
        ),
    )
    parser.add_argument("--state-ref", default="id", help="record identity field")
    parser.add_argument(
        "--metadata-fields",
        help="comma-separated record fields to expose as context.metadata",
    )
    parser.add_argument("--max-chunks", type=_nonnegative_int)
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=4,
        help="maximum in-flight requests",
    )
    parser.add_argument("--format", choices=("jsonl", "pretty"))
    parser.add_argument("--filter", choices=("keep", "policy"))
    parser.add_argument("--filter-policy")
    parser.add_argument("--consistency", type=_consistency_count)
    parser.add_argument("--consistency-sigma", type=_nonnegative_float, default=2.0)
    parser.add_argument("--prefilter", choices=("bm25",))
    parser.add_argument("--prefilter-top", type=_positive_int)
    parser.add_argument("--prefilter-fields")
    parser.add_argument("--prefilter-query")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="supply a declared preset parameter as key=value",
    )
    if include_query:
        parser.add_argument("--query", action="append")
    if include_predicate:
        parser.add_argument("--predicate", action="append")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _consistency_count(value: str) -> int:
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def main(
    argv: list[str] | None = None,
    *,
    judge_fn: Callable[[State, Mapping[str, object], str], object] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    cache_store: CacheStore | None = None,
) -> int:
    parser = _parser()
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    try:
        args = parser.parse_args(argv)
        if args.command == "preset":
            return _preset_command(args, output)
        if args.command == "cache":
            return _cache_command(args, output, errors, cache_store)
        if args.command == "calibrate":
            return _calibration_command(
                args, output, errors, judge_fn, cache_store
            )
        return _judgment_command(
            args,
            stdin or sys.stdin,
            output,
            errors,
            judge_fn,
            cache_store,
        )
    except SystemExit:
        raise
    except (_UsageError, _OperationalError, PresetError, ValueError) as exc:
        errors.write(f"jm: error: {exc}\n")
        errors.flush()
        return getattr(exc, "exit_code", 64)
    except PresetNotFoundError as exc:
        errors.write(f"jm: error: {exc}\n")
        errors.flush()
        return 64
    except BrokenPipeError:
        return 0
    except OSError as exc:
        errors.write(f"jm: error: {exc}\n")
        errors.flush()
        return 2


def _calibration_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    judge_fn: Callable[[State, Mapping[str, object], str], object] | None,
    cache_store: CacheStore | None,
) -> int:
    preset = resolve_preset_or_path(args.preset)
    tolerances = tolerances_for_preset(preset)
    overrides = {
        "threshold_margin": args.threshold_margin,
        "max_choice_flips": args.max_choice_flips,
        "max_probability_delta": args.max_probability_delta,
        "max_score_delta": args.max_score_delta,
        "max_noul_delta": args.max_noul_delta,
        "max_threshold_crossings": args.max_threshold_crossings,
        "repeats": args.repeats,
    }
    values = tolerances.as_dict()
    values.update({key: value for key, value in overrides.items() if value is not None})
    resolved_tolerances = CalibrationTolerances(**values)
    store = CacheStore(args.cache_dir) if args.cache_dir is not None else cache_store
    active_store = store or CacheStore()
    close_judge = None
    active_judge = judge_fn
    if active_judge is None:
        active_judge, close_judge = make_judge()
    try:
        return run_calibration(
            preset,
            active_store,
            active_judge,
            stdout=stdout,
            stderr=stderr,
            tolerances=resolved_tolerances,
            concurrency=args.concurrency,
        )
    finally:
        if close_judge is not None:
            close_judge()


def _judgment_command(
    args: argparse.Namespace,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    judge_fn: Callable[[State, Mapping[str, object], str], object] | None,
    cache_store: CacheStore | None,
) -> int:
    preset_name = getattr(args, "short_preset", None) or args.preset
    preset = resolve_preset_or_path(preset_name)
    _validate_consistency_options(args, preset)
    by = resolve_chunker(preset, args.by)
    if args.metadata_fields is not None and by != "record":
        raise _UsageError("--metadata-fields requires --by record")
    paths = tuple(getattr(args, "paths", ()))
    inline_filter_policy = None
    if (
        args.filter == "policy"
        and args.filter_policy is None
        and len(paths) == 1
        and paths[0].startswith(("any(", "all("))
    ):
        inline_filter_policy = paths[0]
        args.paths = ()
        paths = ()
    if paths and by != "file":
        raise _UsageError("positional input paths require --by file")
    if paths and args.input is not None:
        raise _UsageError("positional input paths cannot be combined with --input")

    query = _single_alias(getattr(args, "query", None), "query")
    query_option = _single_alias(getattr(args, "query_option", None), "query")
    if query_option is not None:
        if query is not None:
            raise _UsageError("query was supplied more than once")
        query = query_option
    predicate = _single_alias(getattr(args, "predicate", None), "predicate")
    predicate_option = _single_alias(
        getattr(args, "predicate_option", None), "predicate"
    )
    if predicate_option is not None:
        if predicate is not None:
            raise _UsageError("predicate was supplied more than once")
        predicate = predicate_option

    parameters = _parse_parameters(
        preset,
        args.param,
        query=query,
        predicate=predicate,
    )

    effective_preset = _with_max_chunks(preset, args.max_chunks)
    prefilter = resolve_prefilter(
        effective_preset,
        command=args.command,
        ranker=args.prefilter,
        top=args.prefilter_top,
        fields=args.prefilter_fields,
        query=args.prefilter_query,
        parameters=parameters,
    )
    prefilter_parameters = set()
    if prefilter is not None:
        if prefilter["query_source"] == "context.query":
            prefilter_parameters.add("query")
        elif prefilter["query_source"] == "context.predicate":
            prefilter_parameters.add("predicate")
    _validate_preset_parameters(
        effective_preset,
        query,
        predicate,
        parameters=parameters,
        allowed_parameters=prefilter_parameters,
    )
    if judge_fn is None and not resolve_gateway_key():
        raise _OperationalError(
            "Vercel AI Gateway API key is not set; set it before running a "
            "judgment command"
        )
    limits = StateLimits(**effective_preset.chunking["limits"])
    states, rejections = _form_states(
        args,
        stdin,
        by,
        limits,
        effective_preset.chunking,
        query=query,
        predicate=predicate,
        parameters=parameters,
    )

    effective_preset = _with_chunker(effective_preset, by)
    result_filter = _result_filter(
        args.filter,
        args.filter_policy or inline_filter_policy,
        effective_preset,
    )
    formation_diagnostics: tuple[Diagnostic, ...] = ()
    if not states and not rejections:
        message = "no hunks found" if by == "hunk" else "no states found"
        formation_diagnostics = (Diagnostic("info", "empty_input", message),)
    try:
        outcome = _run_pipeline(
            effective_preset,
            states,
            rejections=rejections,
            max_chunks=effective_preset.chunking.get("max_chunks"),
            prefilter=prefilter,
            include_prefilter_warning=args.command == "jgrep",
            cache_store=cache_store or CacheStore(),
            concurrency=args.concurrency,
            judge_fn=judge_fn,
            consistency=args.consistency,
            consistency_sigma=args.consistency_sigma,
            output_format=args.format
            or effective_preset.data["output"]["default_format"],
            jsonl_stream=stdout,
            pretty_stream=stderr,
            result_filter=result_filter,
            pretty_template=effective_preset.data["output"]["pretty_template"],
            policy=args.policy if args.command == "gate" else None,
            require_states=getattr(args, "require_states", 1),
            formation_diagnostics=formation_diagnostics,
            empty_input_error=False,
        )
    except InputError as exc:
        return _emit_input_error(
            effective_preset,
            by,
            str(exc),
            source_ref=getattr(exc, "source_ref", None),
            output_format=args.format
            or effective_preset.data["output"]["default_format"],
            stdout=stdout,
            stderr=stderr,
        )
    if outcome.emitted.broken_pipe:
        return 0
    if outcome.gate_result is not None:
        return outcome.gate_result.exit_code
    if (
        outcome.emitted.coverage is not None
        and outcome.emitted.coverage.coverage == "partial"
    ):
        return 2
    return 0


def _emit_input_error(
    preset: Preset,
    chunker: str,
    message: str,
    *,
    source_ref: str | None = None,
    output_format: str,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    meta = RecordMeta(
        preset.name,
        preset.version,
        preset.model,
        chunker,
        "not_applicable",
        preset_schema=(
            preset.schema if preset.schema in {SCHEMA_V2, SCHEMA_V3} else None
        ),
    )
    records = (
        ErrorRecord(
            None,
            ErrorDetail("input_error", message),
            meta,
            source_ref=source_ref or "stdin:byte=0,line=1",
        ),
        CoverageRecord(
            coverage="partial",
            coverage_counts={
                "discovered": 0,
                "judged": 0,
                "emitted": 0,
                "skipped": 0,
                "failed": 0,
            },
            coverage_reasons=("input_error",),
            meta=meta,
        ),
    )
    emitted = emit(
        records,
        format=output_format,  # type: ignore[arg-type]
        jsonl_stream=stdout,
        pretty_stream=stderr,
    )
    return 0 if emitted.broken_pipe else 2


def _validate_consistency_options(args: argparse.Namespace, preset: Preset) -> None:
    if args.consistency is None:
        if args.consistency_sigma != 2.0:
            raise _UsageError("--consistency-sigma requires --consistency")
        return
    if not any(
        question.get("type") == "noul"
        for question in preset.questions.values()
        if isinstance(question, Mapping)
    ):
        raise _UsageError("--consistency requires at least one Noul question")


def resolve_preset_or_path(identifier: str) -> Preset:
    path = Path(identifier).expanduser()
    if path.exists() or "/" in identifier:
        return load_preset(path)
    return resolve_preset(identifier)


def _with_max_chunks(preset: Preset, max_chunks: int | None) -> Preset:
    if max_chunks is None:
        return preset
    data = copy.deepcopy(dict(preset.data))
    chunking = dict(data["chunking"])
    chunking["max_chunks"] = max_chunks
    data["chunking"] = chunking
    return Preset(validate_preset(data), preset.path)


def _with_chunker(preset: Preset, chunker: str) -> Preset:
    if chunker == preset.default_chunker:
        return preset
    data = copy.deepcopy(dict(preset.data))
    chunking = dict(data["chunking"])
    chunking["by"] = chunker
    supported = CHUNKING_COMMON_SETTINGS | CHUNKER_SETTINGS[chunker]
    for setting in set(chunking) - supported:
        del chunking[setting]
    data["chunking"] = chunking
    return Preset(validate_preset(data), preset.path)


def _form_states(
    args: argparse.Namespace,
    stdin: TextIO,
    by: str,
    limits: StateLimits,
    chunking: Mapping[str, object],
    *,
    query: str | None,
    predicate: str | None,
    parameters: Mapping[str, str] | None = None,
) -> tuple[tuple[State, ...], tuple[StateRejection, ...]]:
    rejections: list[StateRejection] = []
    paths = tuple(getattr(args, "paths", ()))
    if paths:
        states: list[State] = []
        for path in paths:
            states.extend(
                chunk_file(
                    path,
                    limits=limits,
                    query=query,
                    predicate=predicate,
                    parameters=parameters,
                    _rejections=rejections,
                )
            )
        return tuple(states), tuple(rejections)

    if args.input is not None:
        value: str | bytes = args.input.read_bytes()
    else:
        value = _read_stdin_bytes(stdin)
    chunk_kwargs: dict[str, object] = {
        "limits": limits,
    }
    if by == "line":
        chunk_kwargs["adjacent_lines"] = chunking.get("context_lines", 1)
    elif by == "para":
        chunk_kwargs["adjacent_paragraphs"] = chunking.get("context_paragraphs", 1)
    if by == "record":
        chunk_kwargs["state_ref_field"] = args.state_ref
        chunk_kwargs["metadata_fields"] = _parse_metadata_fields(
            args.metadata_fields
        )
    if by in {"line", "para", "file", "record"}:
        chunk_kwargs["query"] = query
        chunk_kwargs["predicate"] = predicate
    chunk_kwargs["parameters"] = parameters or {}
    result = chunk_input(by, value, **chunk_kwargs)
    return result.formed, result.rejections


def _parse_metadata_fields(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    fields = tuple(item.strip() for item in value.split(","))
    if not fields or any(not field for field in fields):
        raise _UsageError("--metadata-fields must not contain empty fields")
    if len(set(fields)) != len(fields):
        raise _UsageError("--metadata-fields must not contain duplicates")
    return fields


def _read_stdin_bytes(stdin: TextIO) -> str | bytes:
    binary = getattr(stdin, "buffer", None)
    return binary.read() if binary is not None else stdin.read()


def _validate_preset_parameters(
    preset: Preset,
    query: str | None,
    predicate: str | None,
    *,
    parameters: Mapping[str, str] | None = None,
    allowed_parameters: set[str] | None = None,
) -> None:
    if preset.schema == SCHEMA_V3:
        declared = set(preset.declared_parameters)
        values = dict(parameters or {})
        unknown = sorted(set(values) - declared)
        if unknown:
            raise _UsageError(
                f"unknown parameter '{unknown[0]}' for preset '{preset.name}'"
            )
        for question in preset.questions.values():
            for field in question["instructions"]["state_fields"]:
                if not field.startswith("context."):
                    continue
                parameter = field.removeprefix("context.")
                if parameter in declared and parameter not in values:
                    raise _UsageError(
                        f"missing required parameter '{parameter}' for preset "
                        f"'{preset.name}'"
                    )
        return
    allowed = allowed_parameters or set()
    required: set[str] = set()
    for question in preset.questions.values():
        for field in question["instructions"]["state_fields"]:
            if field.startswith("context."):
                parameter = field.removeprefix("context.")
                if parameter in {"query", "predicate"}:
                    required.add(parameter)

    values = {"query": query, "predicate": predicate}
    for parameter, value in values.items():
        if (
            value is not None
            and parameter not in required
            and parameter not in allowed
        ):
            raise _UsageError(
                f"unknown parameter '{parameter}' for preset '{preset.name}'"
            )
    for parameter in sorted(required):
        if values[parameter] is None:
            raise _UsageError(
                f"missing required parameter '{parameter}' for preset '{preset.name}'"
            )


def _parse_parameters(
    preset: Preset,
    pairs: list[str],
    *,
    query: str | None,
    predicate: str | None,
) -> dict[str, str]:
    if preset.schema != SCHEMA_V3 and pairs:
        raise _UsageError("--param requires a jm.preset/v3 preset")
    values: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise _UsageError("--param values must use key=value")
        key, value = pair.split("=", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise _UsageError(f"invalid parameter name '{key}'")
        if key in values:
            raise _UsageError(f"parameter '{key}' was supplied more than once")
        values[key] = value
    aliases = {"query": query, "predicate": predicate}
    for key, value in aliases.items():
        if value is None:
            continue
        if preset.schema == SCHEMA_V3 and key in values:
            raise _UsageError(
                f"parameter '{key}' conflicts with its --{key} alias"
            )
        values[key] = value
    return values


def _single_alias(
    values: str | list[str] | None, name: str
) -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        return values
    if len(values) > 1:
        raise _UsageError(f"{name} was supplied more than once")
    return values[0]


def _result_filter(
    requested: str | None,
    policy_expression: str | None,
    preset: Preset,
) -> ResultFilter | None:
    if policy_expression is not None and requested != "policy":
        raise _UsageError("--filter-policy requires --filter policy")
    if requested == "policy":
        if not policy_expression:
            raise _UsageError("--filter policy requires --filter-policy")
        return ResultFilter(kind="policy", expression=policy_expression)
    if requested is None:
        return None
    thresholds = preset.data["thresholds"]
    if len(thresholds) != 1:
        raise _UsageError("--filter=keep requires a single keep threshold")
    question_id, threshold = next(iter(thresholds.items()))
    if "keep_at_least" not in threshold:
        raise _UsageError("--filter=keep requires a keep_at_least threshold")
    minimum = threshold["keep_at_least"]

    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool):
        raise _UsageError("--filter=keep requires a numeric threshold")
    return ResultFilter(
        kind="keep",
        question_id=question_id,
        operator=">=",
        threshold=minimum,
    )


def _preset_command(args: argparse.Namespace, stdout: TextIO) -> int:
    if args.preset_command == "list":
        if getattr(args, "target", None) is not None:
            raise _UsageError("preset list does not accept a target")
        for preset in _available_presets():
            _write_metadata(preset, stdout)
        return 0

    target = args.target or "jgrep"
    preset = resolve_preset_or_path(target)
    if args.preset_command == "show":
        payload = dict(preset.data)
        payload["path"] = str(preset.path)
        payload["effective_chunker"] = preset.default_chunker
        _write_json(payload, stdout)
        return 0

    _write_metadata(preset, stdout)
    return 0


def _available_presets() -> tuple[Preset, ...]:
    locations = (
        Path.cwd(),
        Path(__file__).with_name("presets"),
        Path(os.environ.get("JM_PRESETS", "~/.config/jm/presets")).expanduser(),
    )
    found: dict[str, Preset] = {}
    for directory in locations:
        if not directory.is_dir():
            continue
        for path in sorted((*directory.glob("*.yml"), *directory.glob("*.yaml"))):
            try:
                preset = load_preset(path)
            except PresetError:
                continue
            found.setdefault(preset.name, preset)
    return tuple(found[name] for name in sorted(found))


def _write_metadata(preset: Preset, stdout: TextIO) -> None:
    _write_json(
        {
            "name": preset.name,
            "version": preset.version,
            "model": preset.model,
            "path": str(preset.path),
            "effective_chunker": preset.default_chunker,
        },
        stdout,
    )


def _write_json(payload: Mapping[str, object], stdout: TextIO) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    stdout.flush()


def _cache_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    cache_store: CacheStore | None,
) -> int:
    del stderr
    store = cache_store or CacheStore()
    if args.cache_command == "export":
        store.export_jsonl(args.preset, stdout)
        return 0
    removed = store.clear(args.preset)
    _write_json({"preset": args.preset, "removed": removed}, stdout)
    return 0
