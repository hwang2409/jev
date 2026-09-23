from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

from ._transport import _resolve_gateway_key as resolve_gateway_key
from .answers import NoulAnswer, ResultRecord, ScoreAnswer
from .cache import CacheStore
from .calibrate import (
    CalibrationTolerances,
    run_calibration,
    tolerances_for_preset,
)
from .chunkers import chunk_file, chunk_input
from .client import JevClient
from .presets import (
    Preset,
    PresetError,
    load_preset,
    resolve_chunker,
    resolve_preset,
    validate_preset,
)
from .runner import Runner, State, StateLimits, StateRejection


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
    jfilter.add_argument("--predicate", dest="predicate_option")
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
        "--by", choices=("line", "para", "hunk", "file", "record")
    )
    parser.add_argument("--state-ref", default="id", help="record identity field")
    parser.add_argument("--max-chunks", type=_nonnegative_int)
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=4,
        help="maximum in-flight requests",
    )
    parser.add_argument("--format", choices=("jsonl", "pretty"))
    parser.add_argument("--filter", choices=("keep",))
    if include_query:
        parser.add_argument("--query")
    if include_predicate:
        parser.add_argument("--predicate")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
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
    client = None
    active_judge = judge_fn
    if active_judge is None:
        client = JevClient()
        active_judge = client
    try:
        return run_calibration(
            preset,
            active_store,
            active_judge,
            stdout=stdout,
            stderr=stderr,
            tolerances=resolved_tolerances,
        )
    finally:
        if client is not None:
            client.close()


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
    by = resolve_chunker(preset, args.by)
    paths = tuple(getattr(args, "paths", ()))
    if paths and by != "file":
        raise _UsageError("positional input paths require --by file")
    if paths and args.input is not None:
        raise _UsageError("positional input paths cannot be combined with --input")

    query = getattr(args, "query", None)
    query_option = getattr(args, "query_option", None)
    if query_option is not None:
        if query is not None:
            raise _UsageError("query was supplied more than once")
        query = query_option
    predicate = getattr(args, "predicate", None)
    predicate_option = getattr(args, "predicate_option", None)
    if predicate_option is not None:
        if predicate is not None:
            raise _UsageError("predicate was supplied more than once")
        predicate = predicate_option

    effective_preset = _with_max_chunks(preset, args.max_chunks)
    _validate_preset_parameters(effective_preset, query, predicate)
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
    )

    if judge_fn is None:
        client = JevClient()
        active_judge = client
    else:
        client = None
        active_judge = judge_fn

    try:
        runner = Runner(active_judge)
        result_filter = _result_filter(args.filter, effective_preset)
        run_kwargs: dict[str, object] = {
            "preset": effective_preset,
            "chunker": by,
            "stdout": stdout,
            "stderr": stderr,
            "output_format": args.format
            or effective_preset.data["output"]["default_format"],
            "result_filter": result_filter,
            "rejections": rejections,
            "cache_store": cache_store or CacheStore(),
            "concurrency": args.concurrency,
        }
        if args.command == "gate":
            run_kwargs["policy"] = args.policy
            run_kwargs["require_states"] = args.require_states
        result = runner.run(states, **run_kwargs)
        return result.exit_code
    finally:
        if client is not None:
            client.close()


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


def _form_states(
    args: argparse.Namespace,
    stdin: TextIO,
    by: str,
    limits: StateLimits,
    chunking: Mapping[str, object],
    *,
    query: str | None,
    predicate: str | None,
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
    if by in {"line", "para", "file", "record"}:
        chunk_kwargs["query"] = query
        chunk_kwargs["predicate"] = predicate
    result = chunk_input(by, value, **chunk_kwargs)
    return result.formed, result.rejections


def _read_stdin_bytes(stdin: TextIO) -> str | bytes:
    binary = getattr(stdin, "buffer", None)
    return binary.read() if binary is not None else stdin.read()


def _validate_preset_parameters(
    preset: Preset, query: str | None, predicate: str | None
) -> None:
    required: set[str] = set()
    for question in preset.questions.values():
        for field in question["instructions"]["state_fields"]:
            if field.startswith("context."):
                parameter = field.removeprefix("context.")
                if parameter in {"query", "predicate"}:
                    required.add(parameter)

    values = {"query": query, "predicate": predicate}
    for parameter, value in values.items():
        if value is not None and parameter not in required:
            raise _UsageError(
                f"unknown parameter '{parameter}' for preset '{preset.name}'"
            )
    for parameter in sorted(required):
        if values[parameter] is None:
            raise _UsageError(
                f"missing required parameter '{parameter}' for preset '{preset.name}'"
            )


def _result_filter(
    requested: str | None, preset: Preset
) -> Callable[[ResultRecord], bool] | None:
    if requested is None:
        return None
    thresholds = preset.data["thresholds"]
    if len(thresholds) != 1:
        raise _UsageError("--filter=keep requires a single keep threshold")
    question_id, threshold = next(iter(thresholds.items()))
    if "keep_at_least" not in threshold:
        raise _UsageError("--filter=keep requires a keep_at_least threshold")
    minimum = threshold["keep_at_least"]

    def keep(record: ResultRecord) -> bool:
        answer = record.answers[question_id]
        if isinstance(answer, NoulAnswer):
            value = answer.noul
        elif isinstance(answer, ScoreAnswer):
            value = answer.score
        else:
            raise _UsageError("--filter=keep does not support choice answers")
        return value >= minimum

    return keep


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
