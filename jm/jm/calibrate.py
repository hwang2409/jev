from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any, TextIO

from .answers import (
    Answer,
    ChoiceAnswer,
    ErrorResponse,
    JudgeResponse,
    NoulAnswer,
    ScoreAnswer,
    answer_to_dict,
    probability_keys_for_question,
    score_argmax,
)
from .cache import CacheEntry, CacheStore, battery_hash, canonical_json_bytes
from .presets import Preset
from .runner import State, _repeat_state

CALIBRATION_SCHEMA = "jm.calibration/v3"


@dataclass(frozen=True, slots=True)
class CalibrationTolerances:
    threshold_margin: float = 0.05
    max_choice_flips: int = 0
    max_probability_delta: float = 0.05
    max_score_delta: float = 0.50
    max_noul_delta: float = 0.05
    max_threshold_crossings: int = 0
    repeats: int = 1

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold_margin": self.threshold_margin,
            "max_choice_flips": self.max_choice_flips,
            "max_probability_delta": self.max_probability_delta,
            "max_score_delta": self.max_score_delta,
            "max_noul_delta": self.max_noul_delta,
            "max_threshold_crossings": self.max_threshold_crossings,
            "repeats": self.repeats,
        }


class CalibrationOperationalError(ValueError):
    exit_code = 2


Judge = Callable[[State, Mapping[str, Any], str], object]


def run_calibration(
    preset: Preset,
    cache_store: CacheStore,
    judge_fn: Judge,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    tolerances: CalibrationTolerances | None = None,
    concurrency: int = 4,
) -> int:
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    resolved = tolerances or tolerances_for_preset(preset)
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency <= 0
    ):
        raise CalibrationOperationalError("concurrency must be a positive integer")
    effective_concurrency = min(concurrency, 8)
    try:
        entries = _load_entries(preset, cache_store)
        states = tuple(_state_for_entry(entry, preset) for entry in entries)
    except CalibrationOperationalError as exc:
        summary = _summary(
            preset,
            resolved,
            0,
            Counter(),
            Counter(),
            (),
            (),
            0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            0,
            str(exc),
        )
        _write_json(summary, output)
        errors.write(f"jm calibrate: {exc}\n")
        errors.write(f"{_provenance(summary)}\n")
        output.flush()
        errors.flush()
        return 2

    model_baseline: Counter[str] = Counter()
    model_candidate: Counter[str] = Counter()
    candidate_usages: list[Mapping[str, Any]] = []
    baseline_usages: list[Mapping[str, Any]] = []
    stable_drift = 0
    boundary_noise_cases = 0
    far_side_noise_cases = 0
    choice_flips = 0
    threshold_crossings = 0
    max_probability_delta = 0.0
    max_score_delta = 0.0
    max_noul_delta = 0.0
    operational_error: str | None = None
    completed_cases = 0
    comparison_count = 0
    candidate_jobs = {}
    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        for case_index, state in enumerate(states):
            for repeat_index in range(resolved.repeats):
                candidate_jobs[(case_index, repeat_index)] = executor.submit(
                    _candidate_call,
                    judge_fn,
                    _repeat_state(state),
                    preset,
                )

        for case_index, (entry, _state) in enumerate(zip(entries, states)):
            candidates: list[JudgeResponse] = []
            for repeat_index in range(resolved.repeats):
                response, error = candidate_jobs[(case_index, repeat_index)].result()
                if error is not None:
                    operational_error = error
                    break
                assert response is not None
                candidates.append(response)
            if operational_error is not None:
                break

            comparisons: list[dict[str, Any]] = []
            case_metrics: list[dict[str, float | bool]] = []
            target_group = _target_provenance(entry, preset)
            for question_id, question in preset.questions.items():
                try:
                    baseline_answer = entry.response.answers[question_id]
                    candidate_answers = [
                        response.answers[question_id] for response in candidates
                    ]
                    comparison, metrics = _comparison_record(
                        entry,
                        question_id,
                        question,
                        preset.data["thresholds"].get(question_id),
                        baseline_answer,
                        candidates,
                        candidate_answers,
                        resolved,
                        target_group,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    operational_error = f"malformed calibration answer: {exc}"
                    break
                comparisons.append(comparison)
                case_metrics.append(metrics)
            if operational_error is not None:
                break

            completed_cases += 1
            case_boundary_noise = any(
                bool(metrics["boundary_noise"]) for metrics in case_metrics
            )
            case_far_side_noise = any(
                bool(metrics["far_side_noise"]) for metrics in case_metrics
            )
            boundary_noise_cases += int(case_boundary_noise)
            far_side_noise_cases += int(case_far_side_noise)
            case_choice_flip = any(
                bool(metrics["choice_flip"]) for metrics in case_metrics
            )
            case_threshold_crossing = any(
                bool(metrics["threshold_crossing"]) for metrics in case_metrics
            )
            baseline_model = entry.served_model
            candidate_models = [_served_model(response) for response in candidates]
            if isinstance(entry.usage, Mapping):
                baseline_usages.append(entry.usage)
            candidate_usages.extend(
                response.usage for response in candidates if response.usage is not None
            )
            model_baseline.update([baseline_model])
            model_candidate.update(candidate_models)
            stable_drift += int(
                any(metrics["stable_drift"] for metrics in case_metrics)
            )
            choice_flips += int(case_choice_flip)
            threshold_crossings += int(case_threshold_crossing)
            comparison_count += len(comparisons)
            for metrics in case_metrics:
                max_probability_delta = max(
                    max_probability_delta, float(metrics["probability_delta"])
                )
                max_score_delta = max(max_score_delta, float(metrics["score_delta"]))
                max_noul_delta = max(max_noul_delta, float(metrics["noul_delta"]))

            record = {
                "record_type": "calibration_comparison",
                "calibration_version": CALIBRATION_SCHEMA,
                "calibration_case_id": _calibration_case_id(entry, target_group),
                "cache_entry": {
                    "cache_key": entry.cache_key,
                    "provenance": [dict(target_group)],
                },
                "comparisons": comparisons,
                "provenance": {
                    "baseline_model": baseline_model,
                    "candidate_models": candidate_models,
                },
                "state_ref": target_group["state_refs"][0],
                "state_refs": list(target_group["state_refs"]),
                "boundary_noise": case_boundary_noise,
                "far_side_noise": case_far_side_noise,
                "within_tolerance": None if case_boundary_noise else True,
            }
            _write_json(record, output)

    summary = _summary(
        preset,
        resolved,
        completed_cases,
        model_baseline,
        model_candidate,
        baseline_usages,
        candidate_usages,
        choice_flips,
        max_probability_delta,
        max_score_delta,
        max_noul_delta,
        threshold_crossings,
        stable_drift,
        boundary_noise_cases,
        far_side_noise_cases,
        comparison_count,
        operational_error,
    )
    _write_json(summary, output)
    provenance = _provenance(summary)
    if operational_error:
        errors.write(f"jm calibrate: {operational_error}\n")
    elif completed_cases == 0:
        errors.write("jm calibrate: no qualifying cases\n")
    elif summary["within_tolerance"] is True:
        errors.write("jm calibrate: within tolerance\n")
    elif summary["within_tolerance"] is False:
        errors.write("jm calibrate: drift exceeded tolerance\n")
    else:
        errors.write("jm calibrate: calibration decision is indeterminate\n")
    errors.write(f"{provenance}\n")
    output.flush()
    errors.flush()

    if operational_error or completed_cases == 0:
        return 2
    if summary["within_tolerance"] is True:
        return 0
    if summary["within_tolerance"] is False:
        return 1
    return 2


calibrate = run_calibration


def _candidate_call(
    judge_fn: Judge, state: State, preset: Preset
) -> tuple[JudgeResponse | None, str | None]:
    try:
        response = judge_fn(state, preset.questions, preset.model)
    except Exception as exc:
        return None, f"live calibration call failed: {exc}"
    if isinstance(response, ErrorResponse):
        return None, f"live calibration call failed: {response.error}"
    if not isinstance(response, JudgeResponse):
        return None, "live calibration returned an invalid response"
    if not response.complete or set(response.answers) != set(preset.questions):
        return None, "live calibration returned an incomplete response"
    return response, None


def tolerances_for_preset(preset: Preset) -> CalibrationTolerances:
    raw = preset.data.get("calibration", {})
    if not isinstance(raw, Mapping):
        return CalibrationTolerances()
    if raw.get("schema", CALIBRATION_SCHEMA) not in {
        "jm.calibration/v1",
        CALIBRATION_SCHEMA,
    }:
        raise ValueError(
            "calibration.schema must be 'jm.calibration/v1' or "
            f"{CALIBRATION_SCHEMA!r}"
        )
    defaults = CalibrationTolerances().as_dict()
    values = {name: raw.get(name, default) for name, default in defaults.items()}
    return CalibrationTolerances(**values)


def _load_entries(preset: Preset, store: CacheStore) -> tuple[CacheEntry, ...]:
    try:
        entries = store.calibration_entries(
            preset.name,
            preset.version,
            battery_hash(preset.questions),
        )
    except ValueError as exc:
        raise CalibrationOperationalError(str(exc)) from exc
    for entry in entries:
        target = entry.provenance_for(
            preset.name,
            preset.version,
            battery_hash(preset.questions),
        )
        if target is None:
            raise CalibrationOperationalError(
                f"cache entry {entry.cache_key} has no target provenance"
            )
        if entry.configured_model != preset.model:
            raise CalibrationOperationalError(
                f"cache entry {entry.cache_key} has model "
                f"{entry.configured_model!r}, "
                f"expected {preset.model!r}"
            )
        if set(entry.response.answers) != set(preset.questions):
            raise CalibrationOperationalError(
                f"cache entry {entry.cache_key} is incomplete"
            )
    return entries


def _target_provenance(entry: CacheEntry, preset: Preset) -> Mapping[str, Any]:
    target = entry.provenance_for(
        preset.name,
        preset.version,
        battery_hash(preset.questions),
    )
    if target is None:
        raise CalibrationOperationalError(
            f"cache entry {entry.cache_key} has no target provenance"
        )
    return target


def _state_for_entry(entry: CacheEntry, preset: Preset) -> State:
    raw_state = entry.wire_state
    if not isinstance(raw_state, Mapping):
        raise CalibrationOperationalError(
            f"cache entry {entry.cache_key} has an invalid state"
        )
    focus = raw_state.get("focus")
    context = raw_state.get("context")
    if not isinstance(focus, str) or not isinstance(context, Mapping):
        raise CalibrationOperationalError(
            f"cache entry {entry.cache_key} has an invalid state"
        )
    target = _target_provenance(entry, preset)
    state_ref = context.get("state_ref") or target["state_refs"][0]
    if not isinstance(state_ref, str) or not state_ref:
        raise CalibrationOperationalError(
            f"cache entry {entry.cache_key} has no state reference"
        )
    return State(state_ref, focus, context, wire_context_keys=frozenset(context))


def _comparison_record(
    entry: CacheEntry,
    question_id: str,
    question: Mapping[str, Any],
    threshold: Mapping[str, Any] | None,
    baseline: Answer,
    responses: Sequence[JudgeResponse],
    candidates: Sequence[Answer],
    tolerances: CalibrationTolerances,
    target_group: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    if len(candidates) != tolerances.repeats:
        raise ValueError("candidate repeat count does not match repeats")
    expected_type = question.get("type")
    if getattr(baseline, "type", None) != expected_type or any(
        getattr(candidate, "type", None) != expected_type for candidate in candidates
    ):
        raise ValueError(f"question {question_id} has an invalid answer type")

    baseline_model = _served_model(entry.response)
    candidate_models = [_served_model(response) for response in responses]
    candidate_repeats = [
        {
            "repeat": index,
            "answer": answer_to_dict(answer),
            "usage": response.usage,
            "served_model": model,
        }
        for index, (answer, response, model) in enumerate(
            zip(candidates, responses, candidate_models), start=1
        )
    ]
    choice_flip = _choice_flip(baseline, candidates)
    probability_delta = _probability_delta(baseline, candidates)
    score_delta = _numeric_delta(baseline, candidates, ScoreAnswer)
    noul_delta = _numeric_delta(baseline, candidates, NoulAnswer)
    thresholds = _thresholds(
        threshold,
        baseline,
        candidates,
        tolerances.threshold_margin,
    )
    threshold_crossing = any(item["crossing"] for item in thresholds)
    field_tolerance = {
        "score": tolerances.max_score_delta,
        "noul": tolerances.max_noul_delta,
    }.get(str(expected_type), tolerances.max_probability_delta)
    stable, boundary_noise, far_side_noise = _repeat_classification(
        expected_type,
        baseline,
        candidates,
        field_tolerance,
        thresholds,
    )
    candidate_usage = _sum_usage(response.usage for response in responses)
    baseline_gate_value = _gate_value(baseline)
    candidate_gate_values = [_gate_value(candidate) for candidate in candidates]
    candidate_confidence = [
        _derived_confidence(candidate, question) for candidate in candidates
    ]
    record = {
        "question_id": question_id,
        "primitive": expected_type,
        "baseline_answer": answer_to_dict(baseline),
        "baseline_usage": entry.usage,
        "baseline_served_model": baseline_model,
        "candidate_repeats": candidate_repeats,
        "candidate_usage": candidate_usage,
        "choice_flip": choice_flip,
        "probability_delta": probability_delta,
        "score_delta": score_delta,
        "noul_delta": noul_delta,
        "gate_values": {
            "baseline": baseline_gate_value,
            "candidate": _one_or_many(candidate_gate_values),
        },
        "derived_confidence": {
            "baseline": _derived_confidence(baseline, question),
            "candidate": _one_or_many(candidate_confidence),
        },
        "thresholds": thresholds,
        "stable_drift": stable,
        "boundary_noise": boundary_noise,
        "far_side_noise": far_side_noise,
    }
    if isinstance(baseline, ScoreAnswer):
        candidate_argmax = [
            score_argmax(candidate)
            for candidate in candidates
            if isinstance(candidate, ScoreAnswer)
        ]
        record.update(
            {
                "baseline_argmax": score_argmax(baseline),
                "candidate_argmax": _one_or_many(candidate_argmax),
                "argmax_crossing": any(
                    value != score_argmax(baseline) for value in candidate_argmax
                ),
            }
        )
    metrics: dict[str, float | bool] = {
        "choice_flip": choice_flip,
        "probability_delta": probability_delta,
        "score_delta": score_delta,
        "noul_delta": noul_delta,
        "threshold_crossing": threshold_crossing,
        "stable_drift": stable,
        "boundary_noise": boundary_noise,
        "far_side_noise": far_side_noise,
    }
    return record, metrics


def _calibration_case_id(
    entry: CacheEntry, target_group: Mapping[str, Any]
) -> str:
    value = {
        "cache_key": entry.cache_key,
        "wire_state": entry.wire_state,
        "target_provenance": target_group,
    }
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _choice_flip(baseline: Answer, candidates: Sequence[Answer]) -> bool:
    if not isinstance(baseline, ChoiceAnswer):
        return False
    return any(
        isinstance(candidate, ChoiceAnswer) and candidate.choice != baseline.choice
        for candidate in candidates
    )


def _probability_delta(baseline: Answer, candidates: Sequence[Answer]) -> float:
    if not isinstance(baseline, (ChoiceAnswer, ScoreAnswer)):
        return 0.0
    maps = [baseline.probabilities]
    maps.extend(
        candidate.probabilities
        for candidate in candidates
        if isinstance(candidate, (ChoiceAnswer, ScoreAnswer))
    )
    keys = set().union(*(probabilities for probabilities in maps))
    return round(
        max(
            (
                abs(probabilities.get(key, 0.0) - baseline.probabilities.get(key, 0.0))
                for probabilities in maps[1:]
                for key in keys
            ),
            default=0.0,
        ),
        12,
    )


def _numeric_delta(
    baseline: Answer,
    candidates: Sequence[Answer],
    answer_type: type[ScoreAnswer] | type[NoulAnswer],
) -> float:
    if not isinstance(baseline, answer_type):
        return 0.0
    field = "score" if answer_type is ScoreAnswer else "noul"
    baseline_value = float(getattr(baseline, field))
    return round(
        max(
            (
                abs(float(getattr(candidate, field)) - baseline_value)
                for candidate in candidates
                if isinstance(candidate, answer_type)
            ),
            default=0.0,
        ),
        12,
    )


def _thresholds(
    threshold: Mapping[str, Any] | None,
    baseline: Answer,
    candidates: Sequence[Answer],
    margin: float,
) -> list[dict[str, Any]]:
    if threshold is None:
        return []
    is_score = isinstance(baseline, ScoreAnswer)
    field = "argmax" if is_score else "noul"
    baseline_value = float(_gate_value(baseline))
    candidate_values = [float(_gate_value(candidate)) for candidate in candidates]
    target_name = "keep_at_least" if "keep_at_least" in threshold else "fail_at_least"
    target = float(threshold[target_name])
    values = [baseline_value, *candidate_values]
    distances = [abs(value - target) for value in values]
    sides = [value >= target for value in values]
    return [
        {
            "field": field,
            "operator": ">=",
            "target": threshold[target_name],
            "baseline_value": _number_value(baseline_value),
            "candidate_value": _one_or_many(
                [_number_value(value) for value in candidate_values]
            ),
            "distance": round(min(distances), 12),
            "near_threshold": any(distance <= margin for distance in distances),
            "crossing": any(side != sides[0] for side in sides[1:]),
        }
    ]


def _repeat_classification(
    primitive: Any,
    baseline: Answer,
    candidates: Sequence[Answer],
    tolerance: float,
    thresholds: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, bool]:
    if len(candidates) <= 1:
        changed = bool(candidates) and _gate_value(
            candidates[0]
        ) != _gate_value(baseline)
        boundary = changed and any(item["near_threshold"] for item in thresholds)
        return False, boundary, changed and not boundary
    if isinstance(baseline, ChoiceAnswer):
        choices = [
            candidate.choice
            for candidate in candidates
            if isinstance(candidate, ChoiceAnswer)
        ]
        if all(choice == baseline.choice for choice in choices):
            return False, False, False
        if len(set(choices)) == 1:
            return True, False, False
        return False, False, True

    numeric_field = "score" if isinstance(baseline, ScoreAnswer) else "noul"
    values = [float(_gate_value(candidate)) for candidate in candidates]
    base = float(_gate_value(baseline))
    diagnostic_values = [
        float(getattr(candidate, numeric_field)) for candidate in candidates
    ]
    diagnostic_base = float(getattr(baseline, numeric_field))
    deltas = [value - diagnostic_base for value in diagnostic_values]
    directions = {delta > 0 for delta in deltas if delta != 0}
    same_sides = all(
        len({value >= float(item["target"]) for value in values}) == 1
        for item in thresholds
    )
    coherent = (
        len(directions) <= 1
        and max(diagnostic_values) - min(diagnostic_values) <= tolerance
        and same_sides
    )
    changed = any(value != base for value in values)
    disagreement = len(set(values)) > 1
    near = any(item["near_threshold"] for item in thresholds)
    boundary = changed and (near or not same_sides)
    far_side = changed and not boundary
    if not coherent:
        return False, boundary, far_side or disagreement and not boundary
    if all(abs(delta) > tolerance for delta in deltas):
        return True, boundary, far_side
    if all(abs(delta) <= tolerance for delta in deltas):
        return False, boundary, far_side
    return False, boundary, far_side or disagreement and not boundary


def _gate_value(answer: Answer) -> str | float:
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    if isinstance(answer, ScoreAnswer):
        return score_argmax(answer)
    if isinstance(answer, NoulAnswer):
        return answer.noul
    raise TypeError("unsupported answer type")


def _derived_confidence(answer: Answer, question: Mapping[str, Any]) -> float | None:
    if isinstance(answer, NoulAnswer):
        return None
    if answer.confidence is not None:
        return answer.confidence
    count = len(probability_keys_for_question(question))
    if count == 1:
        return 1.0
    probabilities = answer.probabilities
    if count <= 1 or not probabilities:
        return None
    return (count * max(probabilities.values()) - 1) / (count - 1)


def _served_model(response: JudgeResponse) -> str:
    return response.served_model or "unknown"


def _sum_usage(usages: Any) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for usage in usages:
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[str(key)] = totals.get(str(key), 0) + value
    return totals


def _summary(
    preset: Preset,
    tolerances: CalibrationTolerances,
    cases: int,
    baseline_models: Counter[str],
    candidate_models: Counter[str],
    baseline_usages: Sequence[Mapping[str, Any]],
    candidate_usages: Sequence[Mapping[str, Any]],
    choice_flips: int,
    max_probability_delta: float,
    max_score_delta: float,
    max_noul_delta: float,
    threshold_crossings: int,
    stable_drift: int,
    boundary_noise_cases: int,
    far_side_noise_cases: int,
    comparison_count: int,
    operational_error: str | None,
) -> dict[str, Any]:
    mixed_models = len(baseline_models) > 1 or len(candidate_models) > 1
    has_evidence = cases > 0
    within: bool | None
    if (
        not has_evidence
        or operational_error
        or mixed_models
        or boundary_noise_cases
    ):
        within = None
    else:
        within = (
            choice_flips <= tolerances.max_choice_flips
            and max_probability_delta <= tolerances.max_probability_delta
            and max_score_delta <= tolerances.max_score_delta
            and max_noul_delta <= tolerances.max_noul_delta
            and threshold_crossings <= tolerances.max_threshold_crossings
            and stable_drift == 0
        )
    summary: dict[str, Any] = {
        "record_type": "calibration_summary",
        "calibration_version": CALIBRATION_SCHEMA,
        "preset": preset.name,
        "preset_version": preset.version,
        "repeats": tolerances.repeats,
        "comparison_count": comparison_count,
        "qualified_cases": cases,
        "baseline_model_counts": dict(sorted(baseline_models.items())),
        "candidate_model_counts": dict(sorted(candidate_models.items())),
        "baseline_usage_totals": _sum_usage(baseline_usages),
        "candidate_usage_totals": _sum_usage(candidate_usages),
        "boundary_noise_cases": boundary_noise_cases,
        "far_side_noise_cases": far_side_noise_cases,
        "tolerances": tolerances.as_dict(),
        "choice_flips": choice_flips,
        "max_probability_delta_observed": max_probability_delta,
        "max_score_delta_observed": max_score_delta,
        "max_noul_delta_observed": max_noul_delta,
        "threshold_crossings": threshold_crossings,
        "stable_drift_cases": stable_drift,
        "within_tolerance": within,
    }
    if operational_error:
        summary["error"] = operational_error
    elif cases == 0:
        summary["error"] = "no qualifying cases"
    return summary


def _provenance(summary: Mapping[str, Any]) -> str:
    tolerances = summary["tolerances"]
    fields = (
        ("date", date.today().isoformat()),
        ("preset", summary["preset"]),
        ("preset_version", summary["preset_version"]),
        ("cases", summary["qualified_cases"]),
        ("repeats", summary["repeats"]),
        ("baseline_models", _format_models(summary["baseline_model_counts"])),
        ("candidate_models", _format_models(summary["candidate_model_counts"])),
        ("tol_threshold_margin", f"{tolerances['threshold_margin']:.4f}"),
        ("tol_max_choice_flips", tolerances["max_choice_flips"]),
        ("tol_max_probability_delta", f"{tolerances['max_probability_delta']:.4f}"),
        ("tol_max_score_delta", f"{tolerances['max_score_delta']:.4f}"),
        ("tol_max_noul_delta", f"{tolerances['max_noul_delta']:.4f}"),
        ("tol_max_threshold_crossings", tolerances["max_threshold_crossings"]),
        ("choice_flips", summary["choice_flips"]),
        (
            "max_probability_delta_observed",
            f"{summary['max_probability_delta_observed']:.4f}",
        ),
        ("max_score_delta_observed", f"{summary['max_score_delta_observed']:.4f}"),
        ("max_noul_delta_observed", f"{summary['max_noul_delta_observed']:.4f}"),
        ("threshold_crossings", summary["threshold_crossings"]),
        ("stable_drift_cases", summary["stable_drift_cases"]),
        ("boundary_noise_cases", summary["boundary_noise_cases"]),
        ("far_side_noise_cases", summary["far_side_noise_cases"]),
        ("within_tolerance", str(summary["within_tolerance"]).lower()),
    )
    date_value = fields[0][1]
    remaining = " ".join(f"{key}={value}" for key, value in fields[1:])
    return f"# jm calibrate {date_value}: {remaining}"


def _format_models(counts: Mapping[str, int]) -> str:
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _one_or_many(values: Sequence[Any]) -> Any:
    return values[0] if len(values) == 1 else list(values)


def _number_value(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _write_json(payload: Mapping[str, Any], stdout: TextIO) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
