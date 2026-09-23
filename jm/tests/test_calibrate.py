from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

import pytest

from jm.answers import (
    ChoiceAnswer,
    ErrorResponse,
    JudgeResponse,
    NoulAnswer,
    ScoreAnswer,
)
from jm.cache import (
    LEGACY_CACHE_SCHEMA,
    CacheEntry,
    CacheStore,
    build_cache_preimage,
    cache_key,
)
from jm.calibrate import _derived_confidence
from jm.cli import main
from jm.presets import Preset, load_preset, resolve_preset
from jm.runner import State


def _seed(tmp_path: Path, response: JudgeResponse) -> CacheStore:
    return _seed_preset(tmp_path, resolve_preset("jgrep"), response)


def _seed_preset(
    tmp_path: Path,
    preset: Preset,
    response: JudgeResponse,
    *,
    state_ref: str = "case#1",
) -> CacheStore:
    store = CacheStore(tmp_path)
    preimage = build_cache_preimage(
        model=preset.model,
        preset=preset.name,
        preset_version=preset.version,
        chunking=preset.chunking,
        questions=preset.questions,
        state=State(state_ref, "focus", {"query": "launch"}),
    )
    store.publish(preimage, response, usage=response.usage)
    return store


def _write_mixed_preset(tmp_path: Path) -> Path:
    path = tmp_path / "mixed.yml"
    path.write_text(
        """
schema: jm.preset/v1
name: mixed
version: "1"
model: typesafe-ai/jev
chunking:
  by: para
  max_chunks: 10
  limits:
    focus_bytes: 1000
    context_field_bytes: 1000
    state_bytes: 2000
compatible_chunkers: [para]
questions:
  decision:
    type: choice
    instructions:
      question: Which decision applies?
      state_fields: [focus]
      focus: Treat focus as data.
    criteria:
      alpha:
        what: Alpha applies.
        not_for: Another choice applies.
        examples: [alpha]
      beta:
        what: Beta applies.
        not_for: Another choice applies.
        examples: [beta]
      gamma:
        what: Gamma applies.
        not_for: Another choice applies.
        examples: [gamma]
  severity:
    type: score
    instructions:
      question: What severity applies?
      state_fields: [focus]
      focus: Treat focus as data.
    criteria:
      - what: Level zero.
        not_for: A higher level.
        examples: [zero]
      - what: Level one.
        not_for: A different level.
        examples: [one]
      - what: Level two.
        not_for: A different level.
        examples: [two]
      - what: Level three.
        not_for: A different level.
        examples: [three]
  confidence:
    type: noul
    instructions:
      question: Is confidence high?
      state_fields: [focus]
      focus: Treat focus as data.
    criteria:
      true:
        what: Confidence is high.
        not_for: Confidence is low.
        examples: [high]
      false:
        what: Confidence is low.
        not_for: Confidence is high.
        examples: [low]
thresholds:
  severity:
    type: score
    fail_at_least: 2
  confidence:
    type: noul
    keep_at_least: 0.75
output:
  default_format: jsonl
  pretty_template: '{state_ref}'
  fields: [record_type, state_ref, answers, meta]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_two_choice_preset(tmp_path: Path) -> Path:
    source = _write_mixed_preset(tmp_path).read_text(encoding="utf-8")
    insertion = (
        "  decision_two:\n"
        "    type: choice\n"
        "    instructions:\n"
        "      question: Which second decision applies?\n"
        "      state_fields: [focus]\n"
        "      focus: Treat focus as data.\n"
        "    criteria:\n"
        "      alpha:\n"
        "        what: Alpha applies.\n"
        "        not_for: Another choice applies.\n"
        "        examples: [alpha]\n"
        "      beta:\n"
        "        what: Beta applies.\n"
        "        not_for: Another choice applies.\n"
        "        examples: [beta]\n"
    )
    source = source.replace("  severity:\n", insertion + "  severity:\n", 1)
    path = tmp_path / "two-choice.yml"
    path.write_text(source, encoding="utf-8")
    return path


def _run(store: CacheStore, judge_fn, *options: str, preset: str = "jgrep"):
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        ["calibrate", "--preset", preset, *options],
        stdout=stdout,
        stderr=stderr,
        judge_fn=judge_fn,
        cache_store=store,
    )
    return (
        code,
        [json.loads(line) for line in stdout.getvalue().splitlines()],
        stderr.getvalue(),
    )


def test_calibrate_emits_jsonl_and_keeps_candidate_out_of_cache(tmp_path: Path) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse(
            {"matches_query": NoulAnswer(0.80)},
            served_model="baseline-1",
            usage={"input_tokens": 10},
        ),
    )
    before = sorted(store.root.rglob("*.json"))

    def judge(state, _questions, _model):
        assert state.context["uid"]
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.82)},
            served_model="candidate-1",
            usage={"input_tokens": 12},
        )

    code, records, stderr = _run(store, judge)

    assert code == 0
    assert [record["record_type"] for record in records] == [
        "calibration_case",
        "calibration_summary",
    ]
    assert records[0]["baseline_usage"] == {"input_tokens": 10}
    assert records[0]["candidate_repeats"][0]["usage"] == {"input_tokens": 12}
    assert records[1]["candidate_model_counts"] == {"candidate-1": 1}
    assert "within tolerance" in stderr
    assert stderr.strip() == (
        f"jm calibrate: within tolerance\n"
        f"# jm calibrate {date.today().isoformat()}: "
        "preset=jgrep preset_version=1 cases=1 repeats=1 "
        "baseline=baseline-1 candidate=candidate-1 "
        "baseline_models=baseline-1:1 candidate_models=candidate-1:1 "
        "tol_threshold_margin=0.0500 tol_max_choice_flips=0 "
        "tol_max_probability_delta=0.0500 tol_max_score_delta=0.5000 "
        "tol_max_noul_delta=0.0500 tol_max_threshold_crossings=0 "
        "choice_flips=0 max_probability_delta_observed=0.0000 "
        "max_score_delta_observed=0.0000 max_noul_delta_observed=0.0200 "
        "threshold_crossings=0 stable_drift=0 boundary_noise=0 "
        "within_tolerance=true"
    )
    assert sorted(store.root.rglob("*.json")) == before


def test_repeats_report_stable_drift_and_fresh_uids(tmp_path: Path) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse({"matches_query": NoulAnswer(0.50)}, served_model="baseline"),
    )
    uids: list[str] = []

    def judge(state, _questions, _model):
        uids.append(state.context["uid"])
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.70)}, served_model="candidate"
        )

    code, records, _ = _run(store, judge, "--repeats", "2")
    case = records[0]
    summary = records[-1]

    assert code == 1
    assert len(set(uids)) == 2
    assert len(case["candidate_repeats"]) == 2
    assert case["stable_drift"] is True
    assert case["boundary_noise"] is False
    assert summary["stable_drift"] == 1
    assert summary["within_tolerance"] is False


def test_repeats_within_tolerance_are_not_boundary_noise(tmp_path: Path) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse({"matches_query": NoulAnswer(0.50)}, served_model="baseline"),
    )
    values = iter((0.51, 0.52))

    def judge(*_args):
        return JudgeResponse(
            {"matches_query": NoulAnswer(next(values))}, served_model="candidate"
        )

    code, records, _ = _run(store, judge, "--repeats", "2")

    assert code == 0
    assert records[0]["stable_drift"] is False
    assert records[0]["boundary_noise"] is False
    assert records[-1]["stable_drift"] == 0
    assert records[-1]["boundary_noise"] == 0
    assert records[-1]["within_tolerance"] is True


def test_zero_repeat_delta_has_no_direction_or_noise(tmp_path: Path) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse({"matches_query": NoulAnswer(0.50)}, served_model="baseline"),
    )
    values = iter((0.50, 0.51))

    def judge(*_args):
        return JudgeResponse(
            {"matches_query": NoulAnswer(next(values))}, served_model="candidate"
        )

    code, records, _ = _run(store, judge, "--repeats", "2")

    assert code == 0
    assert records[0]["stable_drift"] is False
    assert records[0]["boundary_noise"] is False
    assert records[-1]["boundary_noise"] == 0
    assert records[-1]["within_tolerance"] is True


@pytest.mark.parametrize(
    ("baseline", "values"),
    (
        (0.50, (0.49, 0.51)),
        (0.74, (0.73, 0.76)),
        (0.50, (0.46, 0.54)),
    ),
    ids=("direction-change", "threshold-straddle", "spread-bound"),
)
def test_within_tolerance_boundary_shapes_are_indeterminate(
    tmp_path: Path, baseline: float, values: tuple[float, float]
) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse({"matches_query": NoulAnswer(baseline)}, served_model="baseline"),
    )
    candidates = iter(values)

    def judge(*_args):
        return JudgeResponse(
            {"matches_query": NoulAnswer(next(candidates))}, served_model="candidate"
        )

    code, records, _ = _run(store, judge, "--repeats", "2")

    assert code == 2
    assert records[0]["stable_drift"] is False
    assert records[0]["boundary_noise"] is True
    assert records[-1]["boundary_noise"] == 1
    assert records[-1]["within_tolerance"] is None


def test_boundary_noise_is_indeterminate_not_drift(tmp_path: Path) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse({"matches_query": NoulAnswer(0.50)}, served_model="baseline"),
    )
    values = iter((0.70, 0.30))

    def judge(*_args):
        return JudgeResponse(
            {"matches_query": NoulAnswer(next(values))}, served_model="candidate"
        )

    code, records, _ = _run(store, judge, "--repeats", "2")

    assert code == 2
    assert records[0]["boundary_noise"] is True
    assert records[0]["stable_drift"] is False
    assert records[-1]["boundary_noise"] == 1
    assert records[-1]["stable_drift"] == 0
    assert records[-1]["within_tolerance"] is None


def test_no_cases_is_not_a_passing_zero_delta(tmp_path: Path) -> None:
    code, records, stderr = _run(
        CacheStore(tmp_path),
        lambda *_: (_ for _ in ()).throw(AssertionError("must not call live API")),
    )

    assert code == 2
    assert records[-1]["cases"] == 0
    assert records[-1]["within_tolerance"] is None
    assert "no qualifying cases" in stderr


def test_no_key_still_emits_empty_summary_without_live_calls(
    tmp_path: Path, monkeypatch
) -> None:
    for name in ("VERCEL_AI_GATEWAY", "AI_GATEWAY_API_KEY", "VERCEL_JEV_KEY"):
        monkeypatch.delenv(name, raising=False)

    code, records, stderr = _run(CacheStore(tmp_path), None)

    assert code == 2
    assert records[-1]["comparison_count"] == 0
    assert records[-1]["within_tolerance"] is None
    assert "no qualifying cases" in stderr


def test_v1_cache_is_read_without_being_overwritten(tmp_path: Path) -> None:
    preset = resolve_preset("jgrep")
    response = JudgeResponse(
        {"matches_query": NoulAnswer(0.80)}, served_model="baseline"
    )
    preimage = build_cache_preimage(
        model=preset.model,
        preset=preset.name,
        preset_version=preset.version,
        chunking=preset.chunking,
        questions=preset.questions,
        state=State("case#1", "focus", {"query": "launch"}),
        cache_schema=LEGACY_CACHE_SCHEMA,
    )
    entry = CacheEntry(cache_key(preimage), preimage, response, None, "now")
    store = CacheStore(tmp_path)
    path = store.path_for(entry.cache_key)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(entry.to_dict()), encoding="utf-8")

    code, records, _ = _run(
        store,
        lambda *_: JudgeResponse(
            {"matches_query": NoulAnswer(0.80)}, served_model="candidate"
        ),
    )

    assert code == 0
    assert records[0]["baseline_served_model"] == "baseline"
    assert (
        json.loads(path.read_text(encoding="utf-8"))["cache_schema"]
        == LEGACY_CACHE_SCHEMA
    )


def test_confidence_uses_full_question_criteria_for_sparse_probabilities() -> None:
    question = {"criteria": ["low", "medium", "high", "critical"]}
    answer = ScoreAnswer(2, probabilities={"2": 0.7})
    assert _derived_confidence(answer, question) == (4 * 0.7 - 1) / 3


def test_confidence_uses_full_choice_options_for_sparse_probabilities() -> None:
    question = {"options": ["low", "medium", "high"]}
    answer = ChoiceAnswer("high", probabilities={"high": 0.7})
    assert _derived_confidence(answer, question) == (3 * 0.7 - 1) / 2


def test_model_counts_are_per_case_and_repeat_not_per_question(tmp_path: Path) -> None:
    preset = resolve_preset("diff-risk-heat")
    answers = {
        question_id: (
            ScoreAnswer(
                1,
                probabilities={"0": 0.1, "1": 0.8, "2": 0.05, "3": 0.05},
                confidence=0.8,
            )
            if question["type"] == "score"
            else NoulAnswer(0.5)
        )
        for question_id, question in preset.questions.items()
    }
    store = _seed_preset(
        tmp_path,
        preset,
        JudgeResponse(answers, served_model="baseline"),
    )

    def judge(*_args):
        return JudgeResponse(answers, served_model="candidate")

    code, records, _ = _run(
        store, judge, "--repeats", "2", preset="diff-risk-heat"
    )

    assert code == 0
    assert records[-1]["cases"] == 1
    assert records[-1]["comparison_count"] == len(preset.questions)
    assert records[-1]["baseline_model_counts"] == {"baseline": 1}
    assert records[-1]["candidate_model_counts"] == {"candidate": 2}


def test_single_choice_flip_uses_choice_tolerance_not_repeat_classification(
    tmp_path: Path,
) -> None:
    preset = load_preset(_write_mixed_preset(tmp_path))
    baseline_answers = {
        "decision": ChoiceAnswer(
            "alpha", {"alpha": 0.8, "beta": 0.1, "gamma": 0.1}, confidence=0.7
        ),
        "severity": ScoreAnswer(
            1, probabilities={"0": 0.2, "1": 0.8}, confidence=0.2
        ),
        "confidence": NoulAnswer(0.72),
    }
    candidate_answers = {
        **baseline_answers,
        "decision": ChoiceAnswer(
            "beta", {"alpha": 0.8, "beta": 0.1, "gamma": 0.1}
        ),
    }
    store = _seed_preset(
        tmp_path, preset, JudgeResponse(baseline_answers, served_model="baseline")
    )

    code, records, _ = _run(
        store,
        lambda *_: JudgeResponse(candidate_answers, served_model="candidate"),
        "--max-choice-flips",
        "1",
        preset=str(preset.path),
    )

    assert code == 0
    choice_record = next(
        record for record in records if record.get("question_id") == "decision"
    )
    assert choice_record["choice_flip"] is True
    assert choice_record["stable_drift"] is False
    assert choice_record["boundary_noise"] is False
    assert records[-1]["stable_drift"] == 0
    assert records[-1]["boundary_noise"] == 0


def test_calibrate_compares_all_primitives_and_reports_provenance(
    tmp_path: Path,
) -> None:
    preset = load_preset(_write_mixed_preset(tmp_path))
    baseline = JudgeResponse(
        {
            "decision": ChoiceAnswer("alpha", {"alpha": 0.8}, confidence=0.7),
            "severity": ScoreAnswer(
                1,
                probabilities={"0": 0.2, "1": 0.3, "2": 0.4, "3": 0.1},
                confidence=0.2,
            ),
            "confidence": NoulAnswer(0.72),
        },
        served_model="baseline",
        usage={"input_tokens": 10},
    )
    store = _seed_preset(tmp_path, preset, baseline)

    def judge(*_args):
        return JudgeResponse(
            {
                "decision": ChoiceAnswer("beta", {"beta": 0.7}),
                "severity": ScoreAnswer(3, probabilities={"2": 0.9}),
                "confidence": NoulAnswer(0.78),
            },
            served_model="candidate",
            usage={"input_tokens": 12},
        )

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "calibrate",
            "--preset",
            str(preset.path),
            "--repeats",
            "2",
        ],
        stdout=stdout,
        stderr=stderr,
        judge_fn=judge,
        cache_store=store,
    )
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    by_question = {record["question_id"]: record for record in records[:-1]}
    summary = records[-1]

    assert code == 1
    assert set(by_question) == {"decision", "severity", "confidence"}
    assert by_question["decision"]["choice_flip"] is True
    assert by_question["decision"]["probability_delta"] == 0.8
    assert by_question["decision"]["gate_values"] == {
        "baseline": "alpha",
        "candidate": ["beta", "beta"],
    }
    assert by_question["decision"]["derived_confidence"]["baseline"] == 0.7
    assert by_question["decision"]["derived_confidence"]["candidate"] == pytest.approx(
        [0.55, 0.55]
    )
    assert by_question["severity"]["score_delta"] == 2.0
    assert by_question["severity"]["probability_delta"] == 0.5
    assert by_question["severity"]["gate_values"] == {
        "baseline": 1,
        "candidate": [3, 3],
    }
    assert by_question["severity"]["derived_confidence"]["candidate"] == pytest.approx(
        [0.866666666667] * 2
    )
    assert by_question["severity"]["thresholds"] == [
        {
            "field": "score",
            "operator": ">=",
            "target": 2,
            "baseline_value": 1,
            "candidate_value": [3, 3],
            "distance": 1.0,
            "near_threshold": False,
            "crossing": True,
        }
    ]
    assert by_question["confidence"]["noul_delta"] == 0.06
    assert by_question["confidence"]["thresholds"][0]["near_threshold"] is True
    assert by_question["confidence"]["thresholds"][0]["crossing"] is True
    assert by_question["confidence"]["candidate_usage"] == {"input_tokens": 24}
    assert summary["candidate_usage"] == {"input_tokens": 24}
    assert summary["baseline_model_counts"] == {"baseline": 1}
    assert summary["candidate_model_counts"] == {"candidate": 2}
    assert summary["choice_flips"] == 1
    assert summary["threshold_crossings"] == 1
    assert stderr.getvalue().strip() == (
        "jm calibrate: drift exceeded tolerance\n"
        f"# jm calibrate {date.today().isoformat()}: "
        "preset=mixed preset_version=1 cases=1 repeats=2 "
        "baseline=baseline candidate=candidate "
        "baseline_models=baseline:1 candidate_models=candidate:2 "
        "tol_threshold_margin=0.0500 tol_max_choice_flips=0 "
        "tol_max_probability_delta=0.0500 tol_max_score_delta=0.5000 "
        "tol_max_noul_delta=0.0500 tol_max_threshold_crossings=0 "
        "choice_flips=1 max_probability_delta_observed=0.8000 "
        "max_score_delta_observed=2.0000 max_noul_delta_observed=0.0600 "
        "threshold_crossings=1 stable_drift=3 boundary_noise=0 "
        "within_tolerance=false"
    )


def test_threshold_crossings_count_cache_cases(tmp_path: Path) -> None:
    preset = load_preset(_write_mixed_preset(tmp_path))
    baseline_answers = {
        "decision": ChoiceAnswer("alpha", {"alpha": 0.8}, confidence=0.8),
        "severity": ScoreAnswer(1.99, confidence=0.5),
        "confidence": NoulAnswer(0.74),
    }
    candidate_answers = {
        "decision": ChoiceAnswer("alpha", {"alpha": 0.8}, confidence=0.8),
        "severity": ScoreAnswer(2.01),
        "confidence": NoulAnswer(0.76),
    }
    store = _seed_preset(
        tmp_path,
        preset,
        JudgeResponse(baseline_answers, served_model="baseline"),
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "calibrate",
            "--preset",
            str(preset.path),
            "--max-threshold-crossings",
            "1",
        ],
        stdout=stdout,
        stderr=stderr,
        judge_fn=lambda *_: JudgeResponse(
            candidate_answers, served_model="candidate"
        ),
        cache_store=store,
    )
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    summary = records[-1]

    assert code == 0
    assert len(records) == 4
    assert all(record["record_type"] == "calibration_case" for record in records[:-1])
    assert all(record["thresholds"][0]["crossing"] for record in records[1:3])
    assert summary["cases"] == 1
    assert summary["comparison_count"] == 3
    assert summary["threshold_crossings"] == 1
    assert stderr.getvalue().strip() == (
        "jm calibrate: within tolerance\n"
        f"# jm calibrate {date.today().isoformat()}: "
        "preset=mixed preset_version=1 cases=1 repeats=1 "
        "baseline=baseline candidate=candidate "
        "baseline_models=baseline:1 candidate_models=candidate:1 "
        "tol_threshold_margin=0.0500 tol_max_choice_flips=0 "
        "tol_max_probability_delta=0.0500 tol_max_score_delta=0.5000 "
        "tol_max_noul_delta=0.0500 tol_max_threshold_crossings=1 "
        "choice_flips=0 max_probability_delta_observed=0.0000 "
        "max_score_delta_observed=0.0200 max_noul_delta_observed=0.0200 "
        "threshold_crossings=1 stable_drift=0 boundary_noise=0 "
        "within_tolerance=true"
    )


def test_choice_flips_count_cache_cases(tmp_path: Path) -> None:
    preset = load_preset(_write_two_choice_preset(tmp_path))
    baseline_answers = {
        "decision": ChoiceAnswer(
            "alpha", {"alpha": 0.8, "beta": 0.2}, confidence=0.6
        ),
        "decision_two": ChoiceAnswer(
            "alpha", {"alpha": 0.8, "beta": 0.2}, confidence=0.6
        ),
        "severity": ScoreAnswer(1.0, confidence=0.5),
        "confidence": NoulAnswer(0.5),
    }
    candidate_answers = {
        "decision": ChoiceAnswer(
            "beta", {"alpha": 0.8, "beta": 0.2}, confidence=0.6
        ),
        "decision_two": ChoiceAnswer(
            "beta", {"alpha": 0.8, "beta": 0.2}, confidence=0.6
        ),
        "severity": ScoreAnswer(1.0),
        "confidence": NoulAnswer(0.5),
    }
    store = _seed_preset(
        tmp_path,
        preset,
        JudgeResponse(baseline_answers, served_model="baseline"),
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "calibrate",
            "--preset",
            str(preset.path),
            "--max-choice-flips",
            "1",
        ],
        stdout=stdout,
        stderr=stderr,
        judge_fn=lambda *_: JudgeResponse(
            candidate_answers, served_model="candidate"
        ),
        cache_store=store,
    )
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    by_question = {record["question_id"]: record for record in records[:-1]}
    summary = records[-1]

    assert code == 0
    assert by_question["decision"]["choice_flip"] is True
    assert by_question["decision_two"]["choice_flip"] is True
    assert summary["cases"] == 1
    assert summary["comparison_count"] == 4
    assert summary["choice_flips"] == 1
    assert stderr.getvalue().strip() == (
        "jm calibrate: within tolerance\n"
        f"# jm calibrate {date.today().isoformat()}: "
        "preset=mixed preset_version=1 cases=1 repeats=1 "
        "baseline=baseline candidate=candidate "
        "baseline_models=baseline:1 candidate_models=candidate:1 "
        "tol_threshold_margin=0.0500 tol_max_choice_flips=1 "
        "tol_max_probability_delta=0.0500 tol_max_score_delta=0.5000 "
        "tol_max_noul_delta=0.0500 tol_max_threshold_crossings=0 "
        "choice_flips=1 max_probability_delta_observed=0.0000 "
        "max_score_delta_observed=0.0000 max_noul_delta_observed=0.0000 "
        "threshold_crossings=0 stable_drift=0 boundary_noise=0 "
        "within_tolerance=true"
    )


def test_invalid_usage_and_operational_error_exit_codes(tmp_path: Path) -> None:
    stderr = io.StringIO()
    assert (
        main(
            ["calibrate", "--preset", "jgrep", "--repeats", "0"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 64
    )
    assert "positive integer" in stderr.getvalue()

    store = _seed(tmp_path, JudgeResponse({"matches_query": NoulAnswer(0.5)}))
    code, records, _ = _run(
        store,
        lambda *_: ErrorResponse("gateway unavailable"),
    )
    assert code == 2
    assert records[-1]["within_tolerance"] is None


def test_missing_preset_name_and_path_are_usage_errors(tmp_path: Path) -> None:
    for identifier in ("missing-preset", str(tmp_path / "missing.yml")):
        stderr = io.StringIO()
        code = main(
            ["calibrate", "--preset", identifier],
            stdout=io.StringIO(),
            stderr=stderr,
            judge_fn=lambda *_: pytest.fail("must not call live API"),
        )
        assert code == 64
        assert stderr.getvalue().startswith("jm: error:")


@pytest.mark.parametrize("kind", ("malformed", "partial"))
def test_malformed_and_partial_cache_entries_fail_before_live_calls(
    tmp_path: Path, kind: str
) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse({"matches_query": NoulAnswer(0.5)}, served_model="baseline"),
    )
    path = next(store.root.rglob("*.json"))
    if kind == "malformed":
        path.write_text("not json", encoding="utf-8")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["answers"]["matches_query"]
        path.write_text(json.dumps(payload), encoding="utf-8")

    calls = 0

    def judge(*_args):
        nonlocal calls
        calls += 1
        return JudgeResponse({"matches_query": NoulAnswer(0.5)})

    code, records, _ = _run(store, judge)

    assert code == 2
    assert calls == 0
    assert records[-1]["within_tolerance"] is None


@pytest.mark.parametrize(
    "kind", ("empty", "missing_payload_preset", "missing_preimage_preset")
)
def test_missing_or_partial_preset_metadata_fails_before_live_calls(
    tmp_path: Path, kind: str
) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse({"matches_query": NoulAnswer(0.5)}, served_model="baseline"),
    )
    path = next(store.root.rglob("*.json"))
    if kind == "empty":
        payload: dict[str, object] = {}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if kind == "missing_payload_preset":
            del payload["preset"]
        else:
            del payload["preimage"]["preset"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    calls = 0

    def judge(*_args):
        nonlocal calls
        calls += 1
        return JudgeResponse({"matches_query": NoulAnswer(0.5)})

    code, records, _ = _run(store, judge)

    assert code == 2
    assert calls == 0
    assert records[-1]["within_tolerance"] is None


def test_mixed_preset_cache_entries_fail_before_live_calls(tmp_path: Path) -> None:
    preset = resolve_preset("jgrep")
    store = CacheStore(tmp_path)
    first_preimage = build_cache_preimage(
        model=preset.model,
        preset=preset.name,
        preset_version=preset.version,
        chunking=preset.chunking,
        questions=preset.questions,
        state=State("case#1", "focus", {"query": "launch"}),
    )
    store.publish(
        first_preimage,
        JudgeResponse({"matches_query": NoulAnswer(0.5)}),
    )
    second_preimage = build_cache_preimage(
        model=preset.model,
        preset=preset.name,
        preset_version="2",
        chunking=preset.chunking,
        questions=preset.questions,
        state=State("case#2", "focus", {"query": "launch"}),
    )
    store.publish(
        second_preimage,
        JudgeResponse({"matches_query": NoulAnswer(0.5)}),
    )

    calls = 0

    def judge(*_args):
        nonlocal calls
        calls += 1
        return JudgeResponse({"matches_query": NoulAnswer(0.5)})

    code, records, _ = _run(store, judge)

    assert code == 2
    assert calls == 0
    assert records[-1]["within_tolerance"] is None


def test_cache_path_digest_mismatch_fails_before_live_calls(tmp_path: Path) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse({"matches_query": NoulAnswer(0.5)}, served_model="baseline"),
    )
    path = next(store.root.rglob("*.json"))
    wrong_key = "sha256:" + "0" * 64
    wrong_path = store.path_for(wrong_key)
    wrong_path.parent.mkdir(parents=True, exist_ok=True)
    path.rename(wrong_path)

    calls = 0

    def judge(*_args):
        nonlocal calls
        calls += 1
        return JudgeResponse({"matches_query": NoulAnswer(0.5)})

    code, records, _ = _run(store, judge)

    assert code == 2
    assert calls == 0
    assert records[-1]["within_tolerance"] is None


def test_command_tolerance_overrides_preset_tolerance(tmp_path: Path) -> None:
    source = resolve_preset("jgrep").path.read_text(encoding="utf-8")
    path = tmp_path / "strict.yml"
    path.write_text(
        source.replace(
            "thresholds:\n",
            "calibration:\n"
            "  schema: jm.calibration/v1\n"
            "  max_noul_delta: 0.001\n"
            "thresholds:\n",
        ),
        encoding="utf-8",
    )
    preset = load_preset(path)
    store = _seed_preset(
        tmp_path,
        preset,
        JudgeResponse({"matches_query": NoulAnswer(0.50)}, served_model="baseline"),
    )

    code, records, _ = _run(
        store,
        lambda *_: JudgeResponse(
            {"matches_query": NoulAnswer(0.52)}, served_model="candidate"
        ),
        "--max-noul-delta",
        "0.05",
        preset=str(path),
    )

    assert code == 0
    assert records[-1]["tolerances"]["max_noul_delta"] == 0.05


def test_mixed_candidate_models_do_not_make_a_decision(tmp_path: Path) -> None:
    store = _seed(
        tmp_path,
        JudgeResponse(
            {"matches_query": NoulAnswer(0.5)},
            served_model="baseline",
            usage={"input_tokens": 10},
        ),
    )
    models = iter(("candidate-a", "candidate-b"))
    usages = iter(({"input_tokens": 2}, {"input_tokens": 3}))

    def judge(*_args):
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.5)},
            served_model=next(models),
            usage=next(usages),
        )

    code, records, stderr = _run(store, judge, "--repeats", "2")
    case = records[0]
    assert code == 2
    assert case["candidate_repeats"] == [
        {
            "repeat": 1,
            "answer": {"type": "noul", "noul": 0.5},
            "usage": {"input_tokens": 2},
            "served_model": "candidate-a",
        },
        {
            "repeat": 2,
            "answer": {"type": "noul", "noul": 0.5},
            "usage": {"input_tokens": 3},
            "served_model": "candidate-b",
        },
    ]
    assert case["candidate_usage"] == {"input_tokens": 5}
    assert records[-1]["candidate_model_counts"] == {
        "candidate-a": 1,
        "candidate-b": 1,
    }
    assert records[-1]["candidate_usage"] == {"input_tokens": 5}
    assert records[-1]["within_tolerance"] is None
    assert stderr.strip() == (
        "jm calibrate: calibration decision is indeterminate\n"
        f"# jm calibrate {date.today().isoformat()}: "
        "preset=jgrep preset_version=1 cases=1 repeats=2 "
        "baseline=baseline candidate=mixed "
        "baseline_models=baseline:1 "
        "candidate_models=candidate-a:1,candidate-b:1 "
        "tol_threshold_margin=0.0500 tol_max_choice_flips=0 "
        "tol_max_probability_delta=0.0500 tol_max_score_delta=0.5000 "
        "tol_max_noul_delta=0.0500 tol_max_threshold_crossings=0 "
        "choice_flips=0 max_probability_delta_observed=0.0000 "
        "max_score_delta_observed=0.0000 max_noul_delta_observed=0.0000 "
        "threshold_crossings=0 stable_drift=0 boundary_noise=0 "
        "within_tolerance=none"
    )


def test_mixed_baseline_models_do_not_make_a_decision(tmp_path: Path) -> None:
    preset = resolve_preset("jgrep")
    store = CacheStore(tmp_path)
    for index, served_model in enumerate(("baseline-a", "baseline-b"), start=1):
        preimage = build_cache_preimage(
            model=preset.model,
            preset=preset.name,
            preset_version=preset.version,
            chunking=preset.chunking,
            questions=preset.questions,
            state=State(f"case#{index}", "focus", {"query": "launch"}),
        )
        store.publish(
            preimage,
            JudgeResponse(
                {"matches_query": NoulAnswer(0.5)},
                served_model=served_model,
            ),
        )

    calls = 0

    def judge(*_args):
        nonlocal calls
        calls += 1
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.5)},
            served_model="candidate",
        )

    code, records, _ = _run(store, judge)

    assert code == 2
    assert calls == 2
    assert len(records) == 3
    assert records[-1]["baseline_model_counts"] == {
        "baseline-a": 1,
        "baseline-b": 1,
    }
    assert records[-1]["candidate_model_counts"] == {"candidate": 2}
    assert records[-1]["within_tolerance"] is None
