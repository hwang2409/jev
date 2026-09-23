from __future__ import annotations

import io
import json
from pathlib import Path

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
from jm.presets import resolve_preset
from jm.runner import State


def _seed(tmp_path: Path, response: JudgeResponse) -> CacheStore:
    preset = resolve_preset("jgrep")
    store = CacheStore(tmp_path)
    preimage = build_cache_preimage(
        model=preset.model,
        preset=preset.name,
        preset_version=preset.version,
        chunking=preset.chunking,
        questions=preset.questions,
        state=State("case#1", "focus", {"query": "launch"}),
    )
    store.publish(preimage, response, usage=response.usage)
    return store


def _run(store: CacheStore, judge_fn, *options: str):
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        ["calibrate", "--preset", "jgrep", *options],
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


def test_mixed_candidate_models_do_not_make_a_decision(tmp_path: Path) -> None:
    store = _seed(tmp_path, JudgeResponse({"matches_query": NoulAnswer(0.5)}))
    models = iter(("candidate-a", "candidate-b"))

    def judge(*_args):
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.5)}, served_model=next(models)
        )

    code, records, _ = _run(store, judge, "--repeats", "2")
    assert code == 2
    assert records[-1]["candidate_model_counts"] == {
        "candidate-a": 1,
        "candidate-b": 1,
    }
    assert records[-1]["within_tolerance"] is None


def test_comparison_helpers_cover_choice_and_score_gate_values() -> None:
    choice = ChoiceAnswer("yes", {"yes": 0.8, "no": 0.2})
    score = ScoreAnswer(2, probabilities={"0": 0.1, "2": 0.9})
    assert choice.choice == "yes"
    assert score.score == 2
