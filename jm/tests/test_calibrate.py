from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from pathlib import Path

import pytest

from jm.answers import (
    ChoiceAnswer,
    ErrorResponse,
    JudgeResponse,
    NoulAnswer,
    ScoreAnswer,
)
from jm.cache import CacheEntry, CacheStore, battery_hash, canonical_json_bytes
from jm.calibrate import (
    CalibrationOperationalError,
    CalibrationTolerances,
    _comparison_record,
    _load_entries,
    _repeat_classification,
    run_calibration,
)
from jm.cli import main
from jm.presets import resolve_preset
from jm.runner import FormationReport, State, judge


def _seed(tmp_path: Path, *, preset_name: str = "jgrep") -> tuple[CacheStore, object]:
    preset = resolve_preset(preset_name)
    store = CacheStore(tmp_path / "cache")
    state = State(
        "case#1",
        "focus",
        {"query": "launch"},
        wire_context_keys=frozenset({"query"}),
    )
    tuple(
        judge(
            preset,
            (state,),
            formation_report=FormationReport((), ()),
            cache_store=store,
            judge_fn=lambda *_args: JudgeResponse(
                {"matches_query": NoulAnswer(0.8)},
                served_model="baseline",
                usage={"input_tokens": 10},
            ),
        )
    )
    return store, preset


def _seed_many(tmp_path: Path, count: int) -> tuple[CacheStore, object]:
    preset = resolve_preset("jgrep")
    store = CacheStore(tmp_path / "cache")
    states = tuple(
        State(
            f"case#{index}",
            f"focus-{index}",
            {"query": "launch"},
            wire_context_keys=frozenset({"query"}),
        )
        for index in range(1, count + 1)
    )
    tuple(
        judge(
            preset,
            states,
            formation_report=FormationReport(),
            cache_store=store,
            judge_fn=lambda *_args: JudgeResponse(
                {"matches_query": NoulAnswer(0.8)}, served_model="baseline"
            ),
        )
    )
    return store, preset


def test_calibration_selects_exact_v3_provenance_tuple(tmp_path: Path) -> None:
    store, preset = _seed(tmp_path)
    entries = _load_entries(preset, store)
    assert len(entries) == 1
    assert entries[0].battery_hash == battery_hash(preset.questions)


def test_calibration_uses_stored_wire_state_and_fresh_uid(tmp_path: Path) -> None:
    store, preset = _seed(tmp_path)
    seen: list[State] = []

    def judge(state: State, *_args: object) -> JudgeResponse:
        seen.append(state)
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.8)}, served_model="candidate"
        )

    stdout = io.StringIO()
    assert (
        run_calibration(
            preset,
            store,
            judge,
            stdout=stdout,
            stderr=io.StringIO(),
            tolerances=CalibrationTolerances(repeats=3),
        )
        == 0
    )
    assert len(seen) == 3
    assert len({state.api_payload["context"]["uid"] for state in seen}) == 3
    assert {state.state_ref for state in seen} == {"case#1"}
    assert seen[0].api_payload["context"]["query"] == "launch"
    assert isinstance(seen[0].api_payload["context"]["uid"], str)
    case = json.loads(stdout.getvalue().splitlines()[0])
    assert case["state_refs"] == ["case#1"]
    entry = _load_entries(preset, store)[0]
    target = entry.provenance_for(
        preset.name, preset.version, battery_hash(preset.questions)
    )
    assert target is not None
    expected = "sha256:" + hashlib.sha256(
        canonical_json_bytes(
            {
                "cache_key": entry.cache_key,
                "wire_state": entry.wire_state,
                "target_provenance": target,
            }
        )
    ).hexdigest()
    assert case["calibration_case_id"] == expected
    assert len(case["comparisons"][0]["candidate_repeats"]) == 3


def test_calibration_selects_the_matching_version_group(tmp_path: Path) -> None:
    store, first = _seed(tmp_path)
    second_data = dict(first.data)
    second_data["name"] = "jgrep-alt"
    second_data["version"] = "2"
    second = type(first)(second_data, first.path)
    tuple(
        judge(
            second,
            (State(
                "case#1",
                "focus",
                {"query": "launch"},
                wire_context_keys=frozenset({"query"}),
            ),),
            formation_report=FormationReport((), ()),
            cache_store=store,
            judge_fn=lambda *_args: JudgeResponse(
                {"matches_query": NoulAnswer(0.8)}, served_model="baseline"
            ),
        )
    )

    def candidate(*_args: object) -> JudgeResponse:
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.8)}, served_model="candidate"
        )

    for preset, state_ref in ((first, "case#1"), (second, "case#1")):
        stdout = io.StringIO()
        assert run_calibration(
            preset, store, candidate, stdout=stdout, stderr=io.StringIO()
        ) == 0
        case = json.loads(stdout.getvalue().splitlines()[0])
        assert case["state_refs"] == [state_ref]

    first_entries = _load_entries(first, store)
    second_entries = _load_entries(second, store)
    assert first_entries[0].cache_key == second_entries[0].cache_key
    assert second_entries[0].provenance_for(
        second.name, second.version, battery_hash(second.questions)
    ) is not None


def test_calibration_rejects_malformed_v3_entries_before_live_calls(
    tmp_path: Path,
) -> None:
    store, preset = _seed(tmp_path)
    path = next((tmp_path / "cache" / "answers").rglob("*.json"))
    payload = json.loads(path.read_text())
    payload["response"] = {}
    path.write_text(json.dumps(payload))
    with pytest.raises(CalibrationOperationalError):
        _load_entries(preset, store)


def test_calibration_v3_output_is_deterministic_across_pool_sizes(
    tmp_path: Path,
) -> None:
    store, preset = _seed(tmp_path)

    def candidate(*_args: object) -> JudgeResponse:
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.8)},
            served_model="candidate",
            usage={"input_tokens": 2},
        )

    outputs: list[str] = []
    for concurrency in (1, 4):
        stdout = io.StringIO()
        assert (
            run_calibration(
                preset,
                store,
                candidate,
                stdout=stdout,
                stderr=io.StringIO(),
                concurrency=concurrency,
            )
            == 0
        )
        outputs.append(stdout.getvalue())

    assert outputs[0] == outputs[1]
    comparison = json.loads(outputs[0].splitlines()[0])
    summary = json.loads(outputs[0].splitlines()[-1])
    assert comparison["record_type"] == "calibration_comparison"
    assert comparison["calibration_version"] == "jm.calibration/v3"
    assert summary["qualified_cases"] == 1
    assert summary["comparison_count"] == 1


def test_calibration_scheduler_runs_delayed_cases_in_parallel_and_orders_output(
    tmp_path: Path,
) -> None:
    store, preset = _seed_many(tmp_path, 4)
    lock = threading.Lock()
    active = 0
    maximum = 0

    def candidate(state: State, *_args: object) -> JudgeResponse:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.8)}, served_model="baseline"
        )

    stdout = io.StringIO()
    assert (
        run_calibration(
            preset,
            store,
            candidate,
            stdout=stdout,
            stderr=io.StringIO(),
            concurrency=4,
        )
        == 0
    )
    serial_stdout = io.StringIO()
    assert (
        run_calibration(
            preset,
            store,
            candidate,
            stdout=serial_stdout,
            stderr=io.StringIO(),
            concurrency=1,
        )
        == 0
    )
    assert maximum >= 2
    assert stdout.getvalue() == serial_stdout.getvalue()
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    comparisons = [
        record
        for record in records
        if record["record_type"] == "calibration_comparison"
    ]
    assert len(comparisons) == 4
    assert {record["state_ref"] for record in comparisons} == {
        "case#1",
        "case#2",
        "case#3",
        "case#4",
    }


def test_calibration_near_threshold_noise_has_a_null_verdict(tmp_path: Path) -> None:
    store, preset = _seed(tmp_path)

    def candidate(*_args: object) -> JudgeResponse:
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.74)}, served_model="baseline"
        )

    stdout = io.StringIO()
    assert (
        run_calibration(
            preset,
            store,
            candidate,
            stdout=stdout,
            stderr=io.StringIO(),
        )
        == 2
    )
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    comparison = records[0]
    summary = records[-1]
    assert comparison["boundary_noise"] is True
    assert comparison["within_tolerance"] is None
    assert summary["boundary_noise_cases"] == 1
    assert summary["within_tolerance"] is None


@pytest.mark.parametrize(
    ("candidate", "arguments", "expected"),
    [
        (NoulAnswer(0.8), [], 0),
        (NoulAnswer(0.9), [], 1),
        (ErrorResponse("candidate unavailable"), [], 2),
        (NoulAnswer(0.8), ["--concurrency", "0"], 64),
    ],
)
def test_calibration_cli_exit_matrix(
    tmp_path: Path,
    candidate: NoulAnswer | ErrorResponse,
    arguments: list[str],
    expected: int,
) -> None:
    store, _preset = _seed(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    def judge_fn(*_args: object) -> JudgeResponse | ErrorResponse:
        if isinstance(candidate, ErrorResponse):
            return candidate
        return JudgeResponse({"matches_query": candidate}, served_model="baseline")

    assert (
        main(
            ["calibrate", "--preset", "jgrep", *arguments],
            judge_fn=judge_fn,
            stdout=stdout,
            stderr=stderr,
            cache_store=store,
        )
        == expected
    )


def test_calibration_reuses_capped_scheduler_and_503_backoff(tmp_path: Path) -> None:
    preset = resolve_preset("jgrep")
    store = CacheStore(tmp_path / "cache")
    states = [
        State(
            f"case#{index}",
            f"focus-{index}",
            {"query": "launch"},
            wire_context_keys=frozenset({"query"}),
        )
        for index in range(8)
    ]
    tuple(
        judge(
            preset,
            states,
            formation_report=FormationReport(),
            cache_store=store,
            judge_fn=lambda *_args: JudgeResponse(
                {"matches_query": NoulAnswer(0.8)}, served_model="baseline"
            ),
        )
    )
    lock = threading.Lock()
    active = 0
    maximum = 0

    def candidate(state: State, *_args: object) -> JudgeResponse | ErrorResponse:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        if int(state.state_ref.rsplit("#", 1)[1]) < 4:
            return ErrorResponse("temporarily unavailable", http_status=503)
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.8)}, served_model="candidate"
        )

    stderr = io.StringIO()
    assert (
        run_calibration(
            preset,
            store,
            candidate,
            stdout=io.StringIO(),
            stderr=stderr,
            concurrency=12,
        )
        == 2
    )
    assert maximum <= 8
    assert "code=concurrency_capped" in stderr.getvalue()
    assert "code=concurrency_backoff" in stderr.getvalue()


def test_calibration_argmax_crossing_is_threshold_side_change() -> None:
    baseline = ScoreAnswer(1.99, probabilities={"1": 1.0})
    entry = CacheEntry(
        "key",
        {"focus": "focus", "context": {"state_ref": "hunk#1"}},
        "battery",
        {},
        ({"state_refs": ["hunk#1"]},),
        "model",
        "model",
        None,
        "now",
        JudgeResponse({"risk": baseline}, served_model="baseline"),
    )
    question = {"type": "score"}
    threshold = {"fail_at_least": 2}
    same_side, _ = _comparison_record(
        entry,
        "risk",
        question,
        threshold,
        baseline,
        [JudgeResponse({"risk": ScoreAnswer(2.01, probabilities={"1": 1.0})})],
        [ScoreAnswer(2.01, probabilities={"1": 1.0})],
        CalibrationTolerances(),
        {"state_refs": ["hunk#1"]},
    )
    crossed, _ = _comparison_record(
        entry,
        "risk",
        question,
        threshold,
        baseline,
        [JudgeResponse({"risk": ScoreAnswer(2.01, probabilities={"2": 1.0})})],
        [ScoreAnswer(2.01, probabilities={"2": 1.0})],
        CalibrationTolerances(),
        {"state_refs": ["hunk#1"]},
    )
    assert same_side["argmax_crossing"] is False
    assert crossed["argmax_crossing"] is True
    assert crossed["score_delta"] == 0.02


def test_calibration_noise_split_covers_choice_score_and_noul() -> None:
    choice = _repeat_classification(
        "choice", ChoiceAnswer("a"), [ChoiceAnswer("b"), ChoiceAnswer("c")], 0.05, []
    )
    score = _repeat_classification(
        "score",
        ScoreAnswer(1.0, probabilities={"1": 1.0}),
        [
            ScoreAnswer(3.0, probabilities={"3": 1.0}),
            ScoreAnswer(3.0, probabilities={"3": 1.0}),
        ],
        0.05,
        [],
    )
    noul = _repeat_classification(
        "noul",
        NoulAnswer(0.2),
        [NoulAnswer(0.8), NoulAnswer(0.9)],
        0.05,
        [{"target": 0.75, "near_threshold": False}],
    )
    assert choice == (False, False, True)
    assert score == (True, False, True)
    assert noul == (False, False, True)
