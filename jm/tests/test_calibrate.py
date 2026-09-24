from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from jm.answers import JudgeResponse, NoulAnswer
from jm.cache import CacheStore, battery_hash, canonical_json_bytes
from jm.calibrate import CalibrationOperationalError, _load_entries, run_calibration
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
        run_calibration(preset, store, judge, stdout=stdout, stderr=io.StringIO()) == 0
    )
    assert len(seen) == 1
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


def test_calibration_selects_the_matching_version_group(tmp_path: Path) -> None:
    store, first = _seed(tmp_path)
    second_data = dict(first.data)
    second_data["version"] = "2"
    second = type(first)(second_data, first.path)
    tuple(
        judge(
            second,
            (State(
                "case#2",
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

    for preset, state_ref in ((first, "case#1"), (second, "case#2")):
        stdout = io.StringIO()
        assert run_calibration(
            preset, store, candidate, stdout=stdout, stderr=io.StringIO()
        ) == 0
        case = json.loads(stdout.getvalue().splitlines()[0])
        assert case["state_refs"] == [state_ref]


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
