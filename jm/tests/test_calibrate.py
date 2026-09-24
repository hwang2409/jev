from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from jm.answers import JudgeResponse, NoulAnswer
from jm.cache import CacheStore, battery_hash
from jm.calibrate import CalibrationOperationalError, _load_entries, run_calibration
from jm.client import build_canonical_request
from jm.presets import resolve_preset
from jm.runner import State


def _seed(tmp_path: Path, *, preset_name: str = "jgrep") -> tuple[CacheStore, object]:
    preset = resolve_preset(preset_name)
    store = CacheStore(tmp_path / "cache")
    state = State(
        "case#1",
        "focus",
        {"query": "launch"},
        wire_context_keys=frozenset({"query"}),
    )
    request = build_canonical_request(state, preset.questions, model=preset.model)
    store.publish(
        request.payload["state"],
        JudgeResponse(
            {"matches_query": NoulAnswer(0.8)},
            served_model="baseline",
            usage={"input_tokens": 10},
        ),
        battery=preset.questions,
        preset=preset.name,
        preset_version=preset.version,
        configured_model=preset.model,
        transport_identity=request.transport_identity,
        state_ref="case#1",
        effective_preset={"chunking": preset.chunking},
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
