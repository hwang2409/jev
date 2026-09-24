from __future__ import annotations

import io
import threading
import time

import pytest

from jm.answers import ErrorResponse, JudgeResponse, NoulAnswer
from jm.cache import CacheStore
from jm.client import runtime_preset
from jm.runner import (
    ConfigurationError,
    FormationEvent,
    FormationReport,
    InputError,
    ResultFilter,
    State,
    emit,
    judge,
)


def _response() -> JudgeResponse:
    return JudgeResponse({"matches_query": NoulAnswer(0.9)})


def test_judge_has_no_output_side_effect_before_emitter(capfd) -> None:
    records = judge(
        "jgrep",
        [State("state-1", "focus")],
        formation_report=FormationReport(),
        judge_fn=lambda *_: _response(),
    )
    assert capfd.readouterr() == ("", "")
    list(records)
    assert capfd.readouterr() == ("", "")


def test_judge_validates_all_states_before_calling_the_judge() -> None:
    calls: list[str] = []

    def fake(state, *_args):
        calls.append(state.state_ref)
        return _response()

    with pytest.raises(InputError, match="duplicate state reference"):
        list(
            judge(
                "jgrep",
                [State("same", "one"), State("same", "two")],
                formation_report=FormationReport(),
                judge_fn=fake,
            )
        )
    assert calls == []


def test_judge_yields_a_completed_result_before_a_slow_state() -> None:
    release_slow = threading.Event()

    def fake(state, *_args):
        if state.state_ref == "slow":
            release_slow.wait(timeout=2)
        return _response()

    records = judge(
        "jgrep",
        [State("fast", "one"), State("slow", "two")],
        formation_report=FormationReport(),
        judge_fn=fake,
        concurrency=2,
    )
    started = time.monotonic()
    first = next(records)
    elapsed = time.monotonic() - started
    assert first.to_dict()["record_type"] == "result"
    assert elapsed < 1
    release_slow.set()
    assert next(records).to_dict()["record_type"] == "result"
    assert next(records).to_dict()["record_type"] == "coverage"


def test_judge_turns_api_exceptions_into_error_records() -> None:
    records = list(
        judge(
            "jgrep",
            [State("state-1", "focus")],
            formation_report=FormationReport(),
            judge_fn=lambda *_: (_ for _ in ()).throw(RuntimeError("secret")),
        )
    )
    assert records[0].to_dict()["error"]["message"] == "request failed"
    assert "secret" not in str(records[0].to_dict())
    assert records[-1].to_dict()["record_type"] == "coverage"


def test_judge_configuration_errors_raise() -> None:
    with pytest.raises(ConfigurationError):
        list(
            judge(
                "missing-preset",
                [],
                formation_report=FormationReport(),
                judge_fn=lambda *_: _response(),
            )
        )


def test_formation_event_rejects_mismatched_input_error_reason() -> None:
    with pytest.raises(ConfigurationError, match="require reason input_error"):
        FormationEvent(
            "input_error",
            "scan_cap",
            "invalid input",
            None,
            "stdin:byte=0,line=1",
            None,
        )


def test_public_judge_applies_503_backoff_diagnostics() -> None:
    def fake(state, *_args):
        if int(state.state_ref.rsplit("-", 1)[1]) < 4:
            return ErrorResponse("temporarily unavailable", http_status=503)
        return _response()

    records = list(
        judge(
            "jgrep",
            [State(f"state-{index}", "focus") for index in range(8)],
            formation_report=FormationReport(),
            judge_fn=fake,
            concurrency=4,
        )
    )
    diagnostics = [
        record.to_dict()["diagnostic"]
        for record in records
        if record.to_dict()["record_type"] == "diagnostic"
    ]
    assert any(
        diagnostic["code"] == "concurrency_backoff"
        and diagnostic["message"] == "status=503 consecutive=2 effective=2"
        for diagnostic in diagnostics
    )


def test_public_judge_cache_hits_on_identical_typed_requests(tmp_path) -> None:
    calls = 0

    def fake(*_args):
        nonlocal calls
        calls += 1
        return _response()

    preset = runtime_preset({"matches_query": {"type": "noul"}})
    store = CacheStore(tmp_path)
    state = State("state-1", "focus")
    for _ in range(2):
        records = list(
            judge(
                preset,
                [state],
                formation_report=FormationReport(),
                cache_store=store,
                judge_fn=fake,
            )
        )
    result = next(
        record
        for record in records
        if record.to_dict()["record_type"] == "result"
    )
    assert calls == 1
    assert result.to_dict()["meta"]["cache"] == "hit"


def test_emit_stops_without_waiting_for_slow_pending_judgments() -> None:
    release = threading.Event()

    class BrokenStream(io.StringIO):
        def write(self, _value: str) -> int:
            raise BrokenPipeError

    def fake(state, *_args):
        if state.state_ref != "fast":
            release.wait(timeout=2)
        return _response()

    started = time.monotonic()
    try:
        result = emit(
            judge(
                "jgrep",
                [State("fast", "focus"), State("slow", "focus")],
                formation_report=FormationReport(),
                judge_fn=fake,
                concurrency=2,
            ),
            jsonl_stream=BrokenStream(),
            pretty_stream=io.StringIO(),
        )
    finally:
        release.set()
    assert result.broken_pipe is True
    assert time.monotonic() - started < 1


def test_emit_filters_results_but_keeps_terminal_coverage() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = emit(
        judge(
            "jgrep",
            [State("state-1", "focus")],
            formation_report=FormationReport(),
            judge_fn=lambda *_: _response(),
        ),
        format="pretty",
        jsonl_stream=stdout,
        pretty_stream=stderr,
        result_filter=ResultFilter(
            kind="keep", question_id="matches_query", operator=">=", threshold=1
        ),
    )
    assert result.records_suppressed == 1
    assert result.coverage is not None
    assert [line for line in stdout.getvalue().splitlines()] == [
        '{"coverage":"complete","coverage_counts":{"discovered":1,"emitted":1,"failed":0,"judged":1,"skipped":0},"coverage_reasons":[],"meta":{"cache":"not_applicable","chunker":"para","model":"typesafe-ai/jev","preset":"jgrep","preset_version":"1","served_model":"unknown"},"record_type":"coverage"}'
    ]
    assert stderr.getvalue() == ""
