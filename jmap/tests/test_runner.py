from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest

from jmap.answers import (
    ChoiceAnswer,
    ErrorResponse,
    JudgeResponse,
    NoulAnswer,
    ScoreAnswer,
)
from jmap.api import MAX_RESPONSE_BYTES, MAX_WAIT_SECONDS, TypeSafeClient
from jmap.presets import PresetUsageError, PresetValidationError, resolve_preset
from jmap.runner import (
    FakeJudge,
    Runner,
    State,
    StateAdmission,
    StateLimits,
    StateRejection,
    admit_states,
)

QUESTIONS = {
    "is_relevant": {"type": "noul"},
    "kind": {"type": "choice"},
    "risk": {"type": "score"},
}


def _complete_payload() -> dict[str, object]:
    return {"answers": {"is_relevant": {"type": "noul", "noul": 0.9}}}


def test_runner_requires_an_explicit_judge_function() -> None:
    with pytest.raises(TypeError, match="judge_fn"):
        Runner(None)


def test_fake_judge_is_injected_without_http() -> None:
    state = State("docs/guide.md#P1", "the focus", {"source": "docs/guide.md"})
    calls = []

    def fake(state_arg, questions_arg, model_arg):
        calls.append((state_arg, questions_arg, model_arg))
        return "answer"

    runner = Runner(judge_fn=fake, model="jev-1.13.0")
    assert runner.judge(state, QUESTIONS) == "answer"
    assert calls == [(state, QUESTIONS, "jev-1.13.0")]


def test_runner_uses_one_validated_preset_for_runtime_values() -> None:
    calls = []
    preset = resolve_preset("jgrep")

    def judge(state_arg, questions_arg, model_arg):
        calls.append((state_arg, questions_arg, model_arg))
        return FakeJudge()(state_arg, questions_arg, model_arg)

    result = Runner(judge).run(
        [State("stdin#L1", "launch")],
        preset=preset,
        chunker="file",
    )

    assert calls[0][1] == preset.questions
    assert calls[0][2] == preset.model
    assert result.records[0].to_dict()["meta"] == {
        "preset": preset.name,
        "preset_version": preset.version,
        "model": preset.model,
        "chunker": "file",
        "cache": "not_applicable",
    }


def test_runner_rejects_questions_with_a_preset() -> None:
    calls = []

    def judge(*args):
        calls.append(args)
        return FakeJudge()(*args)

    with pytest.raises(PresetUsageError, match="questions"):
        Runner(judge).run(
            [State("stdin#L1", "launch")],
            {"matches": {"type": "noul"}},
            preset="jgrep",
        )

    assert calls == []


@pytest.mark.parametrize(
    ("runner_kwargs", "run_kwargs", "message"),
    [
        ({"model": "jev-9.9.9"}, {}, "model"),
        ({"limits": StateLimits(focus_bytes=1)}, {}, "limits"),
        ({}, {"preset_version": "2"}, "preset_version"),
        ({}, {"max_chunks": 1}, "max_chunks"),
        ({}, {"chunker": "record"}, "incompatible"),
        ({}, {"chunking": {"by": "line"}}, "chunking"),
    ],
)
def test_runner_rejects_conflicting_loose_preset_values(
    runner_kwargs, run_kwargs, message
) -> None:
    with pytest.raises(PresetUsageError, match=message):
        Runner(FakeJudge(), **runner_kwargs).run(
            [State("stdin#L1", "launch")], preset="jgrep", **run_kwargs
        )


def test_invalid_preset_is_validated_before_processing(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yml"
    path.write_text("schema: jmap.preset/v1\n", encoding="utf-8")
    calls = []

    def judge(*args):
        calls.append(args)
        return FakeJudge()(*args)

    with pytest.raises(PresetValidationError):
        Runner(judge).run(
            [State("stdin#L1", "launch")],
            preset=path,
        )

    assert calls == []


def test_fake_judge_returns_deterministic_typed_answers() -> None:
    state = State("stdin#L1", "launch", {"source": "stdin"})
    fake = FakeJudge()

    first = fake(state, QUESTIONS, "jev-1.13.0")
    second = fake(state, QUESTIONS, "jev-1.13.0")

    assert first == second
    assert isinstance(first.answers["is_relevant"], NoulAnswer)
    assert isinstance(first.answers["kind"], ChoiceAnswer)
    assert isinstance(first.answers["risk"], ScoreAnswer)


def test_fake_judge_can_return_incomplete_answers() -> None:
    state = State("stdin#L1", "launch", {"source": "stdin"})
    response = FakeJudge(mode="incomplete")(state, QUESTIONS, "jev-1.13.0")

    assert response.complete is False
    assert response.missing_questions == ("risk",)
    assert "risk" not in response.answers


def test_fake_judge_can_return_an_operational_error() -> None:
    state = State("stdin#L1", "launch", {"source": "stdin"})
    first = FakeJudge(mode="error")(state, QUESTIONS, "jev-1.13.0")
    second = FakeJudge(mode="error")(state, QUESTIONS, "jev-1.13.0")

    assert first == second == ErrorResponse("fake operational error")


def test_runner_admits_states_in_input_order_and_keeps_skipped_refs() -> None:
    states = [State(f"stdin#L{i}", str(i), {"line": i}) for i in range(1, 4)]
    admission = Runner(judge_fn=lambda *_: None).admit(states, max_chunks=2)

    assert isinstance(admission, StateAdmission)
    assert admission.discovered == 3
    assert [state.state_ref for state in admission.admitted] == ["stdin#L1", "stdin#L2"]
    assert [state.state_ref for state in admission.skipped] == ["stdin#L3"]
    assert admission.skip_boundary == "max_chunks=2"


def test_runner_emits_jsonl_then_terminal_coverage_and_flushes_each_record() -> None:
    class FlushCapture(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    stdout = FlushCapture()
    result = Runner(FakeJudge()).run_jsonl(
        [State("stdin#L1", "launch")],
        stdout=stdout,
        preset="jgrep",
        chunker="para",
    )
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert [line["record_type"] for line in lines] == ["result", "coverage"]
    assert lines[-1]["coverage"] == "complete"
    assert result.exit_code == 0
    assert stdout.flush_count == 2


def test_run_jsonl_rejects_questions_with_a_preset() -> None:
    with pytest.raises(PresetUsageError, match="questions"):
        Runner(FakeJudge()).run_jsonl(
            [State("stdin#L1", "launch")],
            {"matches": {"type": "noul"}},
            io.StringIO(),
            preset="jgrep",
        )


def test_runner_emits_partial_result_and_operational_exit() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = Runner(FakeJudge(mode="incomplete")).run(
        [State("stdin#L1", "launch")],
        {"matches": {"type": "noul"}, "risk": {"type": "score"}},
        stdout=stdout,
        stderr=stderr,
        chunker="para",
    )
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert [line["record_type"] for line in lines] == ["partial_result", "coverage"]
    assert lines[0]["missing_questions"] == ["risk"]
    assert lines[0]["meta"]["partial"] is True
    assert lines[1]["coverage_reasons"] == ["partial_answer"]
    assert lines[1]["coverage"] == "partial"
    assert lines[1]["coverage_counts"]["failed"] == 1
    assert result.exit_code == 2
    assert "partial" in stderr.getvalue()


def test_runner_emits_exact_partial_json() -> None:
    def judge(*_):
        return JudgeResponse({"matches": NoulAnswer(0.5)}, ("risk",))

    stdout = io.StringIO()
    result = Runner(judge).run(
        [State("stdin#L1", "launch")],
        {"matches": {"type": "noul"}, "risk": {"type": "score"}},
        stdout=stdout,
        chunker="para",
    )

    assert [json.loads(line) for line in stdout.getvalue().splitlines()] == [
        {
            "record_type": "partial_result",
            "state_ref": "stdin#L1",
            "answers": {"matches": {"type": "noul", "noul": 0.5}},
            "missing_questions": ["risk"],
            "meta": {
                "preset": "jmap",
                "preset_version": "1",
                "model": "jev-1.13.0",
                "chunker": "para",
                "cache": "not_applicable",
                "partial": True,
            },
        },
        {
            "record_type": "coverage",
            "coverage": "partial",
            "coverage_counts": {
                "discovered": 1,
                "judged": 1,
                "emitted": 1,
                "skipped": 0,
                "failed": 1,
            },
            "coverage_reasons": ["partial_answer"],
            "meta": {
                "preset": "jmap",
                "preset_version": "1",
                "model": "jev-1.13.0",
                "chunker": "para",
                "cache": "not_applicable",
            },
        },
    ]
    assert result.exit_code == 2


def test_runner_groups_cap_skips_and_keeps_eight_samples() -> None:
    states = [State(f"notes:paragraph={index}", str(index)) for index in range(11)]
    stdout = io.StringIO()
    result = Runner(FakeJudge()).run(
        states,
        {"matches": {"type": "noul"}},
        max_chunks=2,
        stdout=stdout,
        chunker="para",
    )
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    skip = next(line for line in lines if line["record_type"] == "error")
    summary = skip["error"]["skip_summary"]

    assert skip == {
        "record_type": "error",
        "state_ref": None,
        "source_ref": None,
        "error": {
            "kind": "scan_cap",
            "message": "scan cap reached before visit",
            "http_status": None,
            "attempts": 0,
            "skip_summary": {
                "boundary": "max_chunks=2",
                "count": 9,
                "sample_refs": [
                    f"notes:paragraph={index}" for index in range(2, 10)
                ],
            },
        },
        "meta": {
            "preset": "jmap",
            "preset_version": "1",
            "model": "jev-1.13.0",
            "chunker": "para",
            "cache": "not_applicable",
        },
    }
    assert summary["boundary"] == "max_chunks=2"
    assert summary["count"] == 9
    assert summary["sample_refs"] == [
        f"notes:paragraph={index}" for index in range(2, 10)
    ]
    assert lines[-1]["coverage_counts"] == {
        "discovered": 11,
        "judged": 2,
        "emitted": 2,
        "skipped": 9,
        "failed": 0,
    }
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "mode,max_chunks,rejections,expected_reasons",
    [
        ("complete", None, (), ()),
        ("complete", 1, (), ("scan_cap",)),
        (
            "complete",
            None,
            (
                StateRejection(
                    None, "input_error", "invalid JSON", "stdin:byte=0,line=1"
                ),
            ),
            ("input_error",),
        ),
        (
            "complete",
            None,
            (StateRejection("stdin#L3", "scan_cap", "cap reached"),),
            ("scan_cap",),
        ),
        (
            "complete",
            None,
            (StateRejection("stdin#L3", "context_limit", "too large"),),
            ("context_limit",),
        ),
        ("error", None, (), ("api_error",)),
        ("incomplete", None, (), ("partial_answer",)),
    ],
)
def test_coverage_equations_hold_for_each_run_path(
    mode, max_chunks, rejections, expected_reasons
) -> None:
    states = [State("stdin#L1", "one"), State("stdin#L2", "two")]
    result = Runner(FakeJudge(mode=mode)).run(
        states,
        {"matches": {"type": "noul"}, "risk": {"type": "score"}},
        max_chunks=max_chunks,
        rejections=rejections,
    )
    coverage = result.records[-1].to_dict()
    counts = coverage["coverage_counts"]
    skip_count = sum(
        record.to_dict()["error"]["skip_summary"]["count"]
        for record in result.records
        if record.to_dict().get("record_type") == "error"
        and "skip_summary" in record.to_dict()["error"]
    )

    assert counts["discovered"] == counts["judged"] + counts["skipped"]
    assert counts["skipped"] == skip_count
    assert counts["failed"] <= counts["judged"]
    assert tuple(coverage["coverage_reasons"]) == expected_reasons
    assert result.exit_code == (2 if expected_reasons else 0)


def test_runner_empty_input_emits_input_error_and_partial_coverage() -> None:
    stdout = io.StringIO()
    result = Runner(FakeJudge()).run(
        [],
        {"matches": {"type": "noul"}},
        stdout=stdout,
    )

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines == [
        {
            "record_type": "error",
            "state_ref": None,
            "source_ref": "stdin:byte=0,line=1",
            "error": {
                "kind": "input_error",
                "message": "input is empty",
                "http_status": None,
                "attempts": 0,
            },
            "meta": {
                "preset": "jmap",
                "preset_version": "1",
                "model": "jev-1.13.0",
                "chunker": "unknown",
                "cache": "not_applicable",
            },
        },
        {
            "record_type": "coverage",
            "coverage": "partial",
            "coverage_counts": {
                "discovered": 0,
                "judged": 0,
                "emitted": 0,
                "skipped": 0,
                "failed": 0,
            },
            "coverage_reasons": ["input_error"],
            "meta": {
                "preset": "jmap",
                "preset_version": "1",
                "model": "jev-1.13.0",
                "chunker": "unknown",
                "cache": "not_applicable",
            },
        },
    ]
    assert result.exit_code == 2


def test_runner_keeps_interleaved_input_errors_in_input_order() -> None:
    rejections = (
        StateRejection(None, "input_error", "bad first", "stdin:byte=0,line=1"),
        StateRejection("stdin#L2", "context_limit", "too large"),
        StateRejection(None, "input_error", "bad third", "stdin:byte=2,line=3"),
    )
    result = Runner(FakeJudge()).run(
        [],
        {"matches": {"type": "noul"}},
        rejections=rejections,
    )

    rejection_records = [record.to_dict() for record in result.records[:-1]]
    assert [
        (record["error"]["kind"], record.get("source_ref"))
        for record in rejection_records
    ] == [
        ("input_error", "stdin:byte=0,line=1"),
        ("context_limit", None),
        ("input_error", "stdin:byte=2,line=3"),
    ]


def test_runner_keeps_jsonl_on_stdout_and_human_warnings_on_stderr() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    Runner(FakeJudge()).run(
        [State("stdin#L1", "one"), State("stdin#L2", "two")],
        {"matches": {"type": "noul"}},
        max_chunks=1,
        stdout=stdout,
        stderr=stderr,
        chunker="para",
    )

    assert all(
        json.loads(line)["record_type"] for line in stdout.getvalue().splitlines()
    )
    assert "warning" in stderr.getvalue()
    assert not any(line.startswith("{") for line in stderr.getvalue().splitlines())


def test_runner_pretty_output_and_filter_do_not_hide_errors_or_coverage() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    Runner(FakeJudge()).run(
        [State("stdin#L1", "one")],
        {"matches": {"type": "noul"}},
        stdout=stdout,
        stderr=stderr,
        output_format="pretty",
        result_filter=lambda record: False,
        chunker="para",
    )

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [line["record_type"] for line in lines] == ["coverage"]
    assert stderr.getvalue() == ""


def test_state_admission_carries_rejections_in_coverage_counts() -> None:
    rejection = StateRejection("stdin#L2", "context_limit", "too large")
    admission = admit_states(
        (State("stdin#L1", "one"),),
        rejections=(rejection,),
    )
    assert admission.discovered == 2
    assert admission.judged == 1
    assert admission.skipped_count == 1
    assert admission.rejections == (rejection,)


def test_typesafe_client_sends_one_full_battery_request(monkeypatch) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "answers": {
                    "is_relevant": {"type": "noul", "noul": 0.9},
                    "kind": {
                        "type": "choice",
                        "choice": "code",
                        "probabilities": {"code": 1.0},
                        "confidence": 0.9,
                    },
                    "risk": {
                        "type": "score",
                        "score": 2,
                        "legend": {"0": "low"},
                        "probabilities": {"2": 1.0},
                        "confidence": 0.8,
                    },
                }
            },
            request=request,
        )

    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = TypeSafeClient(http_client=client)(
        State("stdin#L1", "focus", {"source": "stdin"}), QUESTIONS, "jev-1.13.0"
    )

    assert response.complete
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer test-secret"
    assert json.loads(request.content) == {
        "state": {
            "focus": "focus",
            "context": {"source": "stdin", "state_ref": "stdin#L1"},
        },
        "model": "jev-1.13.0",
        "questions": QUESTIONS,
    }


def test_typesafe_client_retries_timeout_then_succeeds(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=_complete_payload(), request=request)

    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = TypeSafeClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), {"is_relevant": {"type": "noul"}}, "jev-1.13.0"
    )

    assert response.complete
    assert attempts == 2


def test_typesafe_client_rejects_malformed_success_response(monkeypatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="not json", request=request)
        )
    )

    response = TypeSafeClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), {}, "jev-1.13.0"
    )

    assert response == ErrorResponse(
        "malformed answer", http_status=200, attempts=1
    )


def test_typesafe_client_retries_retryable_statuses_and_timeout(
    monkeypatch,
) -> None:
    for status_or_timeout in (429, 529, "timeout"):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if status_or_timeout == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(status_or_timeout, request=request)

        sleeps: list[float] = []
        monkeypatch.setenv("JEV_API_KEY", "test-secret")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        response = TypeSafeClient(
            http_client=client, sleep=sleeps.append, jitter=lambda: 0.0
        )(
            State("stdin#L1", "focus"), QUESTIONS, "jev-1.13.0"
        )

        assert attempts == 3
        assert len(sleeps) == 2
        assert response.complete is False
        assert response.http_status in {None, status_or_timeout}


def test_typesafe_client_does_not_retry_auth_or_validation_status(monkeypatch) -> None:
    for status in (401, 422):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(status, request=request)

        monkeypatch.setenv("JEV_API_KEY", "test-secret")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        response = TypeSafeClient(http_client=client, sleep=lambda _: None)(
            State("stdin#L1", "focus"), QUESTIONS, "jev-1.13.0"
        )

        assert attempts == 1
        assert response.http_status == status


def test_typesafe_client_honors_retry_after(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429, headers={"Retry-After": "7"}, request=request
            )
        return httpx.Response(200, json={"answers": {}}, request=request)

    sleeps: list[float] = []
    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = TypeSafeClient(
        http_client=client, sleep=sleeps.append, jitter=lambda: 99.0
    )(
        State("stdin#L1", "focus"), {}, "jev-1.13.0"
    )

    assert response.complete
    assert sleeps == [7.0]


def test_typesafe_client_clamps_large_retry_after(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429, headers={"Retry-After": "1e100"}, request=request
            )
        return httpx.Response(200, json={"answers": {}}, request=request)

    sleeps: list[float] = []
    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = TypeSafeClient(
        http_client=client, sleep=sleeps.append, jitter=lambda: 1e100
    )(State("stdin#L1", "focus"), {}, "jev-1.13.0")

    assert response.complete
    assert sleeps == [MAX_WAIT_SECONDS]


def test_typesafe_client_retries_an_incomplete_full_battery(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        answers = {"is_relevant": {"type": "noul", "noul": 0.5}}
        if attempts == 2:
            answers["kind"] = {
                "type": "choice",
                "choice": "code",
                "probabilities": {"code": 1.0},
                "confidence": 1.0,
            }
        return httpx.Response(200, json={"answers": answers}, request=request)

    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = TypeSafeClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"),
        {"is_relevant": {"type": "noul"}, "kind": {"type": "choice"}},
        "jev-1.13.0",
    )

    assert attempts == 2
    assert response.complete


def test_typesafe_client_shares_attempt_budget_across_retries(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=request)
        return httpx.Response(
            200,
            json={"answers": {"is_relevant": {"type": "noul", "noul": 0.5}}},
            request=request,
        )

    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = TypeSafeClient(
        http_client=client, max_attempts=99, sleep=lambda _: None
    )(
        State("stdin#L1", "focus"),
        {"is_relevant": {"type": "noul"}, "kind": {"type": "choice"}},
        "jev-1.13.0",
    )

    assert attempts == 3
    assert response.missing_questions == ("kind",)


def test_typesafe_client_enforces_timeout_on_injected_client(monkeypatch) -> None:
    seen_timeouts: list[dict[str, float | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"answers": {}}, request=request)

    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(timeout=None, transport=httpx.MockTransport(handler))
    response = TypeSafeClient(http_client=client, timeout=2.5)(
        State("stdin#L1", "focus"), {}, "jev-1.13.0"
    )

    assert response.complete
    assert seen_timeouts == [
        {"connect": 2.5, "read": 2.5, "write": 2.5, "pool": 2.5}
    ]


def test_typesafe_client_rejects_oversized_response(monkeypatch) -> None:
    class OversizedStream(httpx.SyncByteStream):
        chunk = b"x" * 4096

        def __init__(self) -> None:
            self.bytes_read = 0

        def __iter__(self):
            while True:
                self.bytes_read += len(self.chunk)
                yield self.chunk

    stream = OversizedStream()

    monkeypatch.setenv("JEV_API_KEY", "test-secret")

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, stream=stream, request=request)
        assert "Content-Length" not in response.headers
        return response

    client = httpx.Client(
        transport=httpx.MockTransport(handler)
    )

    response = TypeSafeClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), {}, "jev-1.13.0"
    )

    assert isinstance(response, ErrorResponse)
    assert response.error == "response too large"
    assert stream.bytes_read <= MAX_RESPONSE_BYTES + len(stream.chunk)


def test_typesafe_client_handles_mid_body_connection_reset(monkeypatch) -> None:
    class ResetStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"answers": '
            raise httpx.ReadError("connection reset")

    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=ResetStream(), request=request)
        )
    )

    response = TypeSafeClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), {}, "jev-1.13.0"
    )

    assert response == ErrorResponse("request failed", attempts=1)


def test_typesafe_client_returns_missing_ids_after_second_incomplete_response(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"answers": {"is_relevant": {"type": "noul", "noul": 0.5}}},
                request=request,
            )
        )
    )
    response = TypeSafeClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"),
        {"is_relevant": {"type": "noul"}, "kind": {"type": "choice"}},
        "jev-1.13.0",
    )

    assert response.missing_questions == ("kind",)
    assert set(response.answers) == {"is_relevant"}


def test_typesafe_client_never_includes_api_key_in_error(monkeypatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", "do-not-leak-this")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                401, text="do-not-leak-this", request=request
            )
        )
    )

    response = TypeSafeClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), QUESTIONS, "jev-1.13.0"
    )

    assert "do-not-leak-this" not in response.error


def test_typesafe_client_rejects_moving_model_name(monkeypatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail("moving model must not make a request")
        )
    )

    response = TypeSafeClient(http_client=client)(
        State("stdin#L1", "focus"), QUESTIONS, "jev-latest"
    )

    assert response == ErrorResponse("model must be a pinned version")
