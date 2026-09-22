from __future__ import annotations

import json

import httpx
import pytest

from jmap.answers import ChoiceAnswer, ErrorResponse, NoulAnswer, ScoreAnswer
from jmap.api import TypeSafeClient
from jmap.runner import (
    FakeJudge,
    Runner,
    State,
    StateAdmission,
    StateRejection,
    admit_states,
)

QUESTIONS = {
    "is_relevant": {"type": "noul"},
    "kind": {"type": "choice"},
    "risk": {"type": "score"},
}


def test_fake_judge_is_injected_without_http() -> None:
    state = State("docs/guide.md#P1", "the focus", {"source": "docs/guide.md"})
    calls = []

    def fake(state_arg, questions_arg, model_arg):
        calls.append((state_arg, questions_arg, model_arg))
        return "answer"

    runner = Runner(judge_fn=fake, model="jev-1.13.0")
    assert runner.judge(state, QUESTIONS) == "answer"
    assert calls == [(state, QUESTIONS, "jev-1.13.0")]


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
