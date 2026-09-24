from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path

import httpx
import pytest

from jm._transport import (
    _MAX_WAIT_SECONDS,
    _GatewayTransport,
    _resolve_gateway_key,
)
from jm.answers import NoulAnswer
from jm.client import JevClient, JevError
from jm.runner import State

QUESTIONS = {
    "is_match": {"type": "noul"},
    "kind": {"type": "choice", "options": ["billing", "technical", "other"]},
    "risk": {
        "type": "score",
        "criteria": [
            {"level": "low"},
            {"level": "medium"},
            {"level": "high"},
        ],
    },
}


def _response(request: httpx.Request, *, status: int = 200, **payload: object):
    return httpx.Response(status, json=payload, request=request)


def _client(**kwargs: object) -> JevClient:
    return JevClient(_transport=_GatewayTransport(**kwargs))


def test_client_sends_exact_gateway_request_and_normalizes_answers(monkeypatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _response(
            request,
            answers={
                "is_match": {"type": "boolean", "probability": 0.82},
                "kind": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {
                        "billing": 0.8,
                        "technical": 0.1,
                        "other": 0.1,
                    },
                },
                "risk": {
                    "type": "score",
                    "score": 1.7,
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                },
            },
            usage={"inputTokens": 120, "outputTokens": 20, "futureField": 3},
            providerMetadata={
                "gateway": {"routing": {"canonicalSlug": "jev-1.13.0"}}
            },
        )

    monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-secret")
    client = _client(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    try:
        response = client.evaluate(
            State("case#1", "focus", {"file": "notes.md"}), QUESTIONS
        )
    finally:
        client.close()

    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
    assert request.headers["authorization"] == "Bearer test-secret"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept-encoding"] == "identity"
    assert request.headers["ai-evaluation-model-specification-version"] == "4"
    assert request.headers["ai-gateway-auth-method"] == "api-key"
    assert request.headers["ai-gateway-protocol-version"] == "0.0.1"
    assert request.headers["ai-model-id"] == "typesafe-ai/jev"
    body = json.loads(request.content)
    assert body == {
        "providerOptions": {"gateway": {"zeroDataRetention": True}},
        "state": {
            "focus": "focus",
            "context": {"file": "notes.md", "state_ref": "case#1"},
        },
        "questions": {
            "is_match": {"type": "boolean"},
            "kind": {
                "type": "choice",
                "options": ["billing", "technical", "other"],
            },
            "risk": {
                "type": "score",
                "criteria": [
                    {"level": "low"},
                    {"level": "medium"},
                    {"level": "high"},
                ],
            },
        },
    }
    assert response.answers["is_match"] == NoulAnswer(0.82)
    assert response.answers["kind"].probabilities == {
        "billing": 0.8,
        "technical": 0.1,
        "other": 0.1,
    }
    assert response.answers["kind"].confidence == pytest.approx(0.7)
    assert response.answers["risk"].legend == {
        "0": {"level": "low"},
        "1": {"level": "medium"},
        "2": {"level": "high"},
    }
    assert response.answers["risk"].confidence == pytest.approx(0.7)
    assert response.usage == {
        "input_tokens": 120,
        "output_tokens": 20,
        "futureField": 3,
    }
    assert response.served_model == "jev-1.13.0"


def test_client_sends_a_configured_model_id(monkeypatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _response(request, answers={})

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = _client(
        http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    try:
        client(State("case#1", "focus"), {}, "jev-custom")
    finally:
        client.close()

    assert seen[0].headers["ai-model-id"] == "jev-custom"


@pytest.mark.parametrize(
    "probabilities",
    [
        {"billing": 0.8, "technical": 0.2},
        {"billing": 0.8, "technical": 0.1, "other": 0.1, "unexpected": 0.0},
    ],
)
def test_client_rejects_malformed_probability_maps(
    monkeypatch, probabilities: dict[str, float]
) -> None:
    monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-secret")
    client = _client(
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _response(
                    request,
                    answers={
                        "kind": {
                            "type": "choice",
                            "choice": "billing",
                            "probabilities": probabilities,
                        }
                    },
                )
            )
        )
    )
    try:
        with pytest.raises(JevError, match="malformed answer"):
            client.evaluate(
                State("case#1", "focus"),
                {"kind": QUESTIONS["kind"]},
            )
    finally:
        client.close()


def test_client_rejects_gateway_noul_shape_in_public_answer_direction(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    transport = httpx.MockTransport(
        lambda request: _response(
            request,
            answers={"is_match": {"type": "noul", "noul": 0.5}},
        )
    )
    client = _client(http_client=httpx.Client(transport=transport))
    try:
        with pytest.raises(JevError, match="malformed answer"):
            client.evaluate(State("case#1", "focus"), {"is_match": {"type": "noul"}})
    finally:
        client.close()


def test_client_does_not_leak_malformed_response_data(monkeypatch) -> None:
    gateway_key = "gw_live_response_key_123"
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = _client(
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _response(
                    request,
                    answers={gateway_key: {"type": "noul", "noul": 0.5}},
                )
            )
        )
    )
    try:
        with pytest.raises(JevError) as error:
            client.evaluate(State("case#1", "focus"), {"is_match": {"type": "noul"}})
    finally:
        client.close()

    assert gateway_key not in error.value.message
    assert gateway_key not in repr(error.value.args)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    formatted = "".join(traceback.format_exception(error.value))
    assert gateway_key not in formatted


@pytest.mark.parametrize(
    "environment_name",
    ["VERCEL_AI_GATEWAY", "AI_GATEWAY_API_KEY", "VERCEL_JEV_KEY"],
)
def test_each_environment_key_beats_zshrc(
    monkeypatch, tmp_path: Path, environment_name: str
) -> None:
    zshrc = tmp_path / ".zshrc"
    zshrc.write_text(
        "export VERCEL_AI_GATEWAY=from-file-vercel\n"
        "AI_GATEWAY_API_KEY=from-file-api\n"
        "VERCEL_JEV_KEY=from-file-jev\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("VERCEL_AI_GATEWAY", "AI_GATEWAY_API_KEY", "VERCEL_JEV_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(environment_name, f"from-env-{environment_name}")
    assert _resolve_gateway_key() == f"from-env-{environment_name}"


def test_environment_key_precedence_is_ordered(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".zshrc").write_text(
        "VERCEL_AI_GATEWAY=from-file-vercel\n"
        "AI_GATEWAY_API_KEY=from-file-api\n"
        "VERCEL_JEV_KEY=from-file-jev\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("VERCEL_AI_GATEWAY", "first")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "second")
    monkeypatch.setenv("VERCEL_JEV_KEY", "third")
    assert _resolve_gateway_key() == "first"


@pytest.mark.parametrize("status", [429, 529])
def test_client_retries_only_retryable_statuses_with_finite_hint(
    monkeypatch, status: int
) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(
                status, headers={"Retry-After": "59"}, request=request
            )
        return _response(request, answers={})

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = _client(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        jitter=lambda: 99.0,
    )
    try:
        assert client.evaluate(State("case#1", "focus"), {}).complete
    finally:
        client.close()
    assert attempts == 3
    assert sleeps == [59.0, 59.0]


@pytest.mark.parametrize("status", [503, 504])
def test_client_retries_503_and_504_without_retry_after_with_bounded_backoff(
    monkeypatch, status: int
) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(status, request=request)
        return _response(request, answers={})

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = _client(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        jitter=lambda: 0.25,
        backoff_base=1.0,
    )
    try:
        assert client.evaluate(State("case#1", "focus"), {}).complete
    finally:
        client.close()

    assert attempts == 3
    assert sleeps == [1.25, 2.25]


def test_request_observer_runs_before_every_transport_attempt(monkeypatch) -> None:
    attempts = 0
    observed: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, request=request)
        return _response(request, answers={})

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = _client(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
        jitter=lambda: 0.0,
    )
    client.set_request_observer(lambda: observed.append(attempts + 1))
    try:
        assert client.evaluate(State("case#1", "focus"), {}).complete
    finally:
        client.close()

    assert observed == [1, 2, 3]


def test_async_request_observer_runs_before_every_transport_attempt(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")

    async def run() -> None:
        attempts = 0
        observed: list[int] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return httpx.Response(503, request=request)
            return _response(request, answers={})

        async def async_sleep(_: float) -> None:
            return None

        client = _client(
            async_http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ),
            async_sleep=async_sleep,
            jitter=lambda: 0.0,
        )
        client.set_request_observer(lambda: observed.append(attempts + 1))
        try:
            assert (await client.evaluate_async(State("case#1", "focus"), {})).complete
        finally:
            await client.aclose()

        assert observed == [1, 2, 3]

    asyncio.run(run())


def test_request_observer_exception_propagates_before_http_send(monkeypatch) -> None:
    sent = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        return _response(request, answers={})

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = _client(
        http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    def observer() -> None:
        raise RuntimeError("observer failed")

    client.set_request_observer(observer)
    try:
        with pytest.raises(RuntimeError, match="observer failed"):
            client.evaluate(State("case#1", "focus"), {})
    finally:
        client.close()

    assert sent == 0


def test_async_request_observer_exception_propagates_before_http_send(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")

    async def run() -> None:
        sent = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal sent
            sent += 1
            return _response(request, answers={})

        client = _client(
            async_http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
        )

        def observer() -> None:
            raise RuntimeError("observer failed")

        client.set_request_observer(observer)
        try:
            with pytest.raises(RuntimeError, match="observer failed"):
                await client.evaluate_async(State("case#1", "focus"), {})
        finally:
            await client.aclose()

        assert sent == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {
                "gateway": {
                    "routing": {"canonicalSlug": "routed"},
                    "modelAttempts": [{"canonicalSlug": "attempt"}],
                }
            },
            "routed",
        ),
        (
            {
                "gateway": {
                    "modelAttempts": [
                        {"canonicalSlug": "failed", "status": "error"},
                        {"canonicalSlug": "successful", "status": "success"},
                    ]
                }
            },
            "successful",
        ),
        (
            {"gateway": {"modelAttempts": [{"canonicalSlug": "failed"}]}},
            "failed",
        ),
        ({"gateway": {"model": "alias"}}, None),
        ({}, None),
    ],
)
def test_served_model_uses_locked_gateway_precedence(metadata, expected) -> None:
    from jm.client import _served_model

    assert _served_model({"providerMetadata": metadata}) == expected


def test_client_bounds_absurd_retry_after_and_never_leaks_key(monkeypatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "1e100"},
            text="test-secret",
            request=request,
        )

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = _client(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        jitter=lambda: 0.0,
    )
    try:
        with pytest.raises(JevError) as error:
            client.evaluate(State("case#1", "focus"), {})
    finally:
        client.close()
    assert attempts == 3
    assert sleeps == [_MAX_WAIT_SECONDS, _MAX_WAIT_SECONDS]
    assert "test-secret" not in str(error.value)
    assert error.value.attempts == 3
    assert error.value.http_status == 429


def test_sync_and_async_methods_return_the_same_contract(monkeypatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    payload = {"answers": {"is_match": {"type": "boolean", "probability": 0.7}}}

    def sync_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async def run() -> None:
        async def async_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload, request=request)

        sync_client = _client(
            http_client=httpx.Client(transport=httpx.MockTransport(sync_handler))
        )
        async_client = _client(
            async_http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(async_handler)
            )
        )
        try:
            sync_response = sync_client.evaluate(
                State("case#1", "focus"), {"is_match": {"type": "noul"}}
            )
            async_response = await async_client.evaluate_async(
                State("case#1", "focus"), {"is_match": {"type": "noul"}}
            )
            assert async_response == sync_response
        finally:
            sync_client.close()
            await async_client.aclose()

    asyncio.run(run())
