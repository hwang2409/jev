from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from jm.answers import NoulAnswer
from jm.client import (
    GATEWAY_ENDPOINT,
    MAX_WAIT_SECONDS,
    JevClient,
    JevError,
    resolve_gateway_key,
)
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
                    "probabilities": {"billing": 0.8},
                },
                "risk": {
                    "type": "score",
                    "score": 1.7,
                    "probabilities": {"1": 0.8},
                },
            },
            usage={"inputTokens": 120, "outputTokens": 20, "futureField": 3},
            providerMetadata={"typesafe": {"model": "jev-1.13.0"}},
        )

    monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-secret")
    client = JevClient(
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
    assert str(request.url) == GATEWAY_ENDPOINT
    assert request.headers["authorization"] == "Bearer test-secret"
    assert request.headers["accept-encoding"] == "identity"
    body = json.loads(request.content)
    assert "model" not in body
    assert body["providerOptions"] == {"gateway": {"zeroDataRetention": True}}
    assert body["state"]["context"]["state_ref"] == "case#1"
    assert body["questions"]["is_match"]["type"] == "boolean"
    assert body["questions"]["kind"] == QUESTIONS["kind"]
    assert response.answers["is_match"] == NoulAnswer(0.82)
    assert response.answers["kind"].probabilities == {
        "billing": 0.8,
        "technical": 0.0,
        "other": 0.0,
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
    client = JevClient(http_client=httpx.Client(transport=transport))
    try:
        with pytest.raises(JevError, match="malformed answer"):
            client.evaluate(State("case#1", "focus"), {"is_match": {"type": "noul"}})
    finally:
        client.close()


def test_key_precedence_checks_environment_before_zshrc(
    monkeypatch, tmp_path: Path
) -> None:
    zshrc = tmp_path / ".zshrc"
    zshrc.write_text(
        "export VERCEL_AI_GATEWAY=from-file\nAI_GATEWAY_API_KEY=other-file\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "from-env")
    monkeypatch.delenv("VERCEL_AI_GATEWAY", raising=False)
    monkeypatch.delenv("VERCEL_JEV_KEY", raising=False)
    assert resolve_gateway_key() == "from-env"


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
    client = JevClient(
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
    client = JevClient(
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
    assert sleeps == [MAX_WAIT_SECONDS, MAX_WAIT_SECONDS]
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

        sync_client = JevClient(
            http_client=httpx.Client(transport=httpx.MockTransport(sync_handler))
        )
        async_client = JevClient(
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
