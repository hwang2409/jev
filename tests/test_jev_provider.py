from __future__ import annotations

from typing import Any, ClassVar, Self

import pytest

from zeta.providers import jev


class Response:
    def __init__(self, status_code: int, data: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._data = data or {}
        self.text = "backend error" if status_code >= 400 else ""

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    def json(self) -> dict[str, Any]:
        return self._data


class Client:
    responses: ClassVar[list[Response]] = []
    requests: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> Response:
        self.requests.append({"url": url, **kwargs})
        return self.responses.pop(0)


def response() -> Response:
    return Response(
        200,
        {
            "answers": {
                "tool": {
                    "choice": "read",
                    "probabilities": {"read": 0.9, "bash": 0.1},
                    "confidence": 0.9,
                },
                "needs_tool": {"noul": 0.99},
                "step_clarity": {"noul": 0.8},
            },
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
    )


@pytest.mark.asyncio
async def test_route_step_builds_the_jev_request(monkeypatch: pytest.MonkeyPatch) -> None:
    Client.responses = [response()]
    Client.requests = []
    monkeypatch.setattr(jev.httpx, "AsyncClient", Client)
    monkeypatch.setenv("JEV_API_KEY", "test-key")

    result = await jev.route_step(
        "read the note",
        {"read": "Read a file", "bash": "Run a command"},
        ["inspect the repo", "find the note"],
    )

    request = Client.requests[0]
    assert request["url"] == jev.API_URL
    assert request["headers"] == {"Authorization": "Bearer test-key"}
    assert request["json"] == {
        "state": {
            "current_step": "read the note",
            "recent_steps": ["inspect the repo", "find the note"],
        },
        "model": "jev-latest",
        "questions": {
            "tool": {
                "type": "choice",
                "instructions": (
                    "An agent is working on the task and describes its current "
                    "step. Which single tool should it call to accomplish this "
                    "step?"
                ),
                "criteria": {"read": "Read a file", "bash": "Run a command"},
            },
            "needs_tool": {
                "type": "noul",
                "instructions": (
                    "Does the current step require calling a tool, rather than "
                    "the agent answering or reasoning directly from what it "
                    "already knows?"
                ),
            },
            "step_clarity": {
                "type": "noul",
                "instructions": (
                    "Is the current step description specific enough to route "
                    "to a single tool with confidence?"
                ),
            },
        },
    }
    assert result.tool == "read"
    assert result.usage == {"input_tokens": 10, "output_tokens": 4}


@pytest.mark.asyncio
async def test_route_step_retries_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    Client.responses = [Response(429), Response(529), response()]
    Client.requests = []
    monkeypatch.setattr(jev.httpx, "AsyncClient", Client)
    monkeypatch.setattr(jev.asyncio, "sleep", lambda _delay: _done())
    monkeypatch.setenv("JEV_API_KEY", "test-key")

    result = await jev.route_step("read it", {"read": "Read"})

    assert result.tool == "read"
    assert len(Client.requests) == 3


async def _done() -> None:
    return None


@pytest.mark.asyncio
async def test_route_step_raises_for_non_retryable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Client.responses = [Response(400)]
    Client.requests = []
    monkeypatch.setattr(jev.httpx, "AsyncClient", Client)
    monkeypatch.setenv("JEV_API_KEY", "test-key")

    with pytest.raises(jev.JevRouterError, match="HTTP 400"):
        await jev.route_step("read it", {"read": "Read"})
