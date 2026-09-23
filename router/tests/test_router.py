import pytest
import requests

import router
from router import RouteResult, build_request, parse_response, route

CATALOG_STUB = {"Read": "read a file", "Bash": "run a command"}

RESPONSE_STUB = {
    "answers": {
        "tool": {
            "type": "choice",
            "choice": "Read",
            "confidence": 0.9,
            "probabilities": {"Read": 0.95, "Bash": 0.05},
        },
        "needs_tool": {"type": "noul", "noul": 0.97},
        "step_clarity": {"type": "noul", "noul": 0.88},
    },
    "usage": {"input_tokens": 400, "output_tokens": 60},
}


def test_build_request_shape():
    body = build_request("fix bug", "read config.py", ["opened repo"], CATALOG_STUB)
    assert "model" not in body
    assert body["state"] == {
        "task": "fix bug",
        "current_step": "read config.py",
        "recent_steps": ["opened repo"],
    }
    q = body["questions"]
    assert q["tool"]["type"] == "choice"
    assert q["tool"]["criteria"] == CATALOG_STUB
    assert q["needs_tool"]["type"] == "noul"
    assert q["step_clarity"]["type"] == "noul"


def test_build_request_limits_history_to_last_five_entries():
    history = [f"step {i}" for i in range(6)]

    body = build_request("fix bug", "read config.py", history, CATALOG_STUB)

    assert body["state"]["recent_steps"] == [
        "step 1", "step 2", "step 3", "step 4", "step 5"
    ]


def test_parse_response():
    r = parse_response(RESPONSE_STUB)
    assert isinstance(r, RouteResult)
    assert r.tool == "Read"
    assert r.probabilities == {"Read": 0.95, "Bash": 0.05}
    assert r.confidence == 0.9
    assert r.needs_tool == 0.97
    assert r.step_clarity == 0.88
    assert r.usage == {"input_tokens": 400, "output_tokens": 60}


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


def test_route_success(monkeypatch):
    sess = FakeSession([FakeResponse(200, RESPONSE_STUB)])
    r = route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
    assert r.tool == "Read"
    assert sess.calls[0]["headers"]["Authorization"] == "Bearer k"
    assert sess.calls[0]["headers"] == {
        "Authorization": "Bearer k",
        **router.GATEWAY_HEADERS,
    }
    assert sess.calls[0]["json"]["questions"]["needs_tool"]["type"] == "boolean"
    assert sess.calls[0]["json"]["providerOptions"] == {
        "gateway": {"zeroDataRetention": True}
    }
    assert sess.calls[0]["json"]["questions"]["tool"]["criteria"] == CATALOG_STUB


def test_route_maps_gateway_booleans_and_derives_missing_confidence():
    response = {
        "answers": {
            "tool": {
                "type": "choice",
                "choice": "Read",
                "probabilities": {"Read": 0.8, "Bash": 0.2},
            },
            "needs_tool": {"type": "boolean", "probability": 0.75},
            "step_clarity": {"type": "boolean", "probability": 0.9},
        },
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
    sess = FakeSession([FakeResponse(200, response)])

    result = route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)

    assert result.needs_tool == 0.75
    assert result.step_clarity == 0.9
    assert result.confidence == pytest.approx(0.6)


def test_route_retries_on_429_then_succeeds(monkeypatch):
    delays = []
    monkeypatch.setattr(router.time, "sleep", delays.append)
    sess = FakeSession([FakeResponse(429), FakeResponse(200, RESPONSE_STUB)])
    r = route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
    assert r.tool == "Read"
    assert len(sess.calls) == 2
    assert delays == [1.0]


def test_route_gives_up_after_three_attempts(monkeypatch):
    delays = []
    monkeypatch.setattr(router.time, "sleep", delays.append)
    sess = FakeSession([FakeResponse(529)] * 3)
    try:
        route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
        raise AssertionError("expected HTTPError")
    except requests.HTTPError:
        pass
    assert len(sess.calls) == 3
    assert delays == [1.0, 2.0]


def test_route_does_not_retry_client_errors(monkeypatch):
    sess = FakeSession([FakeResponse(422)])
    try:
        route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
        raise AssertionError("expected HTTPError")
    except requests.HTTPError:
        pass
    assert len(sess.calls) == 1
