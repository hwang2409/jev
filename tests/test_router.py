from router import RouteResult, build_request, parse_response
import requests

import router
from router import route

CATALOG_STUB = {"Read": "read a file", "Bash": "run a command"}

RESPONSE_STUB = {
    "model": "jev-1.13.0",
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
    assert body["model"] == "jev-latest"
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
    assert sess.calls[0]["json"]["questions"]["tool"]["criteria"] == CATALOG_STUB


def test_route_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    sess = FakeSession([FakeResponse(429), FakeResponse(200, RESPONSE_STUB)])
    r = route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
    assert r.tool == "Read"
    assert len(sess.calls) == 2


def test_route_gives_up_after_three_attempts(monkeypatch):
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    sess = FakeSession([FakeResponse(529)] * 3)
    try:
        route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
        raise AssertionError("expected HTTPError")
    except requests.HTTPError:
        pass
    assert len(sess.calls) == 3


def test_route_does_not_retry_client_errors(monkeypatch):
    sess = FakeSession([FakeResponse(422)])
    try:
        route("t", "s", session=sess, api_key="k", catalog=CATALOG_STUB)
        raise AssertionError("expected HTTPError")
    except requests.HTTPError:
        pass
    assert len(sess.calls) == 1
