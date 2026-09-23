from pathlib import Path

from jm.client import JevError

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


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        return self.response


def test_route_uses_jm_client_and_preserves_request_questions():
    client = FakeClient(RESPONSE_STUB)

    result = route("task", "step", ["old"], CATALOG_STUB, client=client)

    assert result.tool == "Read"
    assert result.needs_tool == 0.97
    assert result.step_clarity == 0.88
    assert result.usage == {"input_tokens": 400, "output_tokens": 60}
    assert client.calls == [
        (
            {
                "task": "task",
                "current_step": "step",
                "recent_steps": ["old"],
            },
            build_request("task", "step", ["old"], CATALOG_STUB)["questions"],
        )
    ]


def test_route_accepts_normalized_jm_response():
    response = {
        "answers": {
            "tool": {
                "choice": "Read",
                "probabilities": {"Read": 0.8, "Bash": 0.2},
                "confidence": 0.6,
            },
            "needs_tool": {"noul": 0.75},
            "step_clarity": {"noul": 0.9},
        },
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }

    result = route("t", "s", catalog=CATALOG_STUB, client=FakeClient(response))

    assert result.tool == "Read"
    assert result.probabilities == {"Read": 0.8, "Bash": 0.2}
    assert result.confidence == 0.6
    assert result.needs_tool == 0.75
    assert result.step_clarity == 0.9


def test_router_has_no_transport_implementation():
    source = (Path(__file__).parents[1] / "router.py").read_text()

    forbidden = (
        "import requests",
        "import httpx",
        "http://",
        "https://",
        "GATEWAY_KEY_NAMES",
        "VERCEL_AI_GATEWAY",
        "AI_GATEWAY_API_KEY",
        "VERCEL_JEV_KEY",
        "for attempt in",
        "while attempts",
        "_gateway_questions",
        "_normalize_gateway_response",
        "_retry_after",
    )
    assert not any(token in source for token in forbidden)


def test_route_keeps_jm_transport_errors():
    class FailingClient:
        def evaluate(self, state, questions):
            raise JevError("gateway unavailable", http_status=503, attempts=3)

    try:
        route("t", "s", catalog=CATALOG_STUB, client=FailingClient())
        raise AssertionError("expected JevError")
    except JevError as error:
        assert str(error) == "gateway unavailable"
        assert error.http_status == 503
        assert error.attempts == 3
