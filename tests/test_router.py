from router import RouteResult, build_request, parse_response

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
