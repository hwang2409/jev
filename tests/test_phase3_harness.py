from types import SimpleNamespace

from harness import run_scenario
from router import RouteResult
from sim_exec import execute


def response(stop_reason, content, input_tokens=0, output_tokens=0, cache_read=0):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
        ),
    )


def tool_use(tool_name, tool_use_id, details):
    return {
        "type": "tool_use",
        "id": tool_use_id,
        "name": tool_name,
        "input": {"details": details},
    }


class StubClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.responses)


def scenario(**updates):
    value = {
        "id": "test",
        "task": "complete the test task",
        "results": {},
    }
    value.update(updates)
    return value


def route_result(tool, confidence=0.95, probabilities=None):
    return RouteResult(
        tool=tool,
        probabilities=probabilities or {tool: confidence},
        confidence=confidence,
        needs_tool=1.0,
        step_clarity=1.0,
        usage={},
    )


def test_jev_injects_selected_schema_then_clears_it():
    client = StubClient(
        [
            response("tool_use", [tool_use("route", "r1", "read the document")]),
            response(
                "tool_use",
                [tool_use("files_read_document", "t1", "read notes.txt")],
            ),
            response("end_turn", [{"type": "text", "text": "done"}]),
        ]
    )

    result = run_scenario(
        scenario(),
        "jev",
        client=client,
        route_fn=lambda task, step, catalog: route_result("files_read_document"),
    )

    assert [tool["name"] for tool in client.requests[0]["tools"]] == ["route"]
    assert [tool["name"] for tool in client.requests[1]["tools"]] == [
        "route",
        "files_read_document",
    ]
    assert [tool["name"] for tool in client.requests[2]["tools"]] == ["route"]
    assert result["tool_calls"] == ["files_read_document"]


def test_jev_expands_top_three_tools_below_point_eight():
    client = StubClient(
        [
            response("tool_use", [tool_use("route", "r1", "choose a file tool")]),
            response("end_turn", [{"type": "text", "text": "done"}]),
        ]
    )
    probabilities = {
        "files_read_document": 0.7,
        "files_search_content": 0.2,
        "files_get_metadata": 0.1,
        "shell_run_command": 0.0,
    }

    result = run_scenario(
        scenario(),
        "jev",
        client=client,
        route_fn=lambda task, step, catalog: route_result(
            "files_read_document", confidence=0.7, probabilities=probabilities
        ),
    )

    assert [tool["name"] for tool in client.requests[1]["tools"]] == [
        "route",
        "files_read_document",
        "files_search_content",
        "files_get_metadata",
    ]
    assert result["expansions"] == [
        ["files_read_document", "files_search_content", "files_get_metadata"]
    ]


def test_tool_results_are_batched_in_one_user_message():
    client = StubClient(
        [
            response(
                "tool_use",
                [
                    tool_use("files_read_document", "t1", "read a"),
                    tool_use("files_search_content", "t2", "search b"),
                ],
            ),
            response("end_turn", [{"type": "text", "text": "done"}]),
        ]
    )

    run_scenario(scenario(), "baseline", client=client)

    messages = client.requests[1]["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["content"] == [
        tool_use("files_read_document", "t1", "read a"),
        tool_use("files_search_content", "t2", "search b"),
    ]
    assert messages[-1]["role"] == "user"
    assert len(messages[-1]["content"]) == 2
    assert all(block["type"] == "tool_result" for block in messages[-1]["content"])


def test_refusal_ends_scenario_without_reading_stop_details_on_other_reasons():
    refusal = response("refusal", [])
    refusal.stop_details = {"reason": "policy"}
    client = StubClient([refusal])

    result = run_scenario(scenario(), "baseline", client=client)

    assert result["refusal"] is True
    assert result["completed"] is False
    assert result["turns"] == 1


def test_turn_cap_stops_after_fourteen_api_turns():
    client = StubClient(
        [
            response("tool_use", [tool_use("files_read_document", str(i), "read")])
            for i in range(14)
        ]
    )

    result = run_scenario(scenario(), "baseline", client=client)

    assert result["turns"] == 14
    assert result["completed"] is False
    assert len(client.requests) == 14


def test_token_accounting_includes_cache_reads_separately():
    client = StubClient(
        [
            response("end_turn", [{"type": "text", "text": "done"}], 11, 3, 5),
            response("end_turn", [{"type": "text", "text": "done"}], 7, 2, 4),
        ]
    )

    first = run_scenario(scenario(), "baseline", client=client)
    second = run_scenario(scenario(), "baseline", client=client)

    assert first["input_tokens"] == 11
    assert first["output_tokens"] == 3
    assert first["cache_read_tokens"] == 5
    assert second["input_tokens"] == 7


def test_request_uses_the_phase_three_api_contract():
    client = StubClient([response("end_turn", [{"type": "text", "text": "done"}])])

    run_scenario(scenario(), "baseline", client=client)

    request = client.requests[0]
    assert request["model"] == "claude-opus-5"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["fallbacks"] == "default"
    assert request["output_config"] == {"effort": "medium"}
    assert request["max_tokens"] == 8000
    assert "thinking" not in request
    assert all(tool["strict"] is True for tool in request["tools"])


def test_sim_exec_uses_override_or_deterministic_generic_result():
    assert execute("files_read_document", "read notes", {"files_read_document": "canned"}) == "canned"
    assert execute("files_read_document", "read notes", {}) == (
        "Simulated files_read_document: read notes"
    )
