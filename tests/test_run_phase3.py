import json

from run_phase3 import (
    build_payload,
    format_table,
    run_scenarios,
    sequence_matches,
    summarize,
)


def scenario(id, expected_tools):
    return {
        "id": id,
        "task": "task",
        "expected_tools": expected_tools,
        "results": {},
        "answer_keys": ["answer"],
    }


def test_sequence_match_requires_exact_order_and_length():
    assert sequence_matches(["a", "b"], ["a", "b"])
    assert not sequence_matches(["a", "b"], ["a", "b", "c"])
    assert not sequence_matches(["a", "b"], ["b", "a"])


def test_runner_uses_jev_for_all_scenarios_before_baseline():
    scenarios = [scenario("one", ["a", "b"]), scenario("two", ["c", "d"])]
    calls = []

    def fake_run(current, mode, client=None, route_fn=None):
        calls.append((mode, current["id"]))
        return {
            "mode": mode,
            "turns": 2,
            "tool_calls": current["expected_tools"],
            "route_calls": 1 if mode == "jev" else 0,
            "expansions": [["a", "b"]] if mode == "jev" else [],
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_tokens": 2,
            "completed": True,
            "final_text": "answer",
        }

    records = run_scenarios(scenarios, run_fn=fake_run)

    assert calls == [("jev", "one"), ("jev", "two"), ("baseline", "one"), ("baseline", "two")]
    assert all(record["sequence_match"] for record in records)
    assert all(record["actual_tool_calls"] == record["expected_tools"] for record in records)


def test_runner_records_mismatch_and_summary_totals():
    current = scenario("one", ["a", "b"])

    def fake_run(current, mode, client=None, route_fn=None):
        return {
            "mode": mode,
            "turns": 3,
            "tool_calls": ["b", "a"],
            "route_calls": 2,
            "expansions": [["a", "b"]],
            "input_tokens": 11,
            "output_tokens": 5,
            "cache_read_tokens": 1,
            "completed": False,
        }

    records = run_scenarios([current], modes=("jev",), run_fn=fake_run)
    summary = summarize(records)["jev"]

    assert records[0]["sequence_match"] is False
    assert records[0]["actual_tool_calls"] == ["b", "a"]
    assert summary["cases"] == 1
    assert summary["completed"] == 0
    assert summary["sequence_matches"] == 0
    assert summary["turns"] == 3
    assert summary["input_tokens"] == 11
    assert summary["route_calls"] == 2
    assert summary["expansions"] == 1


def test_table_contains_rows_and_mode_totals():
    records = [
        {
            "id": "one",
            "mode": "jev",
            "completed": True,
            "sequence_match": True,
            "turns": 2,
            "input_tokens": 10,
            "output_tokens": 4,
            "route_calls": 1,
            "expansions": [],
        }
    ]

    table = format_table(records)

    assert "scenario mode completed sequence_match" in table
    assert "one jev" in table
    assert "total jev" in table


def test_payload_has_full_records_and_summary(tmp_path):
    records = [
        {
            "id": "one",
            "mode": "jev",
            "expected_tools": ["a"],
            "actual_tool_calls": ["a"],
            "sequence_match": True,
            "turn_logs": [{"request": {}, "response": {}}],
        }
    ]

    payload = build_payload(records, ("jev",))
    output = tmp_path / "phase3.json"
    output.write_text(json.dumps(payload, indent=2))
    loaded = json.loads(output.read_text())

    assert loaded["results"][0]["turn_logs"]
    assert loaded["results"][0]["actual_tool_calls"] == ["a"]
    assert loaded["summary"]["jev"]["sequence_matches"] == 1
