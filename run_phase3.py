"""Run the phase-3 scenario comparison with simulated tool execution."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from harness import run_scenario

SCENARIOS_PATH = Path(__file__).with_name("scenarios.jsonl")
MODES = ("jev", "baseline")


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict[str, Any]]:
    """Load non-empty JSONL scenario records."""
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sequence_matches(expected_tools: list[str], actual_tools: list[str]) -> bool:
    return expected_tools == actual_tools


sequence_match = sequence_matches


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    known_fields = (
        "type",
        "id",
        "name",
        "input",
        "text",
        "thinking",
        "signature",
    )
    fields = {
        field: _jsonable(_value(value, field))
        for field in known_fields
        if _value(value, field, None) is not None
    }
    return fields or str(value)


def _request_log(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": request.get("model"),
        "system": request.get("system"),
        "betas": _jsonable(request.get("betas", [])),
        "fallbacks": request.get("fallbacks"),
        "tools": _jsonable(request.get("tools", [])),
        "messages": _jsonable(request.get("messages", [])),
        "max_tokens": request.get("max_tokens"),
        "output_config": _jsonable(request.get("output_config")),
    }


def _response_log(response: Any) -> dict[str, Any]:
    usage = _value(response, "usage", None)
    return {
        "stop_reason": _value(response, "stop_reason"),
        "content": _jsonable(_value(response, "content", [])),
        "usage": {
            "input_tokens": int(_value(usage, "input_tokens", 0) or 0),
            "output_tokens": int(_value(usage, "output_tokens", 0) or 0),
            "cache_read_input_tokens": int(
                _value(usage, "cache_read_input_tokens", 0) or 0
            ),
        },
    }


class _RecordingMessages:
    def __init__(self, delegate: Any, turn_logs: list[dict[str, Any]]) -> None:
        self._delegate = delegate
        self._turn_logs = turn_logs

    def create(self, **request: Any) -> Any:
        turn = {"request": _request_log(request)}
        response = self._delegate.create(**request)
        turn["response"] = _response_log(response)
        self._turn_logs.append(turn)
        return response


class RecordingClient:
    """Proxy an Anthropic client and retain JSON-safe request/response turns."""

    def __init__(self, client: Any) -> None:
        self.turn_logs: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(
            messages=_RecordingMessages(client.beta.messages, self.turn_logs)
        )


def _call_runner(
    run_fn: Callable[..., dict[str, Any]],
    scenario: dict[str, Any],
    mode: str,
    client: Any,
    route_fn: Callable[..., Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if client is not None:
        kwargs["client"] = client
    if route_fn is not None:
        kwargs["route_fn"] = route_fn
    return run_fn(scenario, mode, **kwargs)


def _record(
    scenario: dict[str, Any],
    mode: str,
    result: dict[str, Any],
    turn_logs: list[dict[str, Any]],
) -> dict[str, Any]:
    actual_tools = list(result.get("tool_calls", []))
    return {
        **result,
        "id": scenario["id"],
        "mode": mode,
        "scenario": scenario,
        "expected_tools": list(scenario["expected_tools"]),
        "actual_tool_calls": actual_tools,
        "sequence_match": sequence_matches(scenario["expected_tools"], actual_tools),
        "turn_logs": turn_logs or list(result.get("turn_logs", [])),
    }


def run_scenarios(
    scenarios: list[dict[str, Any]],
    modes: tuple[str, ...] = MODES,
    run_fn: Callable[..., dict[str, Any]] | None = None,
    client_factory: Callable[[], Any] | None = None,
    route_fn: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Run each requested mode in order, with Jev first by default."""
    invalid = set(modes) - set(MODES)
    if invalid:
        raise ValueError(f"invalid mode(s): {sorted(invalid)}")
    run_fn = run_fn or run_scenario
    records = []
    for mode in modes:
        for scenario in scenarios:
            client = client_factory() if client_factory else None
            result = _call_runner(run_fn, scenario, mode, client, route_fn)
            turn_logs = list(getattr(client, "turn_logs", []))
            records.append(_record(scenario, mode, result, turn_logs))
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mode_records = [record for record in records if record["mode"] == mode]
        summary[mode] = {
            "cases": len(mode_records),
            "completed": sum(bool(record.get("completed")) for record in mode_records),
            "sequence_matches": sum(
                bool(record.get("sequence_match")) for record in mode_records
            ),
            "turns": sum(record.get("turns", 0) for record in mode_records),
            "input_tokens": sum(
                record.get("input_tokens", 0) for record in mode_records
            ),
            "output_tokens": sum(
                record.get("output_tokens", 0) for record in mode_records
            ),
            "cache_read_tokens": sum(
                record.get("cache_read_tokens", 0) for record in mode_records
            ),
            "route_calls": sum(
                record.get("route_calls", 0) for record in mode_records
            ),
            "expansions": sum(
                len(record.get("expansions", [])) for record in mode_records
            ),
        }
        summary[mode]["completion_rate"] = (
            summary[mode]["completed"] / summary[mode]["cases"]
            if mode_records
            else 0.0
        )
        summary[mode]["sequence_match_rate"] = (
            summary[mode]["sequence_matches"] / summary[mode]["cases"]
            if mode_records
            else 0.0
        )
    return summary


aggregate = summarize


def format_table(
    records: list[dict[str, Any]], modes: tuple[str, ...] | None = None
) -> str:
    summaries = summarize(records)
    lines = [
        "scenario mode completed sequence_match turns tokens_in tokens_out route_calls expansions",
    ]
    for record in records:
        lines.append(
            f"{record['id']} {record['mode']} {int(bool(record.get('completed')))} "
            f"{int(bool(record.get('sequence_match')))} {record.get('turns', 0)} "
            f"{record.get('input_tokens', 0)} {record.get('output_tokens', 0)} "
            f"{record.get('route_calls', 0)} {len(record.get('expansions', []))}"
        )
    selected_modes = modes or tuple(
        mode for mode in MODES if any(record["mode"] == mode for record in records)
    )
    for mode in selected_modes:
        totals = summaries[mode]
        lines.append(
            f"total {mode} {totals['completed']} {totals['sequence_matches']} "
            f"{totals['turns']} {totals['input_tokens']} {totals['output_tokens']} "
            f"{totals['route_calls']} {totals['expansions']}"
        )
    return "\n".join(lines)


format_report = format_table


def build_payload(
    records: list[dict[str, Any]], modes: tuple[str, ...] = MODES
) -> dict[str, Any]:
    summary = summarize(records)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "modes": list(modes),
        "summary": {mode: summary[mode] for mode in modes},
        "results": records,
    }


def save_payload(
    records: list[dict[str, Any]],
    modes: tuple[str, ...] = MODES,
    output_dir: Path = Path("results"),
) -> Path:
    output_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = output_dir / f"phase3-{stamp}.json"
    output.write_text(json.dumps(build_payload(records, modes), indent=2) + "\n")
    return output


def _real_client_factory() -> Callable[[], RecordingClient]:
    import anthropic

    client = anthropic.Anthropic()
    return lambda: RecordingClient(client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both")
    args = parser.parse_args()
    modes = MODES if args.mode == "both" else (args.mode,)
    records = run_scenarios(
        load_scenarios(), modes=modes, client_factory=_real_client_factory()
    )
    print(format_table(records, modes))
    print(f"\nsaved {save_payload(records, modes)}")


if __name__ == "__main__":
    main()
