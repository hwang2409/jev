"""Phase-3 Anthropic harnesses with simulated tool execution."""

from collections.abc import Callable
from typing import Any

from catalogs import CATALOG_120
from sim_exec import execute
from router import route

MODEL = "claude-opus-5"
MAX_TOKENS = 8000
MAX_TURNS = 14
BETAS = ["server-side-fallback-2026-07-01"]
SYSTEM_PREAMBLE = (
    "You are a task agent. Complete the user's request with the available tools. "
    "Use tools when needed, then provide a concise final answer."
)
JEV_PROTOCOL = (
    " You have one persistent routing tool. Describe your next step to route, "
    "then call the tool it returns."
)


def _input_schema(properties: dict[str, dict[str, str]], required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _catalog_tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": _input_schema(
            {
                "details": {
                    "type": "string",
                    "description": "All arguments for this action, in plain words",
                }
            },
            ["details"],
        ),
        "strict": True,
    }


def _route_tool() -> dict:
    return {
        "name": "route",
        "description": "Choose the catalog tool for the agent's next step.",
        "input_schema": _input_schema(
            {"step": {"type": "string"}},
            ["step"],
        ),
        "strict": True,
    }


def _catalog_tools(names: list[str] | None = None) -> list[dict]:
    selected = names if names is not None else list(CATALOG_120)
    return [_catalog_tool(name, CATALOG_120[name]) for name in selected]


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    texts = []
    for block in content or []:
        if _value(block, "type") == "text":
            text = _value(block, "text", "")
            if text:
                texts.append(str(text))
    return "".join(texts)


def _tool_uses(content: Any) -> list[Any]:
    return [_block for _block in content or [] if _value(_block, "type") == "tool_use"]


def _usage_value(usage: Any, key: str) -> int:
    value = _value(usage, key, 0)
    return int(value or 0)


def _top_tools(probabilities: dict[str, float], selected: str) -> list[str]:
    names = sorted(probabilities, key=probabilities.get, reverse=True)
    if selected not in names:
        names.insert(0, selected)
    return [name for name in names if name in CATALOG_120][:3]


def _route_result_text(result: Any, names: list[str]) -> str:
    selected = _value(result, "tool", "")
    confidence = float(_value(result, "confidence", 0.0) or 0.0)
    probabilities = _value(result, "probabilities", {}) or {}
    text = f"Routed to {selected} (confidence={confidence:.3f})."
    if confidence < 0.8:
        choices = ", ".join(
            f"{name} ({float(probabilities.get(name, 0.0)):.3f})" for name in names
        )
        text += f" Top candidates: {choices}."
    return text


def _request(
    client: Any,
    system: str,
    messages: list[dict],
    tools: list[dict],
) -> Any:
    return client.beta.messages.create(
        model=MODEL,
        system=system,
        messages=messages,
        tools=tools,
        betas=BETAS,
        fallbacks="default",
        output_config={"effort": "medium"},
        max_tokens=MAX_TOKENS,
    )


def run_scenario(
    scenario: dict,
    mode: str,
    client: Any = None,
    route_fn: Callable[..., Any] | None = None,
) -> dict:
    """Run one scenario through the Jev or baseline manual agent loop."""
    if mode not in {"jev", "baseline"}:
        raise ValueError("mode must be 'jev' or 'baseline'")
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    route_fn = route_fn or route

    system = SYSTEM_PREAMBLE + (JEV_PROTOCOL if mode == "jev" else "")
    messages = [{"role": "user", "content": scenario["task"]}]
    overrides = scenario.get("results", {})
    injected_names: list[str] = []
    tool_calls: list[str] = []
    expansions: list[list[str]] = []
    route_calls = 0
    unrouted_attempts = 0
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    final_text = ""
    completed = False
    refusal = False
    failure: str | None = None
    turns = 0

    while turns < MAX_TURNS:
        if mode == "jev":
            tools = [_route_tool(), *_catalog_tools(injected_names)]
        else:
            tools = _catalog_tools()
        response = _request(client, system, messages, tools)
        turns += 1
        usage = _value(response, "usage", None)
        input_tokens += _usage_value(usage, "input_tokens")
        output_tokens += _usage_value(usage, "output_tokens")
        cache_read_tokens += _usage_value(usage, "cache_read_input_tokens")

        stop_reason = _value(response, "stop_reason")
        content = _value(response, "content", [])
        if stop_reason == "refusal":
            _value(response, "stop_details")
            refusal = True
            failure = "refusal"
            break
        if stop_reason != "tool_use":
            final_text = _text_content(content)
            if stop_reason == "end_turn" and final_text:
                completed = True
            else:
                completed = False
            break

        uses = _tool_uses(content)
        results = []
        saw_non_route = False
        current_injected_names = set(injected_names)
        routed_names: list[str] | None = None
        for use in uses:
            name = str(_value(use, "name", ""))
            use_id = str(_value(use, "id", ""))
            tool_input = _value(use, "input", {}) or {}
            details = _value(tool_input, "details", "")
            block = {
                "type": "tool_result",
                "tool_use_id": use_id,
                "content": "",
            }
            try:
                if mode == "jev" and name == "route":
                    route_result = route_fn(
                        scenario["task"],
                        str(_value(tool_input, "step", "")),
                        catalog=CATALOG_120,
                    )
                    route_calls += 1
                    selected = str(_value(route_result, "tool", ""))
                    confidence = float(
                        _value(route_result, "confidence", 0.0) or 0.0
                    )
                    probabilities = _value(route_result, "probabilities", {}) or {}
                    if confidence < 0.8:
                        names = _top_tools(probabilities, selected)
                        expansions.append(names)
                    else:
                        names = [selected] if selected in CATALOG_120 else []
                    routed_names = names
                    block["content"] = _route_result_text(route_result, names)
                else:
                    saw_non_route = True
                    if mode == "jev" and name not in current_injected_names:
                        unrouted_attempts += 1
                        block["content"] = (
                            f"Tool {name} is unavailable; use route first."
                        )
                        block["is_error"] = True
                    else:
                        tool_calls.append(name)
                        block["content"] = execute(name, str(details), overrides)
            except Exception as exc:  # noqa: BLE001 - tool errors become results
                block["content"] = str(exc)
                block["is_error"] = True
            results.append(block)

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": results})
        if mode == "jev":
            if routed_names is not None:
                injected_names = routed_names
            elif saw_non_route:
                injected_names = []
        if turns == MAX_TURNS:
            failure = "turn_cap"
            break
    else:
        completed = False

    return {
        "mode": mode,
        "turns": turns,
        "tool_calls": tool_calls,
        "route_calls": route_calls,
        "unrouted_attempts": unrouted_attempts,
        "expansions": expansions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "completed": completed,
        "refusal": refusal,
        "final_text": final_text,
        "failure": failure,
    }
