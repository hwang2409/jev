"""Deterministic simulated execution for phase-3 scenarios."""


def execute(tool_name: str, details: str, overrides: dict) -> str:
    """Return a canned scenario result or a stable generic acknowledgement."""
    try:
        if tool_name in overrides:
            return str(overrides[tool_name])
        return f"Simulated {tool_name}: {details}"
    except Exception as exc:  # noqa: BLE001 - simulated execution never raises
        return f"Simulated {tool_name}: {details} ({exc})"
