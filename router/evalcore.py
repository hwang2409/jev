"""Shared case loading, routing evaluation, and metric helpers."""

import json
import statistics
from pathlib import Path

from router import route


def load_cases(path: str = "evalset.jsonl") -> list[dict]:
    lines = Path(path).read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def top_k(probabilities: dict[str, float], k: int) -> list[str]:
    return sorted(probabilities, key=probabilities.get, reverse=True)[:k]


def evaluate(cases: list[dict], catalog=None, route_fn=route) -> list[dict]:
    """Route cases and keep errors as result rows."""
    route_fn = route_fn or route
    results = []
    for case in cases:
        row = dict(case)
        try:
            kwargs = {"history": case["history"]}
            if catalog is not None:
                kwargs["catalog"] = catalog
            routed = route_fn(case["task"], case["step"], **kwargs)
            row.update(
                tool=routed.tool,
                probabilities=routed.probabilities,
                confidence=routed.confidence,
                needs_tool=routed.needs_tool,
                step_clarity=routed.step_clarity,
                usage=routed.usage,
                calls=getattr(routed, "calls", 1),
            )
            for field in (
                "latency_ms",
                "category",
                "category_confidence",
                "category_probabilities",
            ):
                value = getattr(routed, field, None)
                if value is not None:
                    row[field] = value
            if catalog is not None:
                row["chosen_probability"] = routed.probabilities[routed.tool]
        except Exception as exc:  # noqa: BLE001 - eval must survive bad calls
            row["error"] = str(exc)
            row["http_status"] = getattr(exc, "http_status", None)
        results.append(row)
    return results


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def auc(higher: list[float], lower: list[float]) -> float | None:
    if not higher or not lower:
        return None
    wins = sum(
        1 if high > low else 0.5 if high == low else 0
        for high in higher
        for low in lower
    )
    return round(wins / (len(higher) * len(lower)), 4)
