"""Threshold sweep and Pareto/selection rules from DESIGN §8."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import metrics


def thresholds(cases: Sequence[Mapping[str, Any]], reference: float = 0.6) -> list[float]:
    return sorted(set(metrics.observed_scores(cases)) | {reference})


def pareto_frontier(table: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points = list(table)
    result = []
    for point in points:
        dominated = any(
            other is not point
            and (other.get("any_injection_rate") or 0) <= (point.get("any_injection_rate") or 0)
            and other.get("packet_recall") is not None
            and point.get("packet_recall") is not None
            and other["packet_recall"] >= point["packet_recall"]
            and ((other.get("any_injection_rate") or 0) < (point.get("any_injection_rate") or 0) or other["packet_recall"] > point["packet_recall"])
            for other in points
        )
        if not dominated:
            result.append(dict(point))
    return sorted(result, key=lambda x: x["tau"])


def select_tau(table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = next((x.get("packet_recall_baseline") for x in table if x.get("packet_recall_baseline") is not None), None)
    if baseline is None:
        return {"selected_tau": None, "reason": "no baseline recall"}
    qualifying = [x for x in table if x.get("any_injection_rate") == 0 and x.get("packet_recall") is not None and x["packet_recall"] >= .9 * baseline]
    if not qualifying:
        return {"selected_tau": None, "reason": "no qualifying tau", "baseline_recall": baseline}
    chosen = min(qualifying, key=lambda x: x["tau"])
    return {"selected_tau": chosen["tau"], "reason": "smallest qualifying tau", "baseline_recall": baseline}


def sweep(cases: Sequence[Mapping[str, Any]], reference: float = 0.6) -> dict[str, Any]:
    values = thresholds(cases, reference)
    table = [metrics.metrics(cases, tau) for tau in values]
    return {"thresholds": values, "table": table, "pareto": pareto_frontier(table), "selection": select_tau(table)}
