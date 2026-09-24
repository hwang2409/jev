"""Run the phase-2 routing curve and calibration experiments."""

import argparse
import json
import statistics
import time
from pathlib import Path

from jm.client import JevClient

from catalogs import SUBSETS, catalog_for_size
from evalcore import _mean, load_cases, top_k
from evalcore import evaluate as evaluate_cases
from hierarchical import route_hierarchical
from router import route

CURVE_SIZES = [15, 30, 60, 120, 180, 250]
RESCUE_THRESHOLD = 0.8
BIN_RANGES = [
    (0.0, 0.5, "[0-0.5)"),
    (0.5, 0.8, "[0.5-0.8)"),
    (0.8, 0.95, "[0.8-0.95)"),
    (0.95, 1.0, "[0.95-1.0]"),
]


def _successful(results: list[dict]) -> list[dict]:
    return [result for result in results if "error" not in result]


def _accuracy(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize_routing(
    results: list[dict], transport_status_counts: dict[str, int] | None = None
) -> dict:
    """Summarize routing metrics with every input case in accuracy denominators."""
    successful = _successful(results)
    correct = [r for r in successful if r["tool"] == r["expected_tool"]]
    incorrect = [r for r in successful if r["tool"] != r["expected_tool"]]
    top3_hits = [
        r for r in successful if r["expected_tool"] in top_k(r["probabilities"], 3)
    ]
    total_input_tokens = sum(
        r["usage"].get("input_tokens", 0) for r in successful
    )
    total_output_tokens = sum(
        r["usage"].get("output_tokens", 0) for r in successful
    )
    latencies = [r["latency_ms"] for r in successful if r.get("latency_ms") is not None]
    low_confidence = [
        r for r in successful if r.get("confidence", 1.0) < RESCUE_THRESHOLD
    ]
    low_confidence_top3 = [
        r for r in low_confidence
        if r["expected_tool"] in top_k(r["probabilities"], 3)
    ]
    rescued_misses = sum(
        r["tool"] != r["expected_tool"]
        and r["expected_tool"] in top_k(r["probabilities"], 3)
        for r in low_confidence
    )
    unrescued_misses = len(incorrect) + len(results) - len(successful) - rescued_misses
    summary = {
        "cases": len(results),
        "errors": len(results) - len(successful),
        "wrong": len(incorrect),
        "total_misses": len(incorrect) + len(results) - len(successful),
        "top1_accuracy": _accuracy(len(correct), len(results)),
        "top3_accuracy": _accuracy(len(top3_hits), len(results)),
        "confusions": [
            {"id": r["id"], "expected": r["expected_tool"], "chosen": r["tool"]}
            for r in incorrect
        ],
        "mean_confidence_correct": _mean([r["confidence"] for r in correct]),
        "mean_confidence_incorrect": _mean([r["confidence"] for r in incorrect]),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_tokens_per_route": _mean(
            [r["usage"].get("input_tokens", 0) for r in successful]
        ),
        "output_tokens_per_route": _mean(
            [r["usage"].get("output_tokens", 0) for r in successful]
        ),
        "calls_per_route": _mean([r.get("calls", 1) for r in results]),
        "p50_latency_ms": round(statistics.median(latencies)) if latencies else None,
        "error_rate": _accuracy(len(results) - len(successful), len(results)),
        "http_503_errors": sum(r.get("http_status") == 503 for r in results),
        "http_503_error_rate": _accuracy(
            sum(r.get("http_status") == 503 for r in results), len(results)
        ),
        "rescue_threshold": RESCUE_THRESHOLD,
        "low_confidence_cases": len(low_confidence),
        "low_confidence_top3_accuracy": _accuracy(
            len(low_confidence_top3), len(low_confidence)
        ),
        "rescued_misses": rescued_misses,
        "unrescued_misses": unrescued_misses,
        "rescue_recovered_all_misses": unrescued_misses == 0,
    }
    if any("category" in result for result in successful):
        category_correct = sum(
            result.get("category") == result["expected_tool"].split("_", 1)[0]
            for result in successful
        )
        summary["category_top1_accuracy"] = _accuracy(
            category_correct, len(results)
        )
        summary["mean_category_confidence"] = _mean(
            [r["category_confidence"] for r in successful]
        )
        summary["top3_scope"] = "selected-category tools"
    if transport_status_counts:
        total_statuses = sum(transport_status_counts.values())
        summary["transport_status_counts"] = transport_status_counts
        summary["transport_503_rate"] = _accuracy(
            transport_status_counts.get("503", 0), total_statuses
        )
        summary["transport_error_rate"] = _accuracy(
            sum(
                count
                for status, count in transport_status_counts.items()
                if status != "200"
            ),
            total_statuses,
        )
    summary["confidence_bins"] = summarize_confidence_bins(results)
    summary["brier_score"] = brier_score(results)
    return summary


def summarize_confidence_bins(results: list[dict]) -> dict[str, dict]:
    bins = {}
    successful = _successful(results)
    for lower, upper, label in BIN_RANGES:
        in_bin = [
            result
            for result in successful
            if lower <= result["confidence"] < upper
            or (
                label == "[0.95-1.0]"
                and lower <= result["confidence"] <= upper
            )
        ]
        correct = sum(result["tool"] == result["expected_tool"] for result in in_bin)
        bins[label] = {
            "cases": len(in_bin),
            "correct": correct,
            "accuracy": _accuracy(correct, len(in_bin)),
        }
    return bins


def brier_score(results: list[dict]) -> float | None:
    successful = _successful(results)
    if not successful:
        return None
    score = statistics.mean(
        (
            (
                result["chosen_probability"]
                if "chosen_probability" in result
                else result["probabilities"][result["tool"]]
            )
            - int(result["tool"] == result["expected_tool"])
        )
        ** 2
        for result in successful
    )
    return round(score, 4)


def run_curve(cases: list[dict], route_fn=None) -> dict[int, dict]:
    data = {}
    for size in CURVE_SIZES:
        results = evaluate_cases(cases, SUBSETS[size], route_fn=route_fn)
        data[size] = {"summary": summarize_routing(results), "results": results}
    return data


def run_comparison(
    cases: list[dict],
    flat_route_fn=route,
    hierarchical_route_fn=route_hierarchical,
    status_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, dict[int, dict]]:
    """Run flat and hierarchical routes over the same fixed cases."""
    data = {}
    for label, route_fn in (
        ("flat", flat_route_fn),
        ("hierarchical", hierarchical_route_fn),
    ):
        data[label] = {}
        for size in CURVE_SIZES:
            results = evaluate_cases(cases, SUBSETS[size], route_fn=route_fn)
            counts = status_counts.get(label, {}) if status_counts else None
            data[label][size] = {
                "summary": summarize_routing(results, counts),
                "results": results,
            }
    return data


def focused_cases(cases: list[dict], count: int = 20) -> list[dict]:
    """Select one fixed case per core tool, then fill from the curve set."""
    selected = []
    seen_tools = set()
    for case in cases:
        if case["expected_tool"] not in seen_tools:
            selected.append(case)
            seen_tools.add(case["expected_tool"])
    selected.extend(case for case in cases if case not in selected)
    return selected[:count]


def run_limit_probe(
    sizes: list[int], route_fn=route
) -> list[dict]:
    """Probe exact Choice option-limit behavior with deterministic catalogs."""
    results = []
    for size in sizes:
        catalog = catalog_for_size(size)
        started = time.perf_counter()
        try:
            routed = route_fn(
                "Probe the catalog option limit",
                "Select the deterministic probe option without executing it.",
                history=[],
                catalog=catalog,
            )
            results.append(
                {
                    "catalog_size": size,
                    "status": "ok",
                    "tool": routed.tool,
                    "calls": getattr(routed, "calls", 1),
                    "latency_ms": getattr(routed, "latency_ms", None)
                    or round((time.perf_counter() - started) * 1000),
                    "usage": routed.usage,
                }
            )
        except Exception as exc:  # noqa: BLE001 - probe records API shape
            results.append(
                {
                    "catalog_size": size,
                    "status": "error",
                    "error": str(exc),
                    "http_status": getattr(exc, "http_status", None),
                    "attempts": getattr(exc, "attempts", None),
                    "error_type": type(exc).__name__,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
            )
    return results


def run_live_comparison(
    cases: list[dict], pace_seconds: float = 2.1
) -> dict[str, dict[int, dict]]:
    """Run the focused comparison through one paced gateway client per variant."""
    selected_cases = focused_cases(cases)
    data = {}
    for label, route_fn in (("flat", route), ("hierarchical", route_hierarchical)):
        statuses: list[str] = []
        client = JevClient()
        client.set_response_observer(lambda status: statuses.append(str(status)))
        last_call = [0.0]

        def paced_route(task, step, history=None, catalog=None):
            now = time.monotonic()
            wait = pace_seconds - (now - last_call[0])
            if last_call[0] and wait > 0:
                time.sleep(wait)
            last_call[0] = time.monotonic()
            return route_fn(
                task, step, history=history, catalog=catalog, client=client
            )

        try:
            data[label] = {}
            for size in CURVE_SIZES:
                statuses.clear()
                results = evaluate_cases(
                    selected_cases, SUBSETS[size], route_fn=paced_route
                )
                counts = {}
                for status in statuses:
                    counts[status] = counts.get(status, 0) + 1
                data[label][size] = {
                    "summary": summarize_routing(results, counts),
                    "results": results,
                }
        finally:
            client.close()
    return data


def run_live_limit_probe(
    sizes: list[int], pace_seconds: float = 2.1
) -> list[dict]:
    """Probe option counts through one paced gateway client."""
    client = JevClient()
    last_call = [0.0]

    def paced_route(task, step, history=None, catalog=None):
        now = time.monotonic()
        wait = pace_seconds - (now - last_call[0])
        if last_call[0] and wait > 0:
            time.sleep(wait)
        last_call[0] = time.monotonic()
        return route(task, step, history=history, catalog=catalog, client=client)

    try:
        results = run_limit_probe(sizes, paced_route)
    finally:
        client.close()
    return results


def summarize_full(results: list[dict]) -> dict:
    summary = summarize_routing(results)
    summary["coverage"] = summarize_routing(
        [result for result in results if result["id"].startswith("cov-")]
    )
    summary["hard"] = summarize_routing(
        [result for result in results if result["id"].startswith("hard-")]
    )
    summary["confidence_bins"] = summarize_confidence_bins(results)
    summary["brier_score"] = brier_score(results)
    return summary


def run_full(cases: list[dict], route_fn=None) -> dict:
    results = evaluate_cases(cases, SUBSETS[120], route_fn=route_fn)
    return {"summary": summarize_full(results), "results": results}


def format_report(curve: dict[int, dict] | None, full: dict | None) -> str:
    lines = ["Jev phase-2 tool-router eval", "=" * 40]
    if curve is not None:
        lines.extend(
            [
                "curve",
                "size  cases  top1    top3    conf-ok  conf-wrong  wrong  errors  "
                "total_misses  input  output",
            ]
        )
        for size, data in curve.items():
            summary = data["summary"]
            lines.append(
                f"{size:>4}  {summary['cases']:>5}  "
                f"{summary['top1_accuracy']!s:<7} {summary['top3_accuracy']!s:<7} "
                f"{summary['mean_confidence_correct']!s:<8} "
                f"{summary['mean_confidence_incorrect']!s:<11} "
                f"{summary['wrong']:>5} {summary['errors']:>6} "
                f"{summary['total_misses']:>12} "
                f"{summary['total_input_tokens']:>5} "
                f"{summary['total_output_tokens']:>6}"
            )
    if full is not None:
        summary = full["summary"]
        lines.extend(
            [
                "full",
                f"cases: {summary['cases']}",
                f"errors: {summary['errors']}",
                f"top1_accuracy: {summary['top1_accuracy']}",
                f"top3_accuracy: {summary['top3_accuracy']}",
                f"coverage: top1={summary['coverage']['top1_accuracy']}, "
                f"top3={summary['coverage']['top3_accuracy']}",
                f"hard: top1={summary['hard']['top1_accuracy']}, "
                f"top3={summary['hard']['top3_accuracy']}",
                f"mean_confidence_correct: {summary['mean_confidence_correct']}",
                f"mean_confidence_incorrect: {summary['mean_confidence_incorrect']}",
                f"brier_score: {summary['brier_score']}",
                "confidence bins:",
            ]
        )
        for label, bin_summary in summary["confidence_bins"].items():
            lines.append(
                f"  {label}: cases={bin_summary['cases']}, "
                f"accuracy={bin_summary['accuracy']}"
            )
        lines.append(f"confusions ({len(summary['confusions'])}):")
        lines.extend(
            f"  {confusion['id']}: expected {confusion['expected']}, "
            f"chose {confusion['chosen']}"
            for confusion in summary["confusions"]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--curve-only", action="store_true")
    experiment.add_argument("--full-only", action="store_true")
    args = parser.parse_args()

    curve = None
    full = None
    if not args.full_only:
        curve = run_curve(load_cases("evalset_curve.jsonl"))
    if not args.curve_only:
        full = run_full(load_cases("evalset_full.jsonl"))

    print(format_report(curve, full))
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"phase2-{stamp}.json"
    out.write_text(json.dumps({"curve": curve, "full": full}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
