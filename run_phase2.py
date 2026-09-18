"""Run the phase-2 routing curve and calibration experiments."""

import argparse
import json
import statistics
import time
from pathlib import Path

from catalogs import CATALOG_120, SUBSETS
from router import route
from run_eval import _mean, top_k


CURVE_SIZES = [15, 30, 60, 120]
BIN_RANGES = [
    (0.0, 0.5, "[0-0.5)"),
    (0.5, 0.8, "[0.5-0.8)"),
    (0.8, 0.95, "[0.8-0.95)"),
    (0.95, 1.0, "[0.95-1.0]"),
]


def load_cases(path: str) -> list[dict]:
    lines = Path(path).read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def evaluate_cases(cases: list[dict], catalog: dict[str, str], route_fn=None) -> list[dict]:
    """Route cases with one catalog and keep errors as result rows."""
    route_fn = route_fn or route
    results = []
    for case in cases:
        row = dict(case)
        try:
            routed = route_fn(
                case["task"],
                case["step"],
                history=case["history"],
                catalog=catalog,
            )
            row.update(
                tool=routed.tool,
                probabilities=routed.probabilities,
                confidence=routed.confidence,
                chosen_probability=routed.probabilities[routed.tool],
                needs_tool=routed.needs_tool,
                step_clarity=routed.step_clarity,
                usage=routed.usage,
            )
        except Exception as exc:  # noqa: BLE001 - eval must survive bad calls
            row["error"] = str(exc)
        results.append(row)
    return results


def _successful(results: list[dict]) -> list[dict]:
    return [result for result in results if "error" not in result]


def _accuracy(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize_routing(results: list[dict]) -> dict:
    """Summarize routing metrics with every input case in accuracy denominators."""
    successful = _successful(results)
    correct = [r for r in successful if r["tool"] == r["expected_tool"]]
    incorrect = [r for r in successful if r["tool"] != r["expected_tool"]]
    top3_hits = [
        r for r in successful if r["expected_tool"] in top_k(r["probabilities"], 3)
    ]
    return {
        "cases": len(results),
        "errors": len(results) - len(successful),
        "top1_accuracy": _accuracy(len(correct), len(results)),
        "top3_accuracy": _accuracy(len(top3_hits), len(results)),
        "confusions": [
            {"id": r["id"], "expected": r["expected_tool"], "chosen": r["tool"]}
            for r in incorrect
        ],
        "mean_confidence_correct": _mean([r["confidence"] for r in correct]),
        "mean_confidence_incorrect": _mean([r["confidence"] for r in incorrect]),
        "total_input_tokens": sum(r["usage"]["input_tokens"] for r in successful),
        "total_output_tokens": sum(r["usage"]["output_tokens"] for r in successful),
    }


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
    results = evaluate_cases(cases, CATALOG_120, route_fn=route_fn)
    return {"summary": summarize_full(results), "results": results}


def format_report(curve: dict[int, dict] | None, full: dict | None) -> str:
    lines = ["Jev phase-2 tool-router eval", "=" * 40]
    if curve is not None:
        lines.extend(
            [
                "curve",
                "size  cases  top1    top3    conf-ok  conf-wrong  wrong  input  output",
            ]
        )
        for size, data in curve.items():
            summary = data["summary"]
            lines.append(
                f"{size:>4}  {summary['cases']:>5}  "
                f"{summary['top1_accuracy']!s:<7} {summary['top3_accuracy']!s:<7} "
                f"{summary['mean_confidence_correct']!s:<8} "
                f"{summary['mean_confidence_incorrect']!s:<11} "
                f"{len(summary['confusions']):>5} "
                f"{summary['total_input_tokens']:>5} {summary['total_output_tokens']:>6}"
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
