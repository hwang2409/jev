"""Run the routing evalset through Jev and report accuracy + calibration."""

import time
from pathlib import Path

from evalcore import _mean, auc, evaluate, load_cases, top_k


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    clear = [r for r in results if not r["vague"] and r["expected_needs_tool"]
             and r["expected_tool"] is not None]
    routed_clear = [r for r in clear if "error" not in r]
    correct = [r for r in routed_clear if r["tool"] == r["expected_tool"]]
    incorrect = [r for r in routed_clear if r["tool"] != r["expected_tool"]]
    in_top3 = [
        r for r in routed_clear if r["expected_tool"] in top_k(r["probabilities"], 3)
    ]
    no_tool = [r for r in ok if not r["expected_needs_tool"]]
    tool_cases = [r for r in ok if r["expected_needs_tool"]]
    vague = [r for r in ok if r["vague"]]
    non_vague = [r for r in ok if not r["vague"]]
    return {
        "cases": len(results),
        "errors": len(results) - len(ok),
        "clear_cases": len(clear),
        "top1_accuracy": round(len(correct) / len(clear), 4) if clear else None,
        "top3_accuracy": round(len(in_top3) / len(clear), 4) if clear else None,
        "confusions": [
            {"id": r["id"], "expected": r["expected_tool"], "chosen": r["tool"]}
            for r in incorrect
        ],
        "mean_confidence_correct": _mean([r["confidence"] for r in correct]),
        "mean_confidence_incorrect": _mean([r["confidence"] for r in incorrect]),
        "needs_tool_mean_on_tool_cases": _mean([r["needs_tool"] for r in tool_cases]),
        "needs_tool_mean_on_no_tool_cases": _mean([r["needs_tool"] for r in no_tool]),
        "needs_tool_auc": auc(
            [r["needs_tool"] for r in tool_cases],
            [r["needs_tool"] for r in no_tool],
        ),
        "clarity_mean_on_clear": _mean([r["step_clarity"] for r in non_vague]),
        "clarity_mean_on_vague": _mean([r["step_clarity"] for r in vague]),
        "clarity_auc": auc(
            [r["step_clarity"] for r in non_vague],
            [r["step_clarity"] for r in vague],
        ),
        "total_input_tokens": sum(r["usage"]["input_tokens"] for r in ok),
        "total_output_tokens": sum(r["usage"]["output_tokens"] for r in ok),
    }


def format_report(summary: dict) -> str:
    lines = ["Jev tool-router eval", "=" * 40]
    for key, value in summary.items():
        if key == "confusions":
            lines.append(f"confusions ({len(value)}):")
            for c in value:
                lines.append(f"  {c['id']}: expected {c['expected']}, chose {c['chosen']}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def main() -> None:
    cases = load_cases()
    results = evaluate(cases)
    summary = summarize(results)
    print(format_report(summary))
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{stamp}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
