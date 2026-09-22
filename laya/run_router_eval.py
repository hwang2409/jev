"""Run the phase-1 60-case router evalset against local Laya checkpoints.

Reuses the router repo's evalset, request builder, and metrics verbatim;
only the backend swaps from the Jev API to a local Laya forward pass.

Usage: uv run python run_router_eval.py [device] [checkpoint ...]
"""

import json
import sys
import time
from pathlib import Path

ROUTER_DIR = Path(__file__).resolve().parent.parent / "router"
sys.path.insert(0, str(ROUTER_DIR))

import laya  # noqa: E402
from catalog import CATALOG  # noqa: E402
from evalcore import evaluate, load_cases  # noqa: E402
from router import RouteResult, build_request  # noqa: E402
from run_eval import format_report, summarize  # noqa: E402

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "mps"
CHECKPOINTS = sys.argv[2:] or ["english", "typed-decisions"]


def make_route_fn(router: laya.Router, model: str, latencies: list[float]):
    def route_laya(task, step, history=None, catalog=None):
        body = build_request(task, step, history or [], catalog or CATALOG)
        t = time.perf_counter()
        result = router.predict(body["state"], body["questions"], model=model)
        latencies.append((time.perf_counter() - t) * 1000)
        answers = result["answers"]
        tool = answers["tool"]
        return RouteResult(
            tool=tool["choice"],
            probabilities=tool["probabilities"],
            confidence=tool["confidence"],
            needs_tool=answers["needs_tool"]["noul"],
            step_clarity=answers["step_clarity"]["noul"],
            usage={"input_tokens": 0, "output_tokens": 0},
        )

    return route_laya


def main() -> None:
    cases = load_cases(str(ROUTER_DIR / "evalset.jsonl"))
    router = laya.Router(device=DEVICE, preload=True)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    for model in CHECKPOINTS:
        latencies: list[float] = []
        results = evaluate(cases, route_fn=make_route_fn(router, model, latencies))
        summary = summarize(results)
        latencies.sort()
        summary["p50_latency_ms"] = round(latencies[len(latencies) // 2], 1)
        summary["p95_latency_ms"] = round(latencies[int(len(latencies) * 0.95)], 1)
        print(f"\n### laya[{model}] on {DEVICE}\n")
        print(format_report(summary))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = out_dir / f"laya-{model}-{stamp}.json"
        out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
