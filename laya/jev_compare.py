"""Run the v0 smoke cases against the TypeSafe Jev API for comparison."""

import json
import os
import time
import urllib.request

from v0_smoke import CASES

API = "https://api.typesafe.ai/v1/systemone"
KEY = os.environ["JEV_API_KEY"]


def main() -> None:
    for name, case in CASES.items():
        body = json.dumps(
            {"state": case["state"], "model": "jev-latest", "questions": case["questions"]}
        ).encode()
        req = urllib.request.Request(
            API,
            data=body,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.load(resp)
        ms = (time.perf_counter() - t) * 1000
        print(f"\n=== {name} ({ms:.0f} ms) ===")
        print(json.dumps(result.get("answers", result), indent=2, default=str))


if __name__ == "__main__":
    main()
