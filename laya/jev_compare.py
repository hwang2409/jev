"""Run the recorded v0 smoke cases through the shared Jev client."""

import json
import time
from dataclasses import asdict

from jm.client import JevClient, JevResponse

from v0_smoke import CASES


def _call_case(client: JevClient, case: dict) -> JevResponse:
    return client.evaluate(case["state"], case["questions"])


def _render_response(response: JevResponse) -> dict:
    return {
        "answers": {
            question_id: asdict(answer)
            for question_id, answer in response.answers.items()
        },
        "usage": dict(response.usage or {}),
    }


def main(client: JevClient | None = None) -> None:
    owns_client = client is None
    client = client or JevClient()
    try:
        for name, case in CASES.items():
            started = time.perf_counter()
            response = _call_case(client, case)
            ms = (time.perf_counter() - started) * 1000
            print(f"\n=== {name} ({ms:.0f} ms) ===")
            print(json.dumps(_render_response(response), indent=2, default=str))
    finally:
        if owns_client:
            client.close()


if __name__ == "__main__":
    main()
