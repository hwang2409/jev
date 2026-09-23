"""Run one small live battery through the Vercel AI Gateway."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from jm.answers import ErrorResponse, answers_to_dict
from jm.client import JevClient, resolve_gateway_key
from jm.runner import State

GatewayClient = JevClient

SMOKE_STATE = {
    "focus": 'Caregiver reply: "yes, I can take the Saturday morning shift"',
    "context": {"source": "live-smoke"},
}
SMOKE_QUESTIONS = {
    "accepted": {
        "type": "noul",
        "instructions": "Did the caregiver accept the offered shift?",
        "criteria": {
            "true": "The caregiver commits to working the shift",
            "false": "The caregiver declines or does not commit",
        },
    },
    "outcome": {
        "type": "choice",
        "instructions": "What is the outcome of this reply?",
        "criteria": {
            "accepted": "commits to the shift",
            "declined": "refuses the shift",
            "needs_clarification": "asks a question before deciding",
        },
    },
    "certainty": {
        "type": "score",
        "instructions": "How unconditional is the commitment?",
        "criteria": ["tentative", "committed pending one detail", "fully committed"],
    },
}


def run_smoke(*, output: TextIO) -> int:
    client = GatewayClient()
    try:
        response = client(
            State(
                "live-smoke",
                SMOKE_STATE["focus"],
                {"source": SMOKE_STATE["context"]["source"]},
            ),
            SMOKE_QUESTIONS,
            "typesafe-ai/jev",
        )
    finally:
        client.close()

    if isinstance(response, ErrorResponse):
        output.write(f"fail: {response.error}\n")
        return 1

    output.write(f"ok in {response.latency_ms}ms\n")
    output.write(f"model: {response.served_model or 'typesafe-ai/jev'}\n")
    output.write(f"usage: {json.dumps(response.usage or {}, sort_keys=True)}\n")
    output.write(
        f"answers: {json.dumps(answers_to_dict(response.answers), sort_keys=True)}\n"
    )
    return 0 if response.complete else 1


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main_cli(argv: list[str] | None = None, *, output: TextIO | None = None) -> int:
    output = output or sys.stdout
    if not resolve_gateway_key():
        output.write(
            "skipped: Vercel AI Gateway API key is not set; no API request made\n"
        )
        return 0
    _parser().parse_args(argv)
    return run_smoke(output=output)


if __name__ == "__main__":
    raise SystemExit(main_cli())
