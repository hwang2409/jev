"""Jev-backed tool router: one systemone call routes an agent step to a tool."""

import os
import time
from dataclasses import dataclass

import requests

from catalog import CATALOG

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


@dataclass
class RouteResult:
    tool: str
    probabilities: dict[str, float]
    confidence: float
    needs_tool: float
    step_clarity: float
    usage: dict[str, int]


def build_request(
    task: str, step: str, history: list[str], catalog: dict[str, str]
) -> dict:
    return {
        "state": {
            "task": task,
            "current_step": step,
            "recent_steps": list(history),
        },
        "model": MODEL,
        "questions": {
            "tool": {
                "type": "choice",
                "instructions": (
                    "An agent is working on the task and describes its current "
                    "step. Which single tool should it call to accomplish this "
                    "step?"
                ),
                "criteria": catalog,
            },
            "needs_tool": {
                "type": "noul",
                "instructions": (
                    "Does the current step require calling a tool, rather than "
                    "the agent answering or reasoning directly from what it "
                    "already knows?"
                ),
            },
            "step_clarity": {
                "type": "noul",
                "instructions": (
                    "Is the current step description specific enough to route "
                    "to a single tool with confidence?"
                ),
            },
        },
    }


def parse_response(data: dict) -> RouteResult:
    answers = data["answers"]
    tool = answers["tool"]
    return RouteResult(
        tool=tool["choice"],
        probabilities=tool["probabilities"],
        confidence=tool["confidence"],
        needs_tool=answers["needs_tool"]["noul"],
        step_clarity=answers["step_clarity"]["noul"],
        usage=data["usage"],
    )
