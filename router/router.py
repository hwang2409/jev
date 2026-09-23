"""Jev-backed tool router: one gateway call routes an agent step to a tool."""

import math
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from catalog import CATALOG

API_URL = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
MODEL = "typesafe-ai/jev"
GATEWAY_KEY_NAMES = ("VERCEL_AI_GATEWAY", "AI_GATEWAY_API_KEY", "VERCEL_JEV_KEY")
GATEWAY_HEADERS = {
    "Content-Type": "application/json",
    "Accept-Encoding": "identity",
    "ai-evaluation-model-specification-version": "4",
    "ai-gateway-auth-method": "api-key",
    "ai-gateway-protocol-version": "0.0.1",
    "ai-model-id": MODEL,
}


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
            "recent_steps": list(history[-5:]),
        },
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
    tool_confidence = _answer_confidence(tool)
    return RouteResult(
        tool=tool["choice"],
        probabilities=tool["probabilities"],
        confidence=tool_confidence,
        needs_tool=answers["needs_tool"]["noul"],
        step_clarity=answers["step_clarity"]["noul"],
        usage=data["usage"],
    )


def route(
    task: str,
    step: str,
    history: list[str] | None = None,
    catalog: dict[str, str] | None = None,
    session=None,
    api_key: str | None = None,
) -> RouteResult:
    body = build_request(task, step, history or [], catalog or CATALOG)
    key = api_key or _resolve_gateway_key()
    if not key:
        raise KeyError("Vercel AI Gateway API key is not set")
    sess = session or requests.Session()
    gateway_body = {
        "providerOptions": {"gateway": {"zeroDataRetention": True}},
        "state": body["state"],
        "questions": _gateway_questions(body["questions"]),
    }
    headers = {"Authorization": f"Bearer {key}", **GATEWAY_HEADERS}
    delay = 1.0
    for attempt in range(3):
        resp = sess.post(API_URL, json=gateway_body, headers=headers, timeout=60)
        if resp.status_code in (429, 529) and attempt < 2:
            time.sleep(_retry_after(resp) or delay)
            delay = min(300.0, delay * 2)
            continue
        resp.raise_for_status()
        return parse_response(_normalize_gateway_response(resp.json()))
    raise AssertionError("unreachable")


def _answer_confidence(answer: dict) -> float:
    raw_confidence = answer.get("confidence")
    if raw_confidence is not None:
        return float(raw_confidence)
    probabilities = {
        str(choice): float(probability)
        for choice, probability in answer.get("probabilities", {}).items()
    }
    if len(probabilities) <= 1:
        return 1.0
    return (len(probabilities) * max(probabilities.values()) - 1) / (
        len(probabilities) - 1
    )


def _gateway_questions(questions: dict) -> dict:
    return {
        question_id: (
            {**question, "type": "boolean"}
            if isinstance(question, dict) and question.get("type") == "noul"
            else question
        )
        for question_id, question in questions.items()
    }


def _normalize_gateway_response(data: dict) -> dict:
    answers = data["answers"]
    normalized = {
        question_id: (
            {"type": "noul", "noul": answer["probability"]}
            if isinstance(answer, dict) and answer.get("type") == "boolean"
            else answer
        )
        for question_id, answer in answers.items()
    }
    return {**data, "answers": normalized}


def _resolve_gateway_key() -> str | None:
    for name in GATEWAY_KEY_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    try:
        zshrc = (Path.home() / ".zshrc").read_text(encoding="utf-8")
    except OSError:
        return None
    for name in GATEWAY_KEY_NAMES:
        match = re.search(
            rf"^\s*(?:export\s+)?{name}=[\"']?([^\"'\s#]+)",
            zshrc,
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    return None


def _retry_after(response) -> float | None:
    value = getattr(response, "headers", {}).get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = retry_at.timestamp() - time.time()
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return max(60.0, seconds)
