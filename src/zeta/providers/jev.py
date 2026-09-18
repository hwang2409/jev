"""Async client for routing agent steps through Jev."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
_MAX_ATTEMPTS = 3


class JevRouterError(RuntimeError):
    """Raised when Jev cannot classify an agent step."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class RouteResult:
    tool: str
    probabilities: dict[str, float]
    confidence: float
    needs_tool: float
    step_clarity: float
    usage: dict[str, int]


@dataclass(frozen=True, slots=True)
class AutoRouteResult:
    tool: str
    probabilities: dict[str, float]
    confidence: float
    needs_tool: float
    usage: dict[str, int]


@dataclass(frozen=True, slots=True)
class TriageResult:
    keep_probabilities: dict[str, float]
    usage: dict[str, int]


def build_request(
    step: str, history: list[str], catalog: dict[str, str]
) -> dict[str, Any]:
    """Build the request body expected by Jev System One."""

    return {
        "state": {
            "current_step": step,
            "recent_steps": list(history[-5:]),
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


def build_auto_route_request(
    task: str,
    last_assistant: str,
    last_results: list[dict[str, str]],
    catalog: dict[str, str],
) -> dict[str, Any]:
    """Build the request for harness-side routing between provider turns."""

    return {
        "state": {
            "task": task[:500],
            "last_assistant": last_assistant[:300],
            "last_results": [
                {
                    "tool": result["tool"],
                    "excerpt": result["excerpt"][:200],
                }
                for result in last_results[-2:]
            ],
        },
        "model": MODEL,
        "questions": {
            "tool": {
                "type": "choice",
                "instructions": (
                    "Which single catalog tool should the agent use for its "
                    "next action, if it needs a tool?"
                ),
                "criteria": catalog,
            },
            "needs_tool": {
                "type": "noul",
                "instructions": (
                    "Does the agent need to call a tool on the next turn, "
                    "rather than answer directly?"
                ),
            },
        },
    }


def parse_response(data: dict[str, Any]) -> RouteResult:
    """Parse one successful Jev response."""

    try:
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
    except (KeyError, TypeError, ValueError) as exc:
        raise JevRouterError(f"invalid Jev response: {exc}") from exc


def build_triage_request(
    task: str,
    items: list[dict[str, str]],
    *,
    latest_assistant_text: str = "",
    recent_tool_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Build the request body for compaction triage."""

    return {
        "state": {
            "task": task[:500],
            "latest_assistant_text": latest_assistant_text[:300],
            "recent_tool_actions": list(recent_tool_actions or [])[-3:],
            "items": items,
        },
        "model": MODEL,
        "questions": {
            item["id"]: {
                "type": "noul",
                "instructions": (
                    f"Will the details of item {item['id']} be needed to finish "
                    "the task, beyond what the excerpt already shows?"
                ),
            }
            for item in items
        },
    }


def parse_triage_response(data: dict[str, Any], item_ids: list[str]) -> TriageResult:
    """Parse one successful Jev triage response."""

    try:
        answers = data["answers"]
        probabilities = {
            item_id: float(answers[item_id]["noul"])
            for item_id in item_ids
        }
        if any(not 0 <= probability <= 1 for probability in probabilities.values()):
            raise ValueError("noul probabilities must be between 0 and 1")
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            raise TypeError("usage must be an object")
        return TriageResult(probabilities, dict(usage))
    except (KeyError, TypeError, ValueError) as exc:
        raise JevRouterError(f"invalid Jev triage response: {exc}") from exc


async def route_step(
    step: str, catalog: dict[str, str], history: list[str] | None = None
) -> RouteResult:
    """Ask Jev which catalog tool best matches the current agent step."""

    body = build_request(step, history or [], catalog)
    return parse_response(await _post_json(body))


async def auto_route(
    task: str,
    last_assistant: str,
    last_results: list[dict[str, str]],
    catalog: dict[str, str],
) -> AutoRouteResult:
    """Ask Jev which tool, if any, the next provider turn needs."""

    data = await _post_json(
        build_auto_route_request(task, last_assistant, last_results, catalog)
    )
    try:
        answers = data["answers"]
        tool = answers["tool"]
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            raise TypeError("usage must be an object")
        return AutoRouteResult(
            tool=tool["choice"],
            probabilities=tool["probabilities"],
            confidence=tool["confidence"],
            needs_tool=answers["needs_tool"]["noul"],
            usage=dict(usage),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise JevRouterError(f"invalid Jev auto-route response: {exc}") from exc


async def _post_json(body: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise JevRouterError("JEV_API_KEY is not set")
    headers = {"Authorization": f"Bearer {key}"}
    delay = 1.0
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await client.post(
                    API_URL, json=body, headers=headers, timeout=60.0
                )
            except httpx.HTTPError as exc:
                raise JevRouterError(f"Jev request failed: {exc}") from exc
            if response.status_code in {429, 529} and attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if response.status_code >= 400:
                detail = getattr(response, "text", "").strip()
                suffix = f": {detail}" if detail else ""
                raise JevRouterError(
                    f"Jev request failed with HTTP {response.status_code}{suffix}",
                    status_code=response.status_code,
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise JevRouterError("Jev response was not valid JSON") from exc
            if not isinstance(data, dict):
                raise JevRouterError("Jev response must be an object")
            return data
    raise JevRouterError("Jev request failed after retries")


async def triage(
    task: str,
    items: list[dict[str, str]],
    *,
    latest_assistant_text: str = "",
    recent_tool_actions: list[str] | None = None,
) -> TriageResult:
    """Ask Jev which tool results can be dropped during compaction."""

    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise JevRouterError("JEV_API_KEY is not set")
    body = build_triage_request(
        task,
        items,
        latest_assistant_text=latest_assistant_text,
        recent_tool_actions=recent_tool_actions,
    )
    headers = {"Authorization": f"Bearer {key}"}
    delay = 1.0
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await client.post(
                    API_URL, json=body, headers=headers, timeout=60.0
                )
            except httpx.HTTPError as exc:
                raise JevRouterError(f"Jev request failed: {exc}") from exc
            if response.status_code in {429, 529} and attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if response.status_code >= 400:
                detail = getattr(response, "text", "").strip()
                suffix = f": {detail}" if detail else ""
                raise JevRouterError(
                    f"Jev request failed with HTTP {response.status_code}{suffix}",
                    status_code=response.status_code,
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise JevRouterError("Jev response was not valid JSON") from exc
            return parse_triage_response(data, [item["id"] for item in items])
    raise JevRouterError("Jev request failed after retries")


__all__ = [
    "API_URL",
    "MODEL",
    "AutoRouteResult",
    "JevRouterError",
    "RouteResult",
    "TriageResult",
    "auto_route",
    "build_auto_route_request",
    "build_request",
    "build_triage_request",
    "parse_response",
    "parse_triage_response",
    "route_step",
    "triage",
]
