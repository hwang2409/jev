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


async def route_step(
    step: str, catalog: dict[str, str], history: list[str] | None = None
) -> RouteResult:
    """Ask Jev which catalog tool best matches the current agent step."""

    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise JevRouterError("JEV_API_KEY is not set")
    body = build_request(step, history or [], catalog)
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
            return parse_response(data)
    raise JevRouterError("Jev request failed after retries")


__all__ = [
    "API_URL",
    "MODEL",
    "JevRouterError",
    "RouteResult",
    "build_request",
    "parse_response",
    "route_step",
]
