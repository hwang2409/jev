"""Run the v0 smoke cases through the Vercel AI Gateway for comparison."""

import json
import math
import os
import re
import time
import urllib.request
from datetime import UTC
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError

from v0_smoke import CASES

API = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
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
_MAX_WAIT_SECONDS = 300.0


def _resolve_gateway_key() -> str:
    for name in GATEWAY_KEY_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    try:
        zshrc = (Path.home() / ".zshrc").read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("Vercel AI Gateway API key is not set") from exc
    for name in GATEWAY_KEY_NAMES:
        match = re.search(
            rf"^\s*(?:export\s+)?{name}=[\"']?([^\"'\s#]+)",
            zshrc,
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    raise RuntimeError("Vercel AI Gateway API key is not set")


def _gateway_questions(questions: dict) -> dict:
    return {
        question_id: (
            {**question, "type": "boolean"}
            if isinstance(question, dict) and question.get("type") == "noul"
            else question
        )
        for question_id, question in questions.items()
    }


def _normalize_gateway_answers(result: dict, questions: dict) -> dict:
    answers = result.get("answers", {})
    normalized = {}
    for question_id, answer in answers.items():
        question = questions.get(question_id, {})
        question_type = question.get("type") if isinstance(question, dict) else None
        if (
            question_type == "noul"
            and isinstance(answer, dict)
            and answer.get("type") == "boolean"
        ):
            normalized[question_id] = {
                "type": "noul",
                "noul": answer["probability"],
            }
            continue
        if isinstance(answer, dict) and question_type in {"choice", "score"}:
            answer = dict(answer)
            if "confidence" not in answer:
                probabilities = answer.get("probabilities", {})
                criteria = question.get("criteria", ())
                option_count = (
                    len(criteria)
                    if isinstance(criteria, (dict, list))
                    else len(probabilities)
                )
                if option_count <= 1 or not probabilities:
                    answer["confidence"] = 1.0
                else:
                    answer["confidence"] = (
                        option_count * max(probabilities.values()) - 1
                    ) / (option_count - 1)
            if question_type == "score" and "legend" not in answer:
                criteria = question.get("criteria", [])
                answer["legend"] = {
                    str(index): criterion for index, criterion in enumerate(criteria)
                }
        normalized[question_id] = answer
    usage = dict(result.get("usage", {}))
    for gateway_key, native_key in (
        ("inputTokens", "input_tokens"),
        ("outputTokens", "output_tokens"),
    ):
        if gateway_key in usage:
            usage[native_key] = usage.pop(gateway_key)
    return {**result, "answers": normalized, "usage": usage}


def _retry_after(error: HTTPError) -> float | None:
    value = error.headers.get("Retry-After")
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
    return min(_MAX_WAIT_SECONDS, seconds)


def _call_case(case: dict) -> dict:
    questions = case["questions"]
    body = json.dumps(
        {
            "providerOptions": {"gateway": {"zeroDataRetention": True}},
            "state": case["state"],
            "questions": _gateway_questions(questions),
        }
    ).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Authorization": f"Bearer {_resolve_gateway_key()}", **GATEWAY_HEADERS},
    )
    delay = 1.0
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.load(resp)
        except HTTPError as exc:
            if exc.code in {429, 529} and attempt < 2:
                # hint honored as given; jm parity (PR #24)
                retry_after = _retry_after(exc)
                time.sleep(delay if retry_after is None else retry_after)
                delay = min(300.0, delay * 2)
                continue
            raise
        return _normalize_gateway_answers(result, questions)
    raise AssertionError("unreachable")


def main() -> None:
    for name, case in CASES.items():
        t = time.perf_counter()
        result = _call_case(case)
        ms = (time.perf_counter() - t) * 1000
        print(f"\n=== {name} ({ms:.0f} ms) ===")
        print(json.dumps(result.get("answers", result), indent=2, default=str))


if __name__ == "__main__":
    main()
