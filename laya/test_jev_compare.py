import io
import json
from urllib.error import HTTPError

import jev_compare
import pytest


def test_normalize_gateway_answers_rebuilds_native_fields() -> None:
    questions = {
        "kind": {
            "type": "choice",
            "criteria": {"yes": "yes", "no": "no"},
        },
        "risk": {
            "type": "score",
            "criteria": ["low", "medium", "high"],
        },
        "matches": {"type": "noul"},
    }
    result = jev_compare._normalize_gateway_answers(
        {
            "answers": {
                "kind": {
                    "type": "choice",
                    "choice": "yes",
                    "probabilities": {"yes": 0.8, "no": 0.2},
                },
                "risk": {
                    "type": "score",
                    "score": 2,
                    "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                },
                "matches": {"type": "boolean", "probability": 0.75},
            },
            "usage": {"inputTokens": 12, "outputTokens": 5},
        },
        questions,
    )

    assert result["answers"]["kind"]["type"] == "choice"
    assert result["answers"]["kind"]["choice"] == "yes"
    assert result["answers"]["kind"]["probabilities"] == {"yes": 0.8, "no": 0.2}
    assert result["answers"]["kind"]["confidence"] == pytest.approx(0.6)
    assert result["answers"]["risk"] == {
        "type": "score",
        "score": 2,
        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
        "confidence": pytest.approx(0.55),
        "legend": {"0": "low", "1": "medium", "2": "high"},
    }
    assert result["answers"]["matches"] == {"type": "noul", "noul": 0.75}
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 5}


def test_gateway_headers_are_literal_and_retry_after_is_honored(monkeypatch) -> None:
    sleeps: list[float] = []
    responses = iter(
        [
            HTTPError(
                jev_compare.API,
                429,
                "rate limited",
                {"Retry-After": "59"},
                io.BytesIO(),
            ),
            io.BytesIO(
                json.dumps(
                    {
                        "answers": {},
                        "usage": {"inputTokens": 1, "outputTokens": 2},
                    }
                ).encode()
            ),
        ]
    )

    def urlopen(request, timeout):
        assert timeout == 60
        assert dict(request.header_items()) == {
            "Authorization": "Bearer test-key",
            "Content-type": "application/json",
            "Accept-encoding": "identity",
            "Ai-evaluation-model-specification-version": "4",
            "Ai-gateway-auth-method": "api-key",
            "Ai-gateway-protocol-version": "0.0.1",
            "Ai-model-id": "typesafe-ai/jev",
        }
        value = next(responses)
        if isinstance(value, HTTPError):
            raise value
        return _ResponseContext(value)

    monkeypatch.setenv("VERCEL_AI_GATEWAY", "test-key")
    monkeypatch.setattr(jev_compare.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(jev_compare.time, "sleep", sleeps.append)

    result = jev_compare._call_case(
        {
            "state": {},
            "questions": {},
        }
    )

    assert sleeps == [59.0]
    assert result["usage"] == {"input_tokens": 1, "output_tokens": 2}


class _ResponseContext:
    def __init__(self, response) -> None:
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, *_args) -> None:
        self.response.close()
