import json
from dataclasses import dataclass
from pathlib import Path

from jm.answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from jm.client import JevResponse

import jev_compare


@dataclass
class FakeJevClient:
    response: JevResponse

    def __post_init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []
        self.closed = False

    def evaluate(self, state: dict, questions: dict) -> JevResponse:
        self.calls.append((state, questions))
        return self.response

    def close(self) -> None:
        self.closed = True


def _response() -> JevResponse:
    return JevResponse(
        answers={
            "department": ChoiceAnswer(
                "billing", {"billing": 0.8, "support": 0.2}, 0.6
            ),
            "is_phishing": NoulAnswer(0.82),
            "urgency": ScoreAnswer(
                2.0,
                {"0": "No response needed", "1": "Today", "2": "Immediately"},
                {"0": 0.1, "1": 0.2, "2": 0.7},
                0.55,
            ),
        },
        usage={"input_tokens": 120, "output_tokens": 20},
    )


def test_recorded_case_renders_normalized_answers() -> None:
    client = FakeJevClient(_response())

    response = jev_compare._call_case(client, jev_compare.CASES["email_triage"])

    assert jev_compare._render_response(response) == {
        "answers": {
            "department": {
                "choice": "billing",
                "probabilities": {"billing": 0.8, "support": 0.2},
                "confidence": 0.6,
                "type": "choice",
            },
            "is_phishing": {"noul": 0.82, "type": "noul"},
            "urgency": {
                "score": 2.0,
                "legend": {
                    "0": "No response needed",
                    "1": "Today",
                    "2": "Immediately",
                },
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                "confidence": 0.55,
                "type": "score",
            },
        },
        "usage": {"input_tokens": 120, "output_tokens": 20},
    }
    assert client.calls == [
        (
            jev_compare.CASES["email_triage"]["state"],
            jev_compare.CASES["email_triage"]["questions"],
        )
    ]


def test_comparison_output_reports_normalized_usage(capsys) -> None:
    client = FakeJevClient(_response())

    jev_compare.main(client)

    output = capsys.readouterr().out
    assert '"input_tokens": 120' in output
    assert '"output_tokens": 20' in output
    assert "inputTokens" not in output
    assert len(client.calls) == len(jev_compare.CASES)
    assert not client.closed


def test_comparison_caller_has_no_transport_or_normalization_logic() -> None:
    source = Path(jev_compare.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "urllib",
        "GATEWAY_HEADERS",
        "_resolve_gateway_key",
        "retry",
        "normalize",
    ):
        assert forbidden not in source


def test_rendered_output_is_json() -> None:
    rendered = jev_compare._render_response(_response())

    assert json.loads(json.dumps(rendered)) == rendered
