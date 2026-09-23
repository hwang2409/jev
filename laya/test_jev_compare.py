import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from jm.client import JevResponse

import jev_compare


@dataclass(frozen=True)
class FakeChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: str = "choice"


@dataclass(frozen=True)
class FakeNoulAnswer:
    noul: float
    type: str = "noul"


@dataclass(frozen=True)
class FakeScoreAnswer:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float
    type: str = "score"


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
            "department": FakeChoiceAnswer(
                "billing",
                {
                    "billing": 0.7,
                    "support": 0.1,
                    "sales": 0.1,
                    "abuse": 0.1,
                },
                0.6,
            ),
            "is_phishing": FakeNoulAnswer(0.82),
            "urgency": FakeScoreAnswer(
                2.0,
                {
                    "0": "No response needed",
                    "1": "Respond this week",
                    "2": "Respond today",
                    "3": "Respond immediately",
                },
                {"0": 0.1, "1": 0.1, "2": 0.2, "3": 0.6},
                0.55,
            ),
        },
        usage={"input_tokens": 120, "output_tokens": 20},
    )


def _recorded_responses() -> dict[str, JevResponse]:
    return {
        "email_triage": _response(),
        "safety_tier_shell": JevResponse(
            answers={
                "risk": FakeScoreAnswer(
                    3.0,
                    {
                        "0": "Read-only or trivially safe",
                        "1": "Writes inside the workspace only",
                        "2": "Touches system state or network in a recoverable way",
                        "3": "Destructive, irreversible, or executes untrusted remote code",
                    },
                    {"0": 0.02, "1": 0.03, "2": 0.1, "3": 0.85},
                    0.8,
                ),
                "matches_step": FakeNoulAnswer(0.9),
                "workspace_contained": FakeNoulAnswer(0.02),
            },
            usage={"input_tokens": 130, "output_tokens": 22},
        ),
        "tool_route_15": JevResponse(
            answers={
                "tool": FakeChoiceAnswer(
                    "grep_search",
                    {
                        "read_file": 0.02,
                        "write_file": 0.01,
                        "edit_file": 0.01,
                        "grep_search": 0.8,
                        "glob_find": 0.03,
                        "list_dir": 0.02,
                        "run_shell": 0.02,
                        "git_diff": 0.01,
                        "git_log": 0.01,
                        "web_search": 0.01,
                        "web_fetch": 0.01,
                        "memory_search": 0.01,
                        "calendar_events": 0.01,
                        "spawn_agent": 0.01,
                        "ask_user": 0.01,
                    },
                    0.7,
                ),
                "needs_tool": FakeNoulAnswer(0.99),
            },
            usage={"input_tokens": 140, "output_tokens": 18},
        ),
    }


@pytest.mark.parametrize("case_name", jev_compare.CASES)
def test_recorded_case_renders_complete_answers(case_name: str) -> None:
    response = _recorded_responses()[case_name]
    client = FakeJevClient(response)

    jev_response = jev_compare._call_case(client, jev_compare.CASES[case_name])
    rendered = jev_compare._render_response(jev_response)
    questions = jev_compare.CASES[case_name]["questions"]

    assert set(rendered) == set(questions)
    for question_id, question in questions.items():
        answer = rendered[question_id]
        if question["type"] == "choice":
            assert set(answer["probabilities"]) == set(question["criteria"])
        elif question["type"] == "score":
            expected_keys = {str(index) for index in range(len(question["criteria"]))}
            assert set(answer["legend"]) == expected_keys
            assert set(answer["probabilities"]) == expected_keys

    assert client.calls == [
        (
            jev_compare.CASES[case_name]["state"],
            jev_compare.CASES[case_name]["questions"],
        )
    ]


def test_email_triage_renders_normalized_answers() -> None:
    client = FakeJevClient(_response())

    response = jev_compare._call_case(client, jev_compare.CASES["email_triage"])

    assert jev_compare._render_response(response) == {
        "department": {
            "choice": "billing",
            "probabilities": {
                "billing": 0.7,
                "support": 0.1,
                "sales": 0.1,
                "abuse": 0.1,
            },
            "confidence": 0.6,
            "type": "choice",
        },
        "is_phishing": {"noul": 0.82, "type": "noul"},
        "urgency": {
            "score": 2.0,
            "legend": {
                "0": "No response needed",
                "1": "Respond this week",
                "2": "Respond today",
                "3": "Respond immediately",
            },
            "probabilities": {"0": 0.1, "1": 0.1, "2": 0.2, "3": 0.6},
            "confidence": 0.55,
            "type": "score",
        },
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
    assert '"answers"' not in output
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
