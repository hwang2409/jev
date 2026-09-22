from __future__ import annotations

from jmap.answers import ChoiceAnswer, ErrorResponse, NoulAnswer, ScoreAnswer
from jmap.runner import FakeJudge, Runner, State, StateAdmission

QUESTIONS = {
    "is_relevant": {"type": "noul"},
    "kind": {"type": "choice"},
    "risk": {"type": "score"},
}


def test_fake_judge_is_injected_without_http() -> None:
    state = State("docs/guide.md#P1", "the focus", {"source": "docs/guide.md"})
    calls = []

    def fake(state_arg, questions_arg, model_arg):
        calls.append((state_arg, questions_arg, model_arg))
        return "answer"

    runner = Runner(judge_fn=fake, model="jev-1.13.0")
    assert runner.judge(state, QUESTIONS) == "answer"
    assert calls == [(state, QUESTIONS, "jev-1.13.0")]


def test_fake_judge_returns_deterministic_typed_answers() -> None:
    state = State("stdin#L1", "launch", {"source": "stdin"})
    fake = FakeJudge()

    first = fake(state, QUESTIONS, "jev-1.13.0")
    second = fake(state, QUESTIONS, "jev-1.13.0")

    assert first == second
    assert isinstance(first.answers["is_relevant"], NoulAnswer)
    assert isinstance(first.answers["kind"], ChoiceAnswer)
    assert isinstance(first.answers["risk"], ScoreAnswer)


def test_fake_judge_can_return_incomplete_answers() -> None:
    state = State("stdin#L1", "launch", {"source": "stdin"})
    response = FakeJudge(mode="incomplete")(state, QUESTIONS, "jev-1.13.0")

    assert response.complete is False
    assert response.missing_questions == ("risk",)
    assert "risk" not in response.answers


def test_fake_judge_can_return_an_operational_error() -> None:
    state = State("stdin#L1", "launch", {"source": "stdin"})
    first = FakeJudge(mode="error")(state, QUESTIONS, "jev-1.13.0")
    second = FakeJudge(mode="error")(state, QUESTIONS, "jev-1.13.0")

    assert first == second == ErrorResponse("fake operational error")


def test_runner_admits_states_in_input_order_and_keeps_skipped_refs() -> None:
    states = [State(f"stdin#L{i}", str(i), {"line": i}) for i in range(1, 4)]
    admission = Runner(judge_fn=lambda *_: None).admit(states, max_chunks=2)

    assert isinstance(admission, StateAdmission)
    assert admission.discovered == 3
    assert [state.state_ref for state in admission.admitted] == ["stdin#L1", "stdin#L2"]
    assert [state.state_ref for state in admission.skipped] == ["stdin#L3"]
    assert admission.skip_boundary == "max_chunks=2"
