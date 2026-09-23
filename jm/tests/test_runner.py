from __future__ import annotations

import io
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from jm.answers import (
    ChoiceAnswer,
    ErrorResponse,
    JudgeResponse,
    NoulAnswer,
    ScoreAnswer,
)
from jm.api import (
    GATEWAY_ENDPOINT,
    GATEWAY_MODEL,
    MAX_RESPONSE_BYTES,
    MAX_WAIT_SECONDS,
    GatewayClient,
)
from jm.gates import PolicySyntaxError
from jm.presets import (
    Preset,
    PresetUsageError,
    PresetValidationError,
    resolve_preset,
    validate_preset,
)
from jm.runner import (
    FakeJudge,
    Runner,
    State,
    StateAdmission,
    StateLimits,
    StateRejection,
    admit_states,
    bm25_rank,
    tokenize,
)

QUESTIONS = {
    "is_relevant": {"type": "noul"},
    "kind": {"type": "choice"},
    "risk": {"type": "score"},
}


@pytest.fixture(autouse=True)
def clear_local_gateway_keys(monkeypatch) -> None:
    for name in ("VERCEL_AI_GATEWAY", "VERCEL_JEV_KEY"):
        monkeypatch.delenv(name, raising=False)


def _complete_payload() -> dict[str, object]:
    return {"answers": {"is_relevant": {"type": "boolean", "probability": 0.9}}}


def test_runner_requires_an_explicit_judge_function() -> None:
    with pytest.raises(TypeError, match="judge_fn"):
        Runner(None)


def test_fake_judge_is_injected_without_http() -> None:
    state = State("docs/guide.md#P1", "the focus", {"source": "docs/guide.md"})
    calls = []

    def fake(state_arg, questions_arg, model_arg):
        calls.append((state_arg, questions_arg, model_arg))
        return "answer"

    runner = Runner(judge_fn=fake, model="typesafe-ai/jev")
    assert runner.judge(state, QUESTIONS) == "answer"
    assert calls == [(state, QUESTIONS, "typesafe-ai/jev")]


def test_bm25_tokenization_ranking_and_ties_are_deterministic() -> None:
    assert tokenize("Äpfel API_2, API_2!") == ("äpfel", "api_2", "api_2")
    states = (
        State("b", "other", {"file": "none"}),
        State("a", "other", {"file": "none"}),
        State("c", "launch launch", {"file": "decision"}),
    )
    ranked = bm25_rank(states, "launch", ("focus", "context.file"))
    assert [state.state_ref for state in ranked] == ["c", "a", "b"]


def test_prefilter_ranks_before_scan_cap_and_assigns_one_skip_reason() -> None:
    calls: list[str] = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return JudgeResponse({"matches": NoulAnswer(0.9)})

    states = (
        State("a", "irrelevant"),
        State("b", "needle"),
        State("c", "needle"),
        State("d", "irrelevant"),
    )
    result = Runner(judge).run(
        states,
        {"matches": {"type": "noul"}},
        max_chunks=1,
        prefilter={
            "ranker": "bm25",
            "top": 2,
            "fields": ("focus",),
            "query": "needle",
        },
        stdout=io.StringIO(),
    )
    assert calls == ["b"]
    assert result.stats.discovered == 4
    assert result.stats.judged == 1
    assert result.stats.skipped == 3
    skip_kinds = {
        record.to_dict()["error"]["kind"]
        for record in result.records
        if record.to_dict().get("record_type") == "error"
    }
    assert skip_kinds == {"prefiltered", "scan_cap"}
    assert result.records[-1].to_dict()["coverage_reasons"] == [
        "prefiltered",
        "scan_cap",
    ]


@pytest.mark.parametrize(
    "prefilter",
    [
        None,
        {
            "ranker": "bm25",
            "top": 1,
            "fields": ("focus",),
            "query": "needle",
        },
    ],
)
def test_negative_max_chunks_is_rejected_on_both_admission_paths(prefilter) -> None:
    calls = []

    def judge(*args):
        calls.append(args)
        return FakeJudge()(*args)

    with pytest.raises(ValueError, match="max_chunks must be non-negative"):
        Runner(judge).run(
            [State("a", "needle")],
            {"matches": {"type": "noul"}},
            max_chunks=-1,
            prefilter=prefilter,
        )

    assert calls == []


def test_runner_uses_one_validated_preset_for_runtime_values() -> None:
    calls = []
    preset = resolve_preset("jgrep")

    def judge(state_arg, questions_arg, model_arg):
        calls.append((state_arg, questions_arg, model_arg))
        return FakeJudge()(state_arg, questions_arg, model_arg)

    result = Runner(judge).run(
        [State("stdin#L1", "launch")],
        preset=preset,
        chunker="file",
    )

    assert calls[0][1] == preset.questions
    assert calls[0][2] == preset.model
    assert result.records[0].to_dict()["meta"] == {
        "preset": preset.name,
        "preset_version": preset.version,
        "model": preset.model,
        "chunker": "file",
        "cache": "not_applicable",
    }


def test_runner_records_gateway_served_model() -> None:
    def judge(*_args):
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.9)},
            served_model="jev-1.13.0",
        )

    result = Runner(judge).run(
        [State("stdin#L1", "launch")],
        {"matches_query": {"type": "noul"}},
    )

    assert result.records[0].to_dict()["meta"]["model"] == "jev-1.13.0"


@pytest.mark.parametrize("noul", [0.5, 0.9])
def test_runner_gate_sets_failure_exit_for_complete_typed_results(noul: float) -> None:
    def judge(*_):
        return JudgeResponse({"matches_query": NoulAnswer(noul)})

    result = Runner(judge).run_gate(
        [State("stdin#L1", "launch")],
        "any(matches_query.noul >= 0.75)",
        preset="jgrep",
    )

    assert result.gate_result is not None
    assert result.gate_result.failed is (noul == 0.9)
    assert result.exit_code == (1 if noul == 0.9 else 0)
    assert result.records[-1].to_dict()["coverage"] == "complete"


def test_consistency_uses_fresh_uids_and_aggregates_only_noul_answers() -> None:
    questions = {
        "match": {"type": "noul"},
        "kind": {"type": "choice"},
        "risk": {"type": "score"},
    }
    values = iter((0.2, 0.4, 0.6))
    uids: list[str] = []

    def judge(state, *_args):
        uids.append(state.context["uid"])
        return JudgeResponse(
            {
                "match": NoulAnswer(next(values)),
                "kind": ChoiceAnswer("first"),
                "risk": ScoreAnswer(1.0),
            },
            usage={"input_tokens": 10, "output_tokens": 2},
        )

    stderr = io.StringIO()
    result = Runner(judge).run(
        [State("stdin#L1", "launch")],
        questions,
        consistency=3,
        stderr=stderr,
    )

    answers = result.records[0].to_dict()["answers"]
    assert len(uids) == len(set(uids)) == 3
    assert answers["match"]["noul"] == pytest.approx(0.4)
    assert answers["match"]["consistency"]["samples"] == 3
    assert answers["match"]["consistency"]["mean"] == pytest.approx(0.4)
    assert answers["match"]["consistency"]["stddev"] == pytest.approx(
        (0.08 / 3) ** 0.5
    )
    assert "consistency" not in answers["kind"]
    assert "consistency" not in answers["risk"]
    assert result.stats.consistency_usage == {"input_tokens": 30, "output_tokens": 6}
    assert "1 states * 3 = 3 attempted calls" in stderr.getvalue()
    assert "live calls: 3" in stderr.getvalue()


def test_consistency_failure_does_not_emit_a_partial_aggregate() -> None:
    calls = 0

    def judge(*_args):
        nonlocal calls
        calls += 1
        if calls == 2:
            return ErrorResponse("retry budget exhausted")
        return JudgeResponse({"match": NoulAnswer(0.5)})

    result = Runner(judge).run(
        [State("stdin#L1", "launch")],
        {"match": {"type": "noul"}},
        consistency=3,
    )

    assert calls == 3
    assert result.exit_code == 2
    assert result.stats.failed == 1
    assert result.records[0].to_dict()["record_type"] == "error"
    assert "answers" not in result.records[0].to_dict()


def test_consistency_requires_a_noul_question_before_judging() -> None:
    calls = []

    def judge(*args):
        calls.append(args)
        return JudgeResponse({"kind": ChoiceAnswer("yes")})

    with pytest.raises(PresetUsageError, match="at least one Noul"):
        Runner(judge).run(
            [State("stdin#L1", "launch")],
            {"kind": {"type": "choice"}},
            consistency=2,
        )
    assert calls == []


def test_runner_gate_fails_closed_for_partial_results() -> None:
    result = Runner(FakeJudge(mode="incomplete")).run_gate(
        [State("stdin#L1", "launch")],
        "any(matches_query.noul < 0.75)",
        preset="jgrep",
    )

    assert result.exit_code == 2
    assert result.gate_result is not None
    assert result.gate_result.fail_closed is True


def test_runner_gate_fails_closed_for_operational_errors() -> None:
    result = Runner(FakeJudge(mode="error")).run_gate(
        [State("stdin#L1", "launch")],
        "any(matches_query.noul < 0.75)",
        preset="jgrep",
    )

    assert result.exit_code == 2
    assert result.gate_result is not None
    assert result.gate_result.reason == "incomplete coverage"


@pytest.mark.parametrize("reason", ["context_limit", "scan_cap"])
def test_runner_gate_fails_closed_for_unvisited_states(reason: str) -> None:
    result = Runner(FakeJudge()).run_gate(
        [State("stdin#L1", "launch")],
        "any(matches_query.noul < 0.75)",
        preset="jgrep",
        rejections=(StateRejection("stdin#L2", reason, "not visited"),),
    )

    assert result.exit_code == 2
    assert result.gate_result is not None
    assert result.gate_result.fail_closed is True


def test_runner_gate_honors_custom_required_state_count() -> None:
    result = Runner(FakeJudge()).run_gate(
        [State("stdin#L1", "launch")],
        "any(matches_query.noul < 0.75)",
        preset="jgrep",
        require_states=2,
    )

    assert result.exit_code == 2
    assert result.gate_result is not None
    assert result.gate_result.reason == "too few judged states"


def test_runner_validates_policy_before_judging_or_writing() -> None:
    calls = []
    stdout = io.StringIO()

    def judge(*args):
        calls.append(args)
        return FakeJudge()(*args)

    with pytest.raises(PolicySyntaxError) as error:
        Runner(judge).run_gate(
            [State("stdin#L1", "launch")],
            "any(matches_query.noul >=)",
            preset="jgrep",
            stdout=stdout,
        )

    assert error.value.exit_code == 64
    assert calls == []
    assert stdout.getvalue() == ""


def test_runner_rejects_questions_with_a_preset() -> None:
    calls = []

    def judge(*args):
        calls.append(args)
        return FakeJudge()(*args)

    with pytest.raises(PresetUsageError, match="questions"):
        Runner(judge).run(
            [State("stdin#L1", "launch")],
            {"matches": {"type": "noul"}},
            preset="jgrep",
        )

    assert calls == []


def test_runner_rejects_explicit_defaults_for_different_preset_values() -> None:
    preset = resolve_preset("jgrep")
    data = deepcopy(preset.data)
    data["chunking"]["limits"]["context_field_bytes"] = 8_192
    custom_preset = Preset(validate_preset(data), preset.path)
    state = State("stdin#L1", "launch")

    with pytest.raises(PresetUsageError, match="model"):
        Runner(FakeJudge(), model="other-model").run(
            [state], preset=custom_preset
        )
    with pytest.raises(PresetUsageError, match="limits"):
        Runner(FakeJudge(), limits=StateLimits()).run(
            [state], preset=custom_preset
        )

    result = Runner(FakeJudge()).run([state], preset=custom_preset)
    assert result.records[0].to_dict()["meta"]["model"] == "typesafe-ai/jev"


@pytest.mark.parametrize(
    ("runner_kwargs", "run_kwargs", "message"),
    [
        ({"model": "jev-9.9.9"}, {}, "model"),
        ({"limits": StateLimits(focus_bytes=1)}, {}, "limits"),
        ({}, {"preset_version": "2"}, "preset_version"),
        ({}, {"max_chunks": 1}, "max_chunks"),
        ({}, {"chunker": "record"}, "incompatible"),
        ({}, {"chunking": {"by": "line"}}, "chunking"),
    ],
)
def test_runner_rejects_conflicting_loose_preset_values(
    runner_kwargs, run_kwargs, message
) -> None:
    with pytest.raises(PresetUsageError, match=message):
        Runner(FakeJudge(), **runner_kwargs).run(
            [State("stdin#L1", "launch")], preset="jgrep", **run_kwargs
        )


def test_invalid_preset_is_validated_before_processing(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yml"
    path.write_text("schema: jm.preset/v1\n", encoding="utf-8")
    calls = []

    def judge(*args):
        calls.append(args)
        return FakeJudge()(*args)

    with pytest.raises(PresetValidationError):
        Runner(judge).run(
            [State("stdin#L1", "launch")],
            preset=path,
        )

    assert calls == []


def test_invalid_pretty_template_is_validated_before_judging(tmp_path: Path) -> None:
    preset = resolve_preset("jgrep")
    path = tmp_path / "invalid-template.yml"
    content = preset.path.read_text(encoding="utf-8").replace(
        "pretty_template: '{state_ref}\\t{answers.matches_query.noul}'",
        "pretty_template: '{answers.missing.noul}'",
    )
    path.write_text(content, encoding="utf-8")
    calls = []

    def judge(*args):
        calls.append(args)
        return FakeJudge()(*args)

    stdout = io.StringIO()
    with pytest.raises(PresetValidationError, match="pretty_template"):
        Runner(judge).run(
            [State("stdin#L1", "launch")],
            preset=path,
            output_format="pretty",
            stdout=stdout,
            stderr=io.StringIO(),
        )

    assert calls == []
    assert stdout.getvalue() == ""


def test_fake_judge_returns_deterministic_typed_answers() -> None:
    state = State("stdin#L1", "launch", {"source": "stdin"})
    fake = FakeJudge()

    first = fake(state, QUESTIONS, "typesafe-ai/jev")
    second = fake(state, QUESTIONS, "typesafe-ai/jev")

    assert first == second
    assert isinstance(first.answers["is_relevant"], NoulAnswer)
    assert isinstance(first.answers["kind"], ChoiceAnswer)
    assert isinstance(first.answers["risk"], ScoreAnswer)


def test_fake_judge_can_return_incomplete_answers() -> None:
    state = State("stdin#L1", "launch", {"source": "stdin"})
    response = FakeJudge(mode="incomplete")(state, QUESTIONS, "typesafe-ai/jev")

    assert response.complete is False
    assert response.missing_questions == ("risk",)
    assert "risk" not in response.answers


def test_fake_judge_can_return_an_operational_error() -> None:
    state = State("stdin#L1", "launch", {"source": "stdin"})
    first = FakeJudge(mode="error")(state, QUESTIONS, "typesafe-ai/jev")
    second = FakeJudge(mode="error")(state, QUESTIONS, "typesafe-ai/jev")

    assert first == second == ErrorResponse("fake operational error")


def test_runner_admits_states_in_input_order_and_keeps_skipped_refs() -> None:
    states = [State(f"stdin#L{i}", str(i), {"line": i}) for i in range(1, 4)]
    admission = Runner(judge_fn=lambda *_: None).admit(states, max_chunks=2)

    assert isinstance(admission, StateAdmission)
    assert admission.discovered == 3
    assert [state.state_ref for state in admission.admitted] == ["stdin#L1", "stdin#L2"]
    assert [state.state_ref for state in admission.skipped] == ["stdin#L3"]
    assert admission.skip_boundary == "max_chunks=2"


def test_runner_emits_jsonl_then_terminal_coverage_and_flushes_each_record() -> None:
    class FlushCapture(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    stdout = FlushCapture()
    result = Runner(FakeJudge()).run_jsonl(
        [State("stdin#L1", "launch")],
        stdout=stdout,
        preset="jgrep",
        chunker="para",
    )
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert [line["record_type"] for line in lines] == ["result", "coverage"]
    assert lines[-1]["coverage"] == "complete"
    assert result.exit_code == 0
    assert stdout.flush_count == 2


def test_run_jsonl_rejects_questions_with_a_preset() -> None:
    with pytest.raises(PresetUsageError, match="questions"):
        Runner(FakeJudge()).run_jsonl(
            [State("stdin#L1", "launch")],
            {"matches": {"type": "noul"}},
            io.StringIO(),
            preset="jgrep",
        )


def test_runner_emits_partial_result_and_operational_exit() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = Runner(FakeJudge(mode="incomplete")).run(
        [State("stdin#L1", "launch")],
        {"matches": {"type": "noul"}, "risk": {"type": "score"}},
        stdout=stdout,
        stderr=stderr,
        chunker="para",
    )
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert [line["record_type"] for line in lines] == ["partial_result", "coverage"]
    assert lines[0]["missing_questions"] == ["risk"]
    assert lines[0]["meta"]["partial"] is True
    assert lines[1]["coverage_reasons"] == ["partial_answer"]
    assert lines[1]["coverage"] == "partial"
    assert lines[1]["coverage_counts"]["failed"] == 1
    assert result.exit_code == 2
    assert "partial" in stderr.getvalue()


def test_runner_emits_exact_partial_json() -> None:
    def judge(*_):
        return JudgeResponse({"matches": NoulAnswer(0.5)}, ("risk",))

    stdout = io.StringIO()
    result = Runner(judge).run(
        [State("stdin#L1", "launch")],
        {"matches": {"type": "noul"}, "risk": {"type": "score"}},
        stdout=stdout,
        chunker="para",
    )

    assert [json.loads(line) for line in stdout.getvalue().splitlines()] == [
        {
            "record_type": "partial_result",
            "state_ref": "stdin#L1",
            "answers": {"matches": {"type": "noul", "noul": 0.5}},
            "missing_questions": ["risk"],
            "meta": {
                "preset": "jm",
                "preset_version": "1",
                "model": "typesafe-ai/jev",
                "chunker": "para",
                "cache": "not_applicable",
                "partial": True,
            },
        },
        {
            "record_type": "coverage",
            "coverage": "partial",
            "coverage_counts": {
                "discovered": 1,
                "judged": 1,
                "emitted": 1,
                "skipped": 0,
                "failed": 1,
            },
            "coverage_reasons": ["partial_answer"],
            "meta": {
                "preset": "jm",
                "preset_version": "1",
                "model": "typesafe-ai/jev",
                "chunker": "para",
                "cache": "not_applicable",
            },
        },
    ]
    assert result.exit_code == 2


def test_runner_groups_cap_skips_and_keeps_eight_samples() -> None:
    states = [State(f"notes:paragraph={index}", str(index)) for index in range(11)]
    stdout = io.StringIO()
    result = Runner(FakeJudge()).run(
        states,
        {"matches": {"type": "noul"}},
        max_chunks=2,
        stdout=stdout,
        chunker="para",
    )
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    skip = next(line for line in lines if line["record_type"] == "error")
    summary = skip["error"]["skip_summary"]

    assert skip == {
        "record_type": "error",
        "state_ref": None,
        "source_ref": None,
        "error": {
            "kind": "scan_cap",
            "message": "scan cap reached before visit",
            "http_status": None,
            "attempts": 0,
            "skip_summary": {
                "boundary": "max_chunks=2",
                "count": 9,
                "sample_refs": [
                    f"notes:paragraph={index}" for index in range(2, 10)
                ],
            },
        },
        "meta": {
            "preset": "jm",
            "preset_version": "1",
            "model": "typesafe-ai/jev",
            "chunker": "para",
            "cache": "not_applicable",
        },
    }
    assert summary["boundary"] == "max_chunks=2"
    assert summary["count"] == 9
    assert summary["sample_refs"] == [
        f"notes:paragraph={index}" for index in range(2, 10)
    ]
    assert lines[-1]["coverage_counts"] == {
        "discovered": 11,
        "judged": 2,
        "emitted": 2,
        "skipped": 9,
        "failed": 0,
    }
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "mode,max_chunks,rejections,expected_reasons",
    [
        ("complete", None, (), ()),
        ("complete", 1, (), ("scan_cap",)),
        (
            "complete",
            None,
            (
                StateRejection(
                    None, "input_error", "invalid JSON", "stdin:byte=0,line=1"
                ),
            ),
            ("input_error",),
        ),
        (
            "complete",
            None,
            (StateRejection("stdin#L3", "scan_cap", "cap reached"),),
            ("scan_cap",),
        ),
        (
            "complete",
            None,
            (StateRejection("stdin#L3", "context_limit", "too large"),),
            ("context_limit",),
        ),
        ("error", None, (), ("api_error",)),
        ("incomplete", None, (), ("partial_answer",)),
    ],
)
def test_coverage_equations_hold_for_each_run_path(
    mode, max_chunks, rejections, expected_reasons
) -> None:
    states = [State("stdin#L1", "one"), State("stdin#L2", "two")]
    result = Runner(FakeJudge(mode=mode)).run(
        states,
        {"matches": {"type": "noul"}, "risk": {"type": "score"}},
        max_chunks=max_chunks,
        rejections=rejections,
    )
    coverage = result.records[-1].to_dict()
    counts = coverage["coverage_counts"]
    skip_count = sum(
        record.to_dict()["error"]["skip_summary"]["count"]
        for record in result.records
        if record.to_dict().get("record_type") == "error"
        and "skip_summary" in record.to_dict()["error"]
    )

    assert counts["discovered"] == counts["judged"] + counts["skipped"]
    assert counts["skipped"] == skip_count
    assert counts["failed"] <= counts["judged"]
    assert tuple(coverage["coverage_reasons"]) == expected_reasons
    assert result.exit_code == (2 if expected_reasons else 0)


def test_runner_empty_input_emits_input_error_and_partial_coverage() -> None:
    stdout = io.StringIO()
    result = Runner(FakeJudge()).run(
        [],
        {"matches": {"type": "noul"}},
        stdout=stdout,
    )

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines == [
        {
            "record_type": "error",
            "state_ref": None,
            "source_ref": "stdin:byte=0,line=1",
            "error": {
                "kind": "input_error",
                "message": "input is empty",
                "http_status": None,
                "attempts": 0,
            },
            "meta": {
                "preset": "jm",
                "preset_version": "1",
                "model": "typesafe-ai/jev",
                "chunker": "unknown",
                "cache": "not_applicable",
            },
        },
        {
            "record_type": "coverage",
            "coverage": "partial",
            "coverage_counts": {
                "discovered": 0,
                "judged": 0,
                "emitted": 0,
                "skipped": 0,
                "failed": 0,
            },
            "coverage_reasons": ["input_error"],
            "meta": {
                "preset": "jm",
                "preset_version": "1",
                "model": "typesafe-ai/jev",
                "chunker": "unknown",
                "cache": "not_applicable",
            },
        },
    ]
    assert result.exit_code == 2


def test_runner_keeps_interleaved_input_errors_in_input_order() -> None:
    rejections = (
        StateRejection(None, "input_error", "bad first", "stdin:byte=0,line=1"),
        StateRejection("stdin#L2", "context_limit", "too large"),
        StateRejection(None, "input_error", "bad third", "stdin:byte=2,line=3"),
    )
    result = Runner(FakeJudge()).run(
        [],
        {"matches": {"type": "noul"}},
        rejections=rejections,
    )

    rejection_records = [record.to_dict() for record in result.records[:-1]]
    assert [
        (record["error"]["kind"], record.get("source_ref"))
        for record in rejection_records
    ] == [
        ("input_error", "stdin:byte=0,line=1"),
        ("context_limit", None),
        ("input_error", "stdin:byte=2,line=3"),
    ]


def test_runner_keeps_jsonl_on_stdout_and_human_warnings_on_stderr() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    Runner(FakeJudge()).run(
        [State("stdin#L1", "one"), State("stdin#L2", "two")],
        {"matches": {"type": "noul"}},
        max_chunks=1,
        stdout=stdout,
        stderr=stderr,
        chunker="para",
    )

    assert all(
        json.loads(line)["record_type"] for line in stdout.getvalue().splitlines()
    )
    assert "warning" in stderr.getvalue()
    assert not any(line.startswith("{") for line in stderr.getvalue().splitlines())


def test_runner_pretty_output_and_filter_do_not_hide_errors_or_coverage() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    Runner(FakeJudge()).run(
        [State("stdin#L1", "one")],
        {"matches": {"type": "noul"}},
        stdout=stdout,
        stderr=stderr,
        output_format="pretty",
        result_filter=lambda record: False,
        chunker="para",
    )

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [line["record_type"] for line in lines] == ["coverage"]
    assert stderr.getvalue() == ""


def test_runner_pretty_output_without_template_includes_state_ref() -> None:
    stderr = io.StringIO()
    Runner(FakeJudge()).run(
        [State("stdin#L1", "one")],
        {"matches": {"type": "noul"}},
        stderr=stderr,
        output_format="pretty",
        chunker="para",
    )

    state_ref, answers = stderr.getvalue().split("\t", 1)
    assert state_ref == "stdin#L1"
    assert json.loads(answers)["matches"]["type"] == "noul"


def test_state_admission_carries_rejections_in_coverage_counts() -> None:
    rejection = StateRejection("stdin#L2", "context_limit", "too large")
    admission = admit_states(
        (State("stdin#L1", "one"),),
        rejections=(rejection,),
    )
    assert admission.discovered == 2
    assert admission.judged == 1
    assert admission.skipped_count == 1
    assert admission.rejections == (rejection,)


def test_gateway_client_sends_one_full_battery_request(monkeypatch) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "answers": {
                    "is_relevant": {"type": "boolean", "probability": 0.9},
                    "kind": {
                        "type": "choice",
                        "choice": "code",
                        "probabilities": {"code": 1.0},
                    },
                    "risk": {
                        "type": "score",
                        "score": 2,
                        "probabilities": {"2": 1.0},
                    },
                }
            },
            headers={
                "content-type": "application/json",
            },
            request=request,
        )

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = GatewayClient(http_client=client)(
        State("stdin#L1", "focus", {"source": "stdin"}), QUESTIONS, "typesafe-ai/jev"
    )

    assert response.complete
    assert response.answers["is_relevant"] == NoulAnswer(0.9)
    assert response.answers["kind"].confidence == 1.0
    assert response.answers["risk"].legend == {}
    assert response.answers["risk"].confidence == 1.0
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == GATEWAY_ENDPOINT
    assert request.headers["authorization"] == "Bearer test-secret"
    assert request.headers["ai-evaluation-model-specification-version"] == "4"
    assert request.headers["ai-gateway-auth-method"] == "api-key"
    assert request.headers["ai-gateway-protocol-version"] == "0.0.1"
    assert request.headers["ai-model-id"] == GATEWAY_MODEL
    assert json.loads(request.content) == {
        "providerOptions": {"gateway": {"zeroDataRetention": True}},
        "state": {
            "focus": "focus",
            "context": {"source": "stdin", "state_ref": "stdin#L1"},
        },
        "questions": {
            "is_relevant": {"type": "boolean"},
            "kind": {"type": "choice"},
            "risk": {"type": "score"},
        },
    }


def test_gateway_client_retries_timeout_then_succeeds(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=_complete_payload(), request=request)

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = GatewayClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), {"is_relevant": {"type": "noul"}}, "typesafe-ai/jev"
    )

    assert response.complete
    assert attempts == 2


def test_gateway_client_rejects_malformed_success_response(monkeypatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="not json", request=request)
        )
    )

    response = GatewayClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), {}, "typesafe-ai/jev"
    )

    assert response == ErrorResponse(
        "malformed answer", http_status=200, attempts=1
    )


def test_gateway_client_retries_retryable_statuses_and_timeout(
    monkeypatch,
) -> None:
    for status_or_timeout in (429, 529, "timeout"):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if status_or_timeout == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(status_or_timeout, request=request)

        sleeps: list[float] = []
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        response = GatewayClient(
            http_client=client, sleep=sleeps.append, jitter=lambda: 0.0
        )(
            State("stdin#L1", "focus"), QUESTIONS, "typesafe-ai/jev"
        )

        assert attempts == 3
        assert len(sleeps) == 2
        assert response.complete is False
        assert response.http_status in {None, status_or_timeout}


def test_gateway_client_does_not_retry_auth_or_validation_status(monkeypatch) -> None:
    for status in (401, 422):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(status, request=request)

        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        response = GatewayClient(http_client=client, sleep=lambda _: None)(
            State("stdin#L1", "focus"), QUESTIONS, "typesafe-ai/jev"
        )

        assert attempts == 1
        assert response.http_status == status


def test_gateway_client_honors_retry_after(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429, headers={"Retry-After": "7"}, request=request
            )
        return httpx.Response(200, json={"answers": {}}, request=request)

    sleeps: list[float] = []
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = GatewayClient(
        http_client=client, sleep=sleeps.append, jitter=lambda: 99.0
    )(
        State("stdin#L1", "focus"), {}, "typesafe-ai/jev"
    )

    assert response.complete
    assert sleeps == [7.0]


def test_gateway_client_honors_long_retry_after(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429, headers={"Retry-After": "59"}, request=request
            )
        return httpx.Response(200, json={"answers": {}}, request=request)

    sleeps: list[float] = []
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = GatewayClient(
        http_client=client, sleep=sleeps.append, jitter=lambda: 99.0
    )(State("stdin#L1", "focus"), {}, "typesafe-ai/jev")

    assert response.complete
    assert sleeps == [59.0]


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "noul": 0.9},
        {"type": "boolean", "probability": 0.9, "extra": True},
    ],
)
def test_gateway_client_rejects_non_gateway_boolean_shapes(
    monkeypatch, answer
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"answers": {"is_relevant": answer}},
                request=request,
            )
        )
    )

    response = GatewayClient(http_client=client)(
        State("stdin#L1", "focus"), {"is_relevant": {"type": "noul"}}, "typesafe-ai/jev"
    )

    assert response == ErrorResponse(
        "malformed answer", http_status=200, attempts=1
    )


def test_gateway_client_clamps_large_retry_after(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429, headers={"Retry-After": "1e100"}, request=request
            )
        return httpx.Response(200, json={"answers": {}}, request=request)

    sleeps: list[float] = []
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = GatewayClient(
        http_client=client, sleep=sleeps.append, jitter=lambda: 1e100
    )(State("stdin#L1", "focus"), {}, "typesafe-ai/jev")

    assert response.complete
    assert sleeps == [MAX_WAIT_SECONDS]


def test_gateway_client_retries_an_incomplete_full_battery(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        answers = {"is_relevant": {"type": "boolean", "probability": 0.5}}
        if attempts == 2:
            answers["kind"] = {
                "type": "choice",
                "choice": "code",
                "probabilities": {"code": 1.0},
                "confidence": 1.0,
            }
        return httpx.Response(200, json={"answers": answers}, request=request)

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = GatewayClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"),
        {"is_relevant": {"type": "noul"}, "kind": {"type": "choice"}},
        "typesafe-ai/jev",
    )

    assert attempts == 2
    assert response.complete


def test_gateway_client_shares_attempt_budget_across_retries(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=request)
        return httpx.Response(
            200,
            json={"answers": {"is_relevant": {"type": "boolean", "probability": 0.5}}},
            request=request,
        )

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = GatewayClient(
        http_client=client, max_attempts=99, sleep=lambda _: None
    )(
        State("stdin#L1", "focus"),
        {"is_relevant": {"type": "noul"}, "kind": {"type": "choice"}},
        "typesafe-ai/jev",
    )

    assert attempts == 3
    assert response.missing_questions == ("kind",)


def test_gateway_client_enforces_timeout_on_injected_client(monkeypatch) -> None:
    seen_timeouts: list[dict[str, float | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"answers": {}}, request=request)

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(timeout=None, transport=httpx.MockTransport(handler))
    response = GatewayClient(http_client=client, timeout=2.5)(
        State("stdin#L1", "focus"), {}, "typesafe-ai/jev"
    )

    assert response.complete
    assert seen_timeouts == [
        {"connect": 2.5, "read": 2.5, "write": 2.5, "pool": 2.5}
    ]


def test_gateway_client_rejects_oversized_response(monkeypatch) -> None:
    class OversizedStream(httpx.SyncByteStream):
        chunk = b"x" * 4096

        def __init__(self) -> None:
            self.bytes_read = 0

        def __iter__(self):
            while True:
                self.bytes_read += len(self.chunk)
                yield self.chunk

    stream = OversizedStream()

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, stream=stream, request=request)
        assert "Content-Length" not in response.headers
        return response

    client = httpx.Client(
        transport=httpx.MockTransport(handler)
    )

    response = GatewayClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), {}, "typesafe-ai/jev"
    )

    assert isinstance(response, ErrorResponse)
    assert response.error == "response too large"
    assert stream.bytes_read <= MAX_RESPONSE_BYTES + len(stream.chunk)


def test_gateway_client_handles_mid_body_connection_reset(monkeypatch) -> None:
    class ResetStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"answers": '
            raise httpx.ReadError("connection reset")

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=ResetStream(), request=request)
        )
    )

    response = GatewayClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), {}, "typesafe-ai/jev"
    )

    assert response == ErrorResponse("request failed", attempts=1)


def test_gateway_client_returns_missing_ids_after_second_incomplete_response(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "answers": {
                        "is_relevant": {"type": "boolean", "probability": 0.5}
                    }
                },
                request=request,
            )
        )
    )
    response = GatewayClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"),
        {"is_relevant": {"type": "noul"}, "kind": {"type": "choice"}},
        "typesafe-ai/jev",
    )

    assert response.missing_questions == ("kind",)
    assert set(response.answers) == {"is_relevant"}


def test_gateway_client_never_includes_api_key_in_error(monkeypatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "do-not-leak-this")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                401, text="do-not-leak-this", request=request
            )
        )
    )

    response = GatewayClient(http_client=client, sleep=lambda _: None)(
        State("stdin#L1", "focus"), QUESTIONS, "typesafe-ai/jev"
    )

    assert "do-not-leak-this" not in response.error


def test_gateway_client_rejects_moving_model_name(monkeypatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail("moving model must not make a request")
        )
    )

    response = GatewayClient(http_client=client)(
        State("stdin#L1", "focus"), QUESTIONS, "jev-latest"
    )

    assert response == ErrorResponse("model must be typesafe-ai/jev")
