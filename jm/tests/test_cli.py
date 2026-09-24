from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import yaml

from jm.answers import ErrorResponse, JudgeResponse, NoulAnswer, ScoreAnswer
from jm.api import GatewayClient
from jm.cache import CacheStore
from jm.cli import main
from jm.runner import BM25CorpusStats, State, bm25_rank, bm25_score, tokenize

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JM_CACHE_DIR", str(tmp_path / "cache"))


class BrokenPipeStream:
    def write(self, _value: str) -> int:
        raise BrokenPipeError(32, "broken pipe")

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class BinaryStdin:
    def __init__(self, value: bytes) -> None:
        self.buffer = io.BytesIO(value)


class ResultEventStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.result_written = threading.Event()

    def write(self, value: str) -> int:
        written = super().write(value)
        if '"record_type":"result"' in value:
            self.result_written.set()
        return written


def _write_v3_state_preset(
    tmp_path: Path,
    *,
    state_fields: list[str],
    prefilter: dict[str, object] | None = None,
) -> Path:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["schema"] = "jm.preset/v3"
    data["chunking"] = {"by": "state", "limits": data["chunking"]["limits"]}
    data["compatible_chunkers"] = ["state"]
    data["parameters"] = {"declared": ["query"]}
    data["questions"]["matches_query"]["instructions"]["state_fields"] = state_fields
    if prefilter is not None:
        data["prefilter"] = prefilter
    path = tmp_path / "v3-preset.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_v3_score_preset(tmp_path: Path) -> Path:
    data = yaml.safe_load(
        (ROOT / "jm" / "presets" / "diff-risk-heat.yml").read_text()
    )
    data["schema"] = "jm.preset/v3"
    data["chunking"] = {
        "by": "state",
        "limits": data["chunking"]["limits"],
    }
    data["compatible_chunkers"] = ["state"]
    data["parameters"] = {"declared": []}
    data["questions"] = {"change_scope": data["questions"]["change_scope"]}
    data["questions"]["change_scope"]["instructions"]["state_fields"] = [
        "focus"
    ]
    data["thresholds"] = {"change_scope": {"type": "score", "fail_at_least": 2}}
    path = tmp_path / "score-preset.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _invoke(
    argv: list[str],
    *,
    input_text: str = "launch decision\n",
    judge_fn=None,
    cache_store: CacheStore | None = None,
) -> tuple[int, list[dict[str, object]], str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        argv,
        judge_fn=judge_fn,
        stdin=io.StringIO(input_text),
        stdout=stdout,
        stderr=stderr,
        cache_store=cache_store,
    )
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return code, records, stderr.getvalue()


def test_sigpipe_returns_quietly_without_a_traceback(tmp_path: Path) -> None:
    stderr = io.StringIO()
    code = main(
        ["jgrep", "--query", "launch"],
        stdin=io.StringIO("launch decision\n"),
        stdout=BrokenPipeStream(),
        stderr=stderr,
        judge_fn=lambda state, questions, model: JudgeResponse(
            {"matches_query": NoulAnswer(0.9)}
        ),
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert "traceback" not in stderr.getvalue().lower()


def test_state_input_rejects_invalid_utf8_and_keeps_valid_lines(tmp_path: Path) -> None:
    preset = _write_v3_state_preset(tmp_path, state_fields=["focus"])
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        ["run", "--preset", str(preset), "--by", "state"],
        stdin=BinaryStdin(
            b'{"state_ref":"first","focus":"ok","context":{}}\n'
            b'{"state_ref":"bad","focus":"bad\xff","context":{}}\n'
            b'{"state_ref":"last","focus":"ok","context":{}}\n'
        ),
        stdout=stdout,
        stderr=stderr,
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert code == 2
    assert {record.get("state_ref") for record in records} >= {"first", "last"}
    assert any(
        record.get("error", {}).get("kind") == "input_error" for record in records
    )
    assert records[-1]["coverage"] == "partial"
    assert records[-1]["coverage_counts"] == {
        "discovered": 2,
        "judged": 2,
        "emitted": 2,
        "skipped": 0,
        "failed": 0,
    }
    assert "\ufffd" not in stdout.getvalue()
    assert "\ufffd" not in stderr.getvalue()


def test_v3_raw_state_unavailable_field_emits_warning_and_continues(
    tmp_path: Path,
) -> None:
    preset = _write_v3_state_preset(
        tmp_path,
        state_fields=["focus", "context.mystery"],
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        ["run", "--preset", str(preset), "--by", "state", "--param", "query=needle"],
        stdin=io.StringIO('{"state_ref":"case-1","focus":"ok","context":{}}\n'),
        stdout=stdout,
        stderr=stderr,
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert code == 0
    assert "code=state_field_unavailable" in stderr.getvalue()
    assert "context.mystery" in stderr.getvalue()
    assert records[-1]["coverage"] == "complete"


def test_v2_prefilter_migration_to_v3_state_preset_runs(tmp_path: Path) -> None:
    preset = _write_v3_state_preset(
        tmp_path,
        state_fields=["focus", "context.query"],
        prefilter={
            "ranker": "bm25",
            "top": 1,
            "query_source": "literal",
            "query": "needle",
            "fields": ["focus"],
        },
    )
    stdout = io.StringIO()
    code = main(
        ["run", "--preset", str(preset), "--by", "state", "--param", "query=needle"],
        stdin=io.StringIO('{"state_ref":"case-1","focus":"needle","context":{}}\n'),
        stdout=stdout,
        stderr=io.StringIO(),
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert code == 0
    assert records[-1]["coverage"] == "complete"
    assert records[-1]["coverage_counts"]["judged"] == 1


def test_v2_prefilter_migration_uses_parameter_query_source(
    tmp_path: Path,
) -> None:
    preset = _write_v3_state_preset(
        tmp_path,
        state_fields=["focus", "context.query"],
        prefilter={
            "ranker": "bm25",
            "top": 1,
            "query_source": "context.query",
            "fields": ["focus"],
        },
    )
    stdout = io.StringIO()
    code = main(
        [
            "run",
            "--preset",
            str(preset),
            "--by",
            "state",
            "--param",
            "query=needle",
        ],
        stdin=io.StringIO('{"state_ref":"case-1","focus":"needle","context":{}}\n'),
        stdout=stdout,
        stderr=io.StringIO(),
        judge_fn=_judge,
    )

    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert code == 0
    assert records[-1]["coverage"] == "complete"
    assert records[-1]["coverage_counts"]["judged"] == 1


def test_duplicate_raw_state_rejection_does_not_read_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preset = _write_v3_state_preset(tmp_path, state_fields=["focus"])
    store = CacheStore(tmp_path / "cache")
    reads = 0
    original_get = store.get

    def counted_get(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_get(*args, **kwargs)

    monkeypatch.setattr(store, "get", counted_get)
    code = main(
        ["run", "--preset", str(preset), "--by", "state"],
        stdin=io.StringIO(
            '{"state_ref":"same","focus":"ok","context":{}}\n'
            '{"state_ref":"same","focus":"ok","context":{}}\n'
        ),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        judge_fn=_judge,
        cache_store=store,
    )

    assert code == 2
    assert reads == 0


def test_sigpipe_gate_returns_zero_without_late_stderr(tmp_path: Path) -> None:
    stderr = io.StringIO()
    code = main(
        [
            "gate",
            "--preset",
            "jgrep",
            "--policy",
            "any(matches_query.noul >= 0.75)",
            "--query",
            "launch",
        ],
        stdin=io.StringIO("launch decision\n"),
        stdout=BrokenPipeStream(),
        stderr=stderr,
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert stderr.getvalue() == ""


def test_sigpipe_partial_judgment_returns_zero_without_late_stderr(
    tmp_path: Path,
) -> None:
    stderr = io.StringIO()
    code = main(
        [
            "jgrep",
            "--query",
            "launch",
            "--prefilter",
            "bm25",
            "--prefilter-top",
            "1",
            "--prefilter-fields",
            "focus",
        ],
        stdin=io.StringIO("launch decision\n\nother text\n\nthird text\n"),
        stdout=BrokenPipeStream(),
        stderr=stderr,
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert stderr.getvalue() == ""


def _judge(*_args) -> JudgeResponse:
    return JudgeResponse({"matches_query": NoulAnswer(0.9)})


def _filter_judge(*_args) -> JudgeResponse:
    return JudgeResponse({"satisfies_predicate": NoulAnswer(0.9)})


@pytest.mark.parametrize("filter_args", [[], ["--filter", "keep"]])
def test_jfilter_input_emission_defaults_to_matching_records(
    tmp_path: Path, filter_args: list[str]
) -> None:
    raw = (
        b'{"id":"one",  "value":"\\u0061", "nested": {"b": 2, "a": 1}}\n'
        b'{"id":"two", "value":"other"}\n'
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    def judge(state: State, *_args: object) -> JudgeResponse:
        score = 0.9 if state.state_ref == "one" else 0.1
        return JudgeResponse({"satisfies_predicate": NoulAnswer(score)})

    code = main(
        ["jfilter", "--predicate", "needle", "--emit=input", *filter_args],
        stdin=BinaryStdin(raw),
        stdout=stdout,
        stderr=stderr,
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert stdout.getvalue() == raw.splitlines(keepends=True)[0].decode()
    assert [
        json.loads(line)["record_type"] for line in stderr.getvalue().splitlines()
    ] == ["coverage"]


@pytest.mark.parametrize("by", ["record", "file"])
@pytest.mark.parametrize("emit_args", [[], ["--emit=input"]])
def test_jfilter_output_modes_keep_judgment_behavior(
    tmp_path: Path, by: str, emit_args: list[str]
) -> None:
    input_text = (
        '{"id":"one","path":"src/one.txt","content":"match"}\n'
        '{"id":"two","path":"src/two.txt","content":"other"}\n'
    )
    calls: list[str] = []
    stdout = io.StringIO()
    stderr = io.StringIO()

    def judge(state: State, *_args: object) -> JudgeResponse:
        calls.append(state.state_ref)
        score = 0.9 if "match" in state.focus else 0.1
        return JudgeResponse({"satisfies_predicate": NoulAnswer(score)})

    code = main(
        [
            "jfilter",
            "--predicate",
            "needle",
            "--by",
            by,
            *emit_args,
        ],
        stdin=io.StringIO(input_text),
        stdout=stdout,
        stderr=stderr,
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / f"{by}-{len(emit_args)}"),
    )

    assert code == 0
    expected_calls = (
        {"one", "two"}
        if by == "record"
        else {"src/one.txt", "src/two.txt"}
    )
    assert set(calls) == expected_calls
    streams = [stdout, stderr] if emit_args else [stdout]
    coverage_lines = [
        json.loads(line)
        for stream in streams
        for line in stream.getvalue().splitlines()
        if line.startswith("{") and '"record_type":"coverage"' in line
    ]
    assert coverage_lines[0]["coverage_counts"] == {
        "discovered": 2,
        "emitted": 2,
        "failed": 0,
        "judged": 2,
        "skipped": 0,
    }


def test_jfilter_input_emission_rejects_output_path(tmp_path: Path) -> None:
    stderr = io.StringIO()
    code = main(
        [
            "jfilter",
            "--predicate",
            "needle",
            "--emit=input",
            "--output",
            str(tmp_path / "results.jsonl"),
        ],
        stdin=io.StringIO('{"id":"one"}\n'),
        stdout=io.StringIO(),
        stderr=stderr,
        judge_fn=_filter_judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 64
    assert "--emit=input cannot be combined with --output" in stderr.getvalue()


def test_jfilter_input_emission_maps_partial_results_to_errors(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    def judge(*_args: object) -> JudgeResponse:
        return JudgeResponse({}, ("satisfies_predicate",))

    code = main(
        ["jfilter", "--predicate", "needle", "--emit=input"],
        stdin=io.StringIO('{"id":"one"}\n'),
        stdout=stdout,
        stderr=stderr,
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    records = [json.loads(line) for line in stderr.getvalue().splitlines()]
    assert code == 2
    assert stdout.getvalue() == ""
    assert [record["record_type"] for record in records] == ["error", "coverage"]
    assert records[0]["error"]["kind"] == "partial_answer"
    assert records[1]["coverage"] == "partial"


@pytest.mark.parametrize("release_ref", ["stdin#L1", "stdin#L2", "stdin#L3"])
def test_judgment_output_streams_before_all_calls_finish(
    tmp_path: Path, release_ref: str
) -> None:
    started = {ref: threading.Event() for ref in ("stdin#L1", "stdin#L2", "stdin#L3")}
    release = {ref: threading.Event() for ref in started}
    stdout = ResultEventStream()
    stderr = io.StringIO()
    result: list[int] = []

    def judge(state: State, *_args: object) -> JudgeResponse:
        started[state.state_ref].set()
        release[state.state_ref].wait(timeout=3)
        return JudgeResponse({"matches_query": NoulAnswer(0.9)})

    def run() -> None:
        result.append(
            main(
                ["jgrep", "--query", "launch", "--by", "line"],
                stdin=io.StringIO("one\ntwo\nthree\n"),
                stdout=stdout,
                stderr=stderr,
                judge_fn=judge,
                cache_store=CacheStore(tmp_path),
            )
        )

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert all(event.wait(timeout=1) for event in started.values())
        release[release_ref].set()
        assert stdout.result_written.wait(timeout=0.4)
    finally:
        for event in release.values():
            event.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert result == [0]


def test_judgment_output_streams_through_real_head_subprocess() -> None:
    script = """
import sys
import threading

from jm.answers import JudgeResponse, NoulAnswer
from jm.cli import main

release = threading.Event()

def judge(state, *_args):
    if state.state_ref == "stdin#L2":
        release.wait(timeout=30)
    return JudgeResponse({"matches_query": NoulAnswer(0.9)})

raise SystemExit(main(
    ["jgrep", "--query", "launch", "--by", "line"],
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr,
    judge_fn=judge,
))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write("one\ntwo\n")
        process.stdin.close()
        first_line = subprocess.run(
            ["head", "-1"],
            stdin=process.stdout,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        assert json.loads(first_line.stdout)["record_type"] == "result"
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=2)


def test_jfilter_file_input_emission_uses_normalized_paths(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "jfilter",
            "--predicate",
            "needle",
            "--by",
            "file",
            "--emit=input",
        ],
        stdin=io.StringIO(
            '{"path":"./src/entry.txt", "content":"needle"}\n'
        ),
        stdout=stdout,
        stderr=stderr,
        judge_fn=_filter_judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert stdout.getvalue() == "src/entry.txt\n"
    assert json.loads(stderr.getvalue().splitlines()[-1])["record_type"] == "coverage"


def test_metrics_are_opt_in_and_coverage_totals_include_filtered_results(
    tmp_path: Path,
) -> None:
    def judge(*_args: object) -> JudgeResponse:
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.9)},
            usage={"input_tokens": 4, "output_tokens": 2, "ignored": "x"},
            latency_ms=7,
        )

    default_code, default_records, _ = _invoke(
        ["jgrep", "--query", "launch", "--filter", "keep"],
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "default"),
    )
    assert default_code == 0
    assert "usage" not in default_records[0]["meta"]
    assert "latency_ms" not in default_records[0]["meta"]
    assert "usage_totals" not in default_records[-1]

    metrics_stdout = io.StringIO()
    metrics_code = main(
        [
            "jgrep",
            "--query",
            "launch",
            "--filter",
            "keep",
            "--metrics",
        ],
        stdin=io.StringIO("launch decision\n"),
        stdout=metrics_stdout,
        stderr=io.StringIO(),
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "metrics"),
    )
    metrics_records = [
        json.loads(line) for line in metrics_stdout.getvalue().splitlines()
    ]
    assert metrics_code == 0
    assert metrics_records[0]["meta"]["usage"] == {
        "input_tokens": 4,
        "output_tokens": 2,
    }
    assert metrics_records[0]["meta"]["latency_ms"] == 7
    assert metrics_records[-1]["usage_totals"] == {
        "input_tokens": 4,
        "output_tokens": 2,
    }
    assert metrics_records[-1]["latency_ms_total"] == 7


def test_output_path_keeps_pretty_rows_on_stderr(tmp_path: Path) -> None:
    output_path = tmp_path / "results.jsonl"
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "jgrep",
            "--query",
            "launch",
            "--format",
            "pretty",
            "--output",
            str(output_path),
        ],
        stdin=io.StringIO("launch decision\n"),
        stdout=stdout,
        stderr=stderr,
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 0
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "stdin#P1\t0.9\n"
    assert [
        json.loads(line)["record_type"]
        for line in output_path.read_text().splitlines()
    ] == ["result", "coverage"]


def test_input_dash_reads_stdin(tmp_path: Path) -> None:
    code, records, _ = _invoke(
        ["jgrep", "--query", "launch", "--input", "-"],
        input_text="launch decision\n",
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert records[-1]["coverage"] == "complete"


def test_short_and_explicit_jgrep_commands_are_equivalent(tmp_path: Path) -> None:
    short_cache = CacheStore(tmp_path / "short")
    explicit_cache = CacheStore(tmp_path / "explicit")
    short = _invoke(
        ["jgrep", "--query", "launch"],
        judge_fn=_judge,
        cache_store=short_cache,
    )
    explicit = _invoke(
        ["run", "--preset", "jgrep.yml", "--query", "launch"],
        judge_fn=_judge,
        cache_store=explicit_cache,
    )
    assert short[0] == explicit[0] == 0
    assert short[1] == explicit[1]


def test_compatible_by_override_and_filter_keep(tmp_path: Path) -> None:
    code, records, stderr = _invoke(
        [
            "jgrep",
            "--query",
            "launch",
            "--by",
            "line",
            "--filter=keep",
            "--format=pretty",
        ],
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path),
    )
    assert code == 0
    assert [record["record_type"] for record in records] == ["result", "coverage"]
    assert records[0]["meta"]["chunker"] == "line"
    assert stderr == "stdin#L1\t0.9\n"


def test_policy_filter_reuses_typed_gate_grammar(tmp_path: Path) -> None:
    code, records, stderr = _invoke(
        [
            "jgrep",
            "--query",
            "launch",
            "--filter",
            "policy",
            "--filter-policy",
            "any(matches_query.noul >= 0.75)",
        ],
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path),
    )
    assert code == 0
    assert [record["record_type"] for record in records] == ["result", "coverage"]
    assert stderr == ""


@pytest.mark.parametrize(
    ("filter_args", "probabilities"),
    [
        (["--filter", "policy", "--filter-policy", "any(change_scope.score >= 2)"], {}),
        (
            ["--filter", "policy", "--filter-policy", "any(change_scope.score >= 2)"],
            {"2.5": 1.0},
        ),
    ],
)
def test_score_filter_malformed_answers_are_operational_errors(
    tmp_path: Path,
    filter_args: list[str],
    probabilities: dict[str, float],
) -> None:
    preset = _write_v3_score_preset(tmp_path)

    def judge(*_args: object) -> JudgeResponse:
        return JudgeResponse(
            {"change_scope": ScoreAnswer(2.0, probabilities=probabilities)}
        )

    code, records, stderr = _invoke(
        [
            "run",
            "--preset",
            str(preset),
            "--by",
            "state",
            *filter_args,
        ],
        input_text='{"state_ref":"case-1","focus":"diff","context":{}}\n',
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 2
    assert [record["record_type"] for record in records] == ["error", "coverage"]
    assert records[0]["error"]["kind"] == "malformed_answer"
    assert records[1]["coverage"] == "partial"
    assert records[1]["coverage_counts"] == {
        "discovered": 1,
        "judged": 1,
        "emitted": 1,
        "skipped": 0,
        "failed": 1,
    }
    assert records[1]["coverage_reasons"] == ["malformed_answer"]
    assert "score answer requires" not in stderr


@pytest.mark.parametrize(
    ("scores", "expected_refs"),
    [
        ([0.9, 0.8, 0.2], ["stdin#L1", "stdin#L2"]),
        ([0.2, 0.3], []),
    ],
)
def test_policy_filter_handles_multiple_matches_and_no_matches(
    tmp_path: Path,
    scores: list[float],
    expected_refs: list[str],
) -> None:
    def judge(state: State, *_args: object) -> JudgeResponse:
        index = int(state.state_ref.rsplit("L", 1)[1]) - 1
        return JudgeResponse({"matches_query": NoulAnswer(scores[index])})

    code, records, _stderr = _invoke(
        [
            "jgrep",
            "--query",
            "launch",
            "--by",
            "line",
            "--filter",
            "policy",
            "--filter-policy",
            "any(matches_query.noul >= 0.75)",
        ],
        input_text="one\ntwo\nthree\n" if len(scores) == 3 else "one\ntwo\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 0
    assert sorted(
        record["state_ref"]
        for record in records
        if record["record_type"] == "result"
    ) == sorted(expected_refs)
    assert records[-1]["record_type"] == "coverage"
    assert records[-1]["coverage"] == "complete"


def test_policy_filter_keeps_mixed_error_records_visible(tmp_path: Path) -> None:
    def judge(state: State, *_args: object) -> JudgeResponse | ErrorResponse:
        if state.state_ref.endswith("L2"):
            return ErrorResponse("temporary failure")
        return JudgeResponse({"matches_query": NoulAnswer(0.9)})

    code, records, _stderr = _invoke(
        [
            "jgrep",
            "--query",
            "launch",
            "--by",
            "line",
            "--filter",
            "policy",
            "--filter-policy",
            "any(matches_query.noul >= 0.75)",
        ],
        input_text="one\ntwo\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 2
    assert {record["record_type"] for record in records} == {
        "result",
        "error",
        "coverage",
    }
    error = next(record for record in records if record["record_type"] == "error")
    assert error["error"]["kind"] == "api_error"
    assert records[-1]["coverage"] == "partial"


def test_real_client_checks_api_key_before_reading_stdin(monkeypatch) -> None:
    class BlockingStdin:
        def read(self):
            raise AssertionError("stdin should not be read")

    monkeypatch.setattr("jm.cli.resolve_gateway_key", lambda: None)
    stderr = io.StringIO()
    code = main(
        ["run", "--preset", "jgrep", "--query", "launch"],
        stdin=BlockingStdin(),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert "Vercel AI Gateway API key is not set" in stderr.getvalue()


def test_jgrep_uses_the_preset_context_paragraph_setting(tmp_path: Path) -> None:
    states = []

    def judge(state, *_args):
        states.append(state)
        return JudgeResponse({"matches_query": NoulAnswer(0.9)})

    code, records, _ = _invoke(
        ["run", "--preset", "jgrep", "--query", "launch"],
        input_text="first\n\nsecond\n\nthird\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert len(records) == 4
    assert [state.context["surrounding"] for state in states] == [[], [], []]


def test_record_metadata_fields_reach_the_chunker(tmp_path: Path) -> None:
    states = []

    def judge(state, *_args):
        states.append(state)
        return _filter_judge(state)

    code, _, _ = _invoke(
        ["jfilter", "--predicate", "failed", "--metadata-fields", "kind"],
        input_text='{"id":"event-1","kind":"payment","secret":"x"}\n',
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert states[0].context["metadata"] == {"kind": "payment"}


def test_line_state_field_warning_names_the_unavailable_key(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["schema"] = "jm.preset/v3"
    data["chunking"] = {"by": "line", "limits": data["chunking"]["limits"]}
    data["compatible_chunkers"] = ["line"]
    data["parameters"] = {"declared": ["query"]}
    data["questions"]["matches_query"]["instructions"]["state_fields"] = [
        "focus",
        "context.heading",
        "context.state_ref",
    ]
    path = tmp_path / "line-preset.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    code, records, stderr = _invoke(
        [
            "run",
            "--preset",
            str(path),
            "--by",
            "line",
            "--param",
            "query=needle",
        ],
        input_text="needle\n",
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 0
    assert records[-1]["coverage"] == "complete"
    assert "context.heading" in stderr
    assert "context.state_ref" not in stderr


def test_oversized_file_is_a_context_limit_skip(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("x" * 16_385, encoding="utf-8")
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    stdout = io.StringIO()
    code = main(
        ["jgrep", "--query", "large", "--by", "file", str(path)],
        stdin=io.StringIO(),
        stdout=stdout,
        stderr=io.StringIO(),
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert code == 2
    assert calls == []
    assert records[-1]["coverage_counts"] == {
        "discovered": 1,
        "judged": 0,
        "emitted": 0,
        "skipped": 1,
        "failed": 0,
    }
    assert records[-1]["coverage_reasons"] == ["context_limit"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "--preset", "jgrep"], "missing required parameter 'query'"),
        (
            ["run", "--preset", "jgrep", "--query", "launch", "--predicate", "p"],
            "unknown parameter 'predicate'",
        ),
        (["run", "--preset", "jfilter"], "missing required parameter 'predicate'"),
        (
            ["run", "--preset", "jfilter", "--predicate", "p", "--query", "q"],
            "unknown parameter 'query'",
        ),
        (
            ["run", "--preset", "diff-risk-heat", "--query", "q"],
            "unknown parameter 'query'",
        ),
    ],
)
def test_preset_parameters_are_validated_before_processing(argv, message) -> None:
    stderr = io.StringIO()
    code = main(
        argv,
        stdin=io.StringIO("input\n"),
        stdout=io.StringIO(),
        stderr=stderr,
        judge_fn=_judge,
    )

    assert code == 64
    assert message in stderr.getvalue()


@pytest.mark.parametrize("limits", [None, "scalar"])
def test_preset_validate_rejects_non_object_limits_without_traceback(
    tmp_path: Path, limits: object
) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["chunking"]["limits"] = limits
    path = tmp_path / "invalid-limits.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    stderr = io.StringIO()

    code = main(
        ["preset", "validate", str(path)],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 64
    assert "chunking.limits must be an object" in stderr.getvalue()
    assert "traceback" not in stderr.getvalue().lower()


def test_invalid_pretty_template_is_usage_error_before_judging(tmp_path: Path) -> None:
    preset = ROOT / "jm" / "presets" / "jgrep.yml"
    path = tmp_path / "invalid-template.yml"
    content = preset.read_text(encoding="utf-8").replace(
        "pretty_template: '{state_ref}\\t{answers.matches_query.noul}'",
        "pretty_template: '{answers.missing.noul}'",
    )
    path.write_text(content, encoding="utf-8")
    calls = []
    stdout = io.StringIO()
    stderr = io.StringIO()

    def judge(*args):
        calls.append(args)
        return _judge(*args)

    code = main(
        ["run", "--preset", str(path), "--query", "launch"],
        judge_fn=judge,
        stdin=io.StringIO("launch\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 64
    assert calls == []
    assert stdout.getvalue() == ""
    assert "pretty_template" in stderr.getvalue()


def test_jgrep_accepts_query_option_without_positional_query(tmp_path: Path) -> None:
    code, records, _ = _invoke(
        ["jgrep", "--query", "launch"],
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert records[-1]["record_type"] == "coverage"


def test_jgrep_consistency_repeats_and_reports_cost(tmp_path: Path) -> None:
    calls = []

    def judge(state, *_args):
        calls.append(state.context["uid"])
        return _judge(state)

    code, records, stderr = _invoke(
        ["jgrep", "--query", "launch", "--consistency", "2"],
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert records[0]["answers"]["matches_query"]["consistency"]["samples"] == 2
    assert "1 states * 2 = 2 attempted calls" in stderr
    assert "cache hits: 0" in stderr
    assert "live calls: 2" in stderr


def test_consistency_metrics_sum_live_latency(tmp_path: Path) -> None:
    def judge(*_args: object) -> JudgeResponse:
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.9)},
            usage={"input_tokens": 4, "output_tokens": 2},
            latency_ms=7,
        )

    code, records, _stderr = _invoke(
        ["jgrep", "--query", "launch", "--consistency", "2", "--metrics"],
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert records[0]["meta"]["latency_ms"] == 14
    assert records[-1]["latency_ms_total"] == 14


def test_main_serializes_consistency_retries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-secret")
    attempts_by_state: dict[str, int] = {}
    active_lock = threading.Lock()
    active = 0
    max_active = 0
    retry_active: list[int] = []
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        state_ref = json.loads(request.content)["state"]["context"]["uid"]
        with active_lock:
            attempt = attempts_by_state.get(state_ref, 0) + 1
            attempts_by_state[state_ref] = attempt
        if attempt == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "answers": {"matches_query": {"type": "boolean", "probability": 0.9}}
            },
            request=request,
        )

    def sleep(delay: float) -> None:
        del delay
        with active_lock:
            retry_active.append(active)

    gateway = GatewayClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleep,
    )

    def judge(state, questions, model):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(state.state_ref)
        try:
            return gateway(state, questions, model)
        finally:
            with active_lock:
                active -= 1

    try:
        code, records, _ = _invoke(
            ["jgrep", "--query", "launch", "--consistency", "2"],
            input_text="first\n\nsecond\n",
            judge_fn=judge,
            cache_store=CacheStore(tmp_path / "cache"),
        )
    finally:
        gateway.close()

    assert code == 0
    assert len(calls) == 4
    assert max_active == 1
    assert retry_active == [1, 1, 1, 1]
    assert len(records) == 3


def test_main_reports_mixed_consistency_cache_counts(
    tmp_path: Path, monkeypatch
) -> None:
    import jm.runner as runner_module

    uids = iter(("first", "second"))
    response_calls = 0

    def repeat_state(state):
        context = dict(state.context)
        context["uid"] = next(uids)
        return State(state.state_ref, state.focus, context)

    monkeypatch.setattr(runner_module, "_repeat_state", repeat_state)
    store = CacheStore(tmp_path / "cache")
    def judge(*_args: object) -> JudgeResponse:
        nonlocal response_calls
        response_calls += 1
        return JudgeResponse(
            {"matches_query": NoulAnswer(0.9)},
            usage={"input_tokens": 4, "output_tokens": 2},
            latency_ms=7,
        )

    first = _invoke(
        ["jgrep", "--query", "launch", "--consistency", "2"],
        judge_fn=judge,
        cache_store=store,
    )
    assert first[0] == 0

    uids = iter(("first", "third"))
    second = _invoke(
        ["jgrep", "--query", "launch", "--consistency", "2", "--metrics"],
        judge_fn=judge,
        cache_store=store,
    )

    assert second[0] == 0
    assert "cache hits: 1" in second[2]
    assert "live calls: 1" in second[2]
    assert response_calls == 3
    assert second[1][0]["meta"]["usage"] == {
        "input_tokens": 8,
        "output_tokens": 4,
    }
    assert second[1][0]["meta"]["latency_ms"] == 7
    assert second[1][-1]["usage_totals"] == {
        "input_tokens": 8,
        "output_tokens": 4,
    }
    assert second[1][-1]["latency_ms_total"] == 7

    uids = iter(("first", "second"))
    cached = _invoke(
        ["jgrep", "--query", "launch", "--consistency", "2", "--metrics"],
        judge_fn=judge,
        cache_store=store,
    )
    assert cached[0] == 0
    assert response_calls == 3
    assert cached[1][0]["meta"]["latency_ms"] == 0
    assert cached[1][-1]["latency_ms_total"] == 0


def test_gate_consistency_is_indeterminate_on_an_interval_edge(
    tmp_path: Path,
) -> None:
    values = iter((0.70, 0.80))

    def judge(*_args):
        return JudgeResponse({"matches_query": NoulAnswer(next(values))})

    code, records, _ = _invoke(
        [
            "gate",
            "--preset",
            "jgrep",
            "--policy",
            "any(matches_query.noul >= 0.75)",
            "--query",
            "launch",
            "--consistency",
            "2",
        ],
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 2
    assert records[0]["answers"]["matches_query"]["consistency"]["stddev"] == (
        pytest.approx(0.05)
    )


@pytest.mark.parametrize(
    ("threshold", "expected_code"),
    [
        ("0.7", 1),
        ("0.7000000000000001", 2),
        ("0.7000000001", 2),
        ("0.8999999999", 2),
        ("0.9", 2),
        ("0.9000000001", 0),
    ],
)
def test_main_gate_consistency_uses_exact_inclusive_interval(
    tmp_path: Path, threshold: str, expected_code: int
) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["thresholds"]["matches_query"]["keep_at_least"] = float(threshold)
    preset_path = tmp_path / "preset.yml"
    preset_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    values = iter((0.75, 0.85))

    def judge(*_args):
        return JudgeResponse({"matches_query": NoulAnswer(next(values))})

    code, _, _ = _invoke(
        [
            "gate",
            "--preset",
            str(preset_path),
            "--policy",
            f"any(matches_query.noul >= {threshold})",
            "--query",
            "launch",
            "--consistency",
            "2",
        ],
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / threshold),
    )

    assert code == expected_code


def test_consistency_without_noul_is_a_usage_error_before_judging(
    tmp_path: Path,
) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    question = data["questions"].pop("matches_query")
    question["type"] = "choice"
    criteria = list(question["criteria"].values())
    question["criteria"] = {
        "yes": criteria[0],
        "no": criteria[1],
    }
    data["questions"]["matches_query"] = question
    data["thresholds"] = {}
    data["output"]["pretty_template"] = "{state_ref}"
    path = tmp_path / "choice.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    calls = []
    stderr = io.StringIO()

    code = main(
        [
            "run",
            "--preset",
            str(path),
            "--query",
            "launch",
            "--consistency",
            "2",
        ],
        judge_fn=lambda *args: calls.append(args),
        stdin=io.StringIO("launch\n"),
        stdout=io.StringIO(),
        stderr=stderr,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 64
    assert calls == []
    assert "at least one Noul" in stderr.getvalue()


def test_context_uid_state_field_is_a_usage_error_before_judging(
    tmp_path: Path,
) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["questions"]["matches_query"]["instructions"]["state_fields"].append(
        "context.uid"
    )
    path = tmp_path / "uid.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    calls = []
    stderr = io.StringIO()

    code = main(
        [
            "run",
            "--preset",
            str(path),
            "--query",
            "launch",
            "--consistency",
            "2",
        ],
        judge_fn=lambda *args: calls.append(args),
        stdin=io.StringIO("launch\n"),
        stdout=io.StringIO(),
        stderr=stderr,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 64
    assert calls == []
    assert "context.uid" in stderr.getvalue()


def test_jgrep_prefilter_emits_recall_warning_and_partial_coverage(
    tmp_path: Path,
) -> None:
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code, records, stderr = _invoke(
        [
            "jgrep",
            "--query",
            "needle",
            "--prefilter",
            "bm25",
            "--prefilter-top",
            "1",
            "--prefilter-fields",
            "focus",
        ],
        input_text="needle here\n\nother text\n\nthird text\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )
    assert code == 2
    assert len(calls) == 1
    assert records[-1]["coverage"] == "partial"
    assert records[-1]["coverage_reasons"] == ["prefiltered"]
    assert (
        "jm: warning: BM25 prefilter skipped 2 of 3 states; recall is bounded "
        "by the shortlist; rerun without --prefilter for full recall"
    ) in stderr


def test_jgrep_prefilter_pins_bm25_document_frequency_and_parameters(
    tmp_path: Path,
) -> None:
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code, _, _ = _invoke(
        [
            "jgrep",
            "--query",
            "needle rare",
            "--prefilter",
            "bm25",
            "--prefilter-top",
            "1",
            "--prefilter-fields",
            "focus",
        ],
        input_text="needle needle\n\nrare\n\nneedle common\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    states = tuple(
        State(state_ref, focus)
        for state_ref, focus in (
            ("p1", "needle needle"),
            ("p2", "rare"),
            ("p3", "needle common"),
        )
    )
    query_tokens = tokenize("needle rare")
    corpus_stats = BM25CorpusStats(
        document_count=3,
        average_length=5 / 3,
        document_frequency={"needle": 2, "rare": 1},
    )
    scores = tuple(
        bm25_score(query_tokens, tokenize(state.focus), corpus_stats)
        for state in states
    )

    assert code == 2
    assert scores == pytest.approx(
        (0.6118390439885317, 1.1727306286009773, 0.4344571362775708)
    )
    ranked_refs = [
        state.state_ref for state in bm25_rank(states, "needle rare", ("focus",))
    ]
    assert ranked_refs == ["p2", "p1", "p3"]
    assert calls == ["stdin#P2"]


def test_prefilter_preset_values_and_gate_rejection_are_command_behaviors(
    tmp_path: Path,
) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["schema"] = "jm.preset/v2"
    data["prefilter"] = {
        "ranker": "bm25",
        "top": 1,
        "query_source": "context.query",
        "fields": ["focus"],
    }
    path = tmp_path / "prefilter.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code, records, _ = _invoke(
        [
            "run",
            "--preset",
            str(path),
            "--query",
            "needle",
        ],
        input_text="needle here\n\nother text\n\nthird text\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )
    assert code == 2
    assert len(calls) == 1
    assert records[-2]["error"] == {
        "kind": "prefiltered",
        "message": "prefilter skipped before visit",
        "http_status": None,
        "attempts": 0,
        "skip_summary": {
            "boundary": "prefilter=bm25,top=1",
            "count": 2,
            "sample_refs": ["stdin#P2", "stdin#P3"],
        },
    }
    assert all(record["meta"]["preset_schema"] == "jm.preset/v2" for record in records)

    override_calls = []

    def override_judge(state, *_args):
        override_calls.append(state.state_ref)
        return _judge(state)

    override_code, _, _ = _invoke(
        [
            "run",
            "--preset",
            str(path),
            "--query",
            "needle",
            "--prefilter-top",
            "2",
        ],
        input_text="needle here\n\nother text\n\nthird text\n",
        judge_fn=override_judge,
        cache_store=CacheStore(tmp_path / "override-cache"),
    )
    assert override_code == 2
    assert len(override_calls) == 2

    gate_stderr = io.StringIO()
    gate_code = main(
        [
            "gate",
            "--preset",
            str(path),
            "--query",
            "needle",
            "--policy",
            "any(matches_query.noul >= 0.75)",
        ],
        judge_fn=judge,
        stdin=io.StringIO("needle\n"),
        stdout=io.StringIO(),
        stderr=gate_stderr,
    )
    assert gate_code == 64
    assert "gate does not support prefiltering" in gate_stderr.getvalue()


def test_prefilter_is_off_without_a_preset_map_or_cli_option(tmp_path: Path) -> None:
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code, records, _ = _invoke(
        ["run", "--preset", "jgrep", "--query", "needle"],
        input_text="needle here\n\nother text\n\nthird text\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert len(calls) == 3
    assert records[-1]["coverage"] == "complete"


def test_jfilter_prefilter_uses_predicate_query_source(tmp_path: Path) -> None:
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _filter_judge(state)

    code, records, _ = _invoke(
        [
            "jfilter",
            "--predicate",
            "needle",
            "--prefilter",
            "bm25",
            "--prefilter-top",
            "1",
            "--prefilter-fields",
            "focus",
        ],
        input_text='{"id":"one","value":"needle"}\n{"id":"two","value":"other"}\n',
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 2
    assert calls == ["one"]
    assert records[-1]["coverage_counts"] == {
        "discovered": 2,
        "judged": 1,
        "emitted": 1,
        "skipped": 1,
        "failed": 0,
    }


def test_generic_run_prefilter_accepts_cli_query_source(tmp_path: Path) -> None:
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code, _, _ = _invoke(
        [
            "run",
            "--preset",
            "jgrep",
            "--query",
            "needle",
            "--prefilter",
            "bm25",
            "--prefilter-top",
            "1",
            "--prefilter-fields",
            "focus",
            "--prefilter-query",
            "needle",
        ],
        input_text="needle here\n\nother text\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 2
    assert calls == ["stdin#P1"]


def test_preset_literal_query_source_ignores_invocation_query(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["schema"] = "jm.preset/v2"
    data["prefilter"] = {
        "ranker": "bm25",
        "top": 1,
        "query_source": "literal",
        "query": "needle",
        "fields": ["focus"],
    }
    path = tmp_path / "literal.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code, _, _ = _invoke(
        ["run", "--preset", str(path), "--query", "other"],
        input_text="needle\n\nother\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 2
    assert calls == ["stdin#P1"]


def test_preset_cli_query_source_requires_and_uses_cli_query(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["schema"] = "jm.preset/v2"
    data["prefilter"] = {
        "ranker": "bm25",
        "top": 1,
        "query_source": "cli",
        "fields": ["focus"],
    }
    path = tmp_path / "cli-query.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code, _, _ = _invoke(
        [
            "run",
            "--preset",
            str(path),
            "--query",
            "other",
            "--prefilter-query",
            "needle",
        ],
        input_text="needle\n\nother\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 2
    assert calls == ["stdin#P1"]


def test_prefilter_field_override_replaces_preset_fields(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["schema"] = "jm.preset/v2"
    data["prefilter"] = {
        "ranker": "bm25",
        "top": 1,
        "query_source": "literal",
        "query": "needle",
        "fields": ["context.query"],
    }
    path = tmp_path / "field-override.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code, _, _ = _invoke(
        [
            "run",
            "--preset",
            str(path),
            "--query",
            "unused",
            "--prefilter-fields",
            "focus",
        ],
        input_text="other\n\nneedle\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 2
    assert calls == ["stdin#P2"]


def test_prefilter_query_override_replaces_preset_query(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "jm" / "presets" / "jgrep.yml").read_text())
    data["schema"] = "jm.preset/v2"
    data["prefilter"] = {
        "ranker": "bm25",
        "top": 1,
        "query_source": "literal",
        "query": "other",
        "fields": ["focus"],
    }
    path = tmp_path / "query-override.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code, _, _ = _invoke(
        [
            "run",
            "--preset",
            str(path),
            "--query",
            "unused",
            "--prefilter-query",
            "needle",
        ],
        input_text="needle\n\nother\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 2
    assert calls == ["stdin#P1"]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ("", "must not be empty"),
        ("focus,focus", "duplicates"),
        ("context.missing", "not in state_fields"),
    ],
)
def test_invalid_cli_prefilter_fields_are_usage_errors(
    fields: str, message: str, tmp_path: Path
) -> None:
    calls = []

    def judge(*args):
        calls.append(args)
        return _judge(*args)

    code, records, stderr = _invoke(
        [
            "jgrep",
            "--query",
            "needle",
            "--prefilter",
            "bm25",
            "--prefilter-top",
            "1",
            "--prefilter-fields",
            fields,
        ],
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 64
    assert records == []
    assert calls == []
    assert message in stderr


@pytest.mark.parametrize("command", ["jgrep", "run"])
def test_negative_max_chunks_is_rejected_before_judging(
    command: str,
) -> None:
    calls = []

    def judge(*args):
        calls.append(args)
        return _judge(*args)

    argv = [command]
    if command == "run":
        argv.extend(["--preset", "jgrep"])
    argv.extend(["--query", "needle", "--max-chunks", "-1"])
    code, records, stderr = _invoke(argv, judge_fn=judge)

    assert code == 64
    assert records == []
    assert calls == []
    assert "non-negative" in stderr


def test_prefilter_rejects_duplicate_state_references_before_judging(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.txt"
    path.write_text("needle\n", encoding="utf-8")
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    stderr = io.StringIO()
    code = main(
        [
            "jgrep",
            "--by",
            "file",
            "--query",
            "needle",
            "--prefilter",
            "bm25",
            "--prefilter-top",
            "1",
            "--prefilter-fields",
            "focus",
            str(path),
            str(path),
        ],
        judge_fn=judge,
        stdin=io.StringIO(),
        stdout=io.StringIO(),
        stderr=stderr,
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 2
    assert calls == []
    assert "duplicate state reference" in stderr.getvalue()


def test_repeated_record_identity_emits_error_and_partial_coverage(
    tmp_path: Path,
) -> None:
    class CountingCacheStore(CacheStore):
        def __init__(self) -> None:
            super().__init__(tmp_path / "cache")
            self.lookups = 0

        def get(self, *args, **kwargs):
            self.lookups += 1
            return super().get(*args, **kwargs)

    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _filter_judge(state)

    cache = CountingCacheStore()
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        ["jfilter", "--predicate", "needle"],
        stdin=io.StringIO(
            '{"id":"same","value":"needle"}\n{"id":"same","value":"other"}\n'
        ),
        stdout=stdout,
        stderr=stderr,
        judge_fn=judge,
        cache_store=cache,
    )

    assert code == 2
    assert calls == []
    assert cache.lookups == 0
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == ["error", "coverage"]
    assert records[0]["error"] == {
        "kind": "input_error",
        "message": "duplicate state reference 'same'",
        "http_status": None,
        "attempts": 0,
    }
    assert records[1]["coverage"] == "partial"
    assert records[1]["coverage_counts"] == {
        "discovered": 0,
        "judged": 0,
        "emitted": 0,
        "skipped": 0,
        "failed": 0,
    }
    assert records[1]["coverage_reasons"] == ["input_error"]
    assert "duplicate state reference 'same'" in stderr.getvalue()


def test_prefilter_ignores_unlisted_context_data(tmp_path: Path) -> None:
    matching_name = tmp_path / "needle.txt"
    nonmatching_name = tmp_path / "other.txt"
    matching_name.write_text("unrelated focus\n", encoding="utf-8")
    nonmatching_name.write_text("needle focus\n", encoding="utf-8")
    calls = []

    def judge(state, *_args):
        calls.append(state.state_ref)
        return _judge(state)

    code = main(
        [
            "jgrep",
            "--by",
            "file",
            "--query",
            "needle",
            "--prefilter",
            "bm25",
            "--prefilter-top",
            "1",
            "--prefilter-fields",
            "focus",
            str(matching_name),
            str(nonmatching_name),
        ],
        judge_fn=judge,
        stdin=io.StringIO(),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert code == 2
    assert calls == [str(nonmatching_name)]


def test_jgrep_positional_argument_is_not_a_query() -> None:
    calls = []

    def judge(*args):
        calls.append(args)
        return _judge(*args)

    code, records, stderr = _invoke(["jgrep", "launch"], judge_fn=judge)

    assert code == 64
    assert records == []
    assert calls == []
    assert "positional input paths require --by file" in stderr


def test_stdin_invalid_utf8_matches_file_input(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    raw = b"ok\xff\n"
    path.write_bytes(raw)
    stdin = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="strict")
    stdin_states = []
    file_states = []

    def stdin_judge(state, *_args):
        stdin_states.append(state)
        return JudgeResponse({"matches_query": NoulAnswer(0.9)})

    def file_judge(state, *_args):
        file_states.append(state)
        return JudgeResponse({"matches_query": NoulAnswer(0.9)})

    assert (
        main(
            ["run", "--preset", "jgrep", "--query", "x", "--by", "line"],
            stdin=stdin,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            judge_fn=stdin_judge,
            cache_store=CacheStore(tmp_path / "stdin-cache"),
        )
        == 0
    )
    assert (
        main(
            [
                "run",
                "--preset",
                "jgrep",
                "--query",
                "x",
                "--by",
                "line",
                "--input",
                str(path),
            ],
            stdin=io.StringIO(),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            judge_fn=file_judge,
            cache_store=CacheStore(tmp_path / "file-cache"),
        )
        == 0
    )
    assert stdin_states[0].focus == file_states[0].focus == "ok\ufffd"


def test_concurrency_bounds_requests_and_emits_completion_order(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0

    def judge(state, *_args):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05 if state.state_ref == "stdin#L1" else 0.01)
        with lock:
            active -= 1
        return JudgeResponse({"matches_query": NoulAnswer(0.9)})

    code, records, _ = _invoke(
        ["jgrep", "--query", "launch", "--by", "line", "--concurrency", "2"],
        input_text="one\ntwo\nthree\nfour\n",
        judge_fn=judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert max_active == 2
    assert records[0]["state_ref"] == "stdin#L2"
    assert {record["state_ref"] for record in records[:-1]} == {
        "stdin#L1",
        "stdin#L2",
        "stdin#L3",
        "stdin#L4",
    }


def test_cli_routes_concurrency_diagnostics_to_stderr(tmp_path: Path) -> None:
    code, _records, stderr = _invoke(
        ["jgrep", "--query", "launch", "--by", "line", "--concurrency", "12"],
        input_text="launch\n",
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path),
    )

    assert code == 0
    assert "code=concurrency_capped" in stderr
    assert "severity=warning" in stderr


def test_incompatible_by_is_usage_error_without_coverage() -> None:
    code, records, stderr = _invoke(
        ["jgrep", "--query", "launch", "--by", "record"],
        judge_fn=_judge,
    )
    assert code == 64
    assert records == []
    assert "allowed set: [line, para, file]" in stderr


def test_gate_uses_require_states_and_jsonl_stdout() -> None:
    code, records, stderr = _invoke(
        [
            "gate",
            "--preset",
            "jgrep",
            "--query",
            "launch",
            "--policy",
            "any(matches_query.noul >= 0.75)",
            "--require-states",
            "1",
        ],
        judge_fn=_judge,
    )
    assert code == 1
    assert records[-1]["record_type"] == "coverage"
    assert all(isinstance(record, dict) for record in records)
    assert stderr == ""


def test_empty_hunk_gate_has_zero_complete_coverage_and_vacuous_truth() -> None:
    code, records, stderr = _invoke(
        [
            "gate",
            "--preset",
            "diff-risk-heat",
            "--by",
            "hunk",
            "--policy",
            "any(change_scope.score >= 2)",
            "--require-states",
            "0",
        ],
        input_text="",
        judge_fn=_judge,
    )

    assert code == 0
    assert records[-1]["coverage"] == "complete"
    assert records[-1]["coverage_counts"] == {
        "discovered": 0,
        "judged": 0,
        "emitted": 0,
        "skipped": 0,
        "failed": 0,
    }
    assert stderr == "jm: no hunks found\n"


def test_preset_and_cache_commands_have_non_judgment_stdout(tmp_path: Path) -> None:
    stdout = io.StringIO()
    assert main(["preset", "list"], stdout=stdout, stderr=io.StringIO()) == 0
    names = {json.loads(line)["name"] for line in stdout.getvalue().splitlines()}
    assert names == {"jfilter", "jgrep", "diff-risk-heat"}

    store = CacheStore(tmp_path)
    code, _, _ = _invoke(
        ["jgrep", "--query", "launch"], judge_fn=_judge, cache_store=store
    )
    assert code == 0
    exported = io.StringIO()
    assert (
        main(
            ["cache", "export", "--preset", "jgrep"],
            stdout=exported,
            stderr=io.StringIO(),
            cache_store=store,
        )
        == 0
    )
    assert len(exported.getvalue().splitlines()) == 1
    cleared = io.StringIO()
    assert (
        main(
            ["cache", "clear", "--preset", "jgrep"],
            stdout=cleared,
            stderr=io.StringIO(),
            cache_store=store,
        )
        == 0
    )
    assert json.loads(cleared.getvalue())["removed"] == 1


def test_module_and_script_help_are_available() -> None:
    environment = os.environ.copy()
    module = subprocess.run(
        [sys.executable, "-m", "jm", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    script = subprocess.run(
        ["uv", "run", "jm", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert module.returncode == script.returncode == 0
    assert "jm run" in module.stdout or "run" in module.stdout
    assert "jm run" in script.stdout or "run" in script.stdout


def test_module_accepts_file_path_arguments_before_judgment(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("launch decision\n", encoding="utf-8")
    commands = (
        [sys.executable, "-m", "jm"],
        ["uv", "run", "jm"],
    )
    for command in commands:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "VERCEL_AI_GATEWAY",
                "AI_GATEWAY_API_KEY",
                "VERCEL_JEV_KEY",
            }
        }
        environment["HOME"] = str(tmp_path)
        result = subprocess.run(
            [*command, "jgrep", "--query", "launch", "--by", "file", str(path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode == 2
        assert "Vercel AI Gateway API key is not set" in result.stderr
        assert "unrecognized arguments" not in result.stderr
