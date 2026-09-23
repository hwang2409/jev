from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from jm.answers import JudgeResponse, NoulAnswer
from jm.cache import CacheStore
from jm.cli import main

ROOT = Path(__file__).parents[1]


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


def _judge(*_args) -> JudgeResponse:
    return JudgeResponse({"matches_query": NoulAnswer(0.9)})


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


def test_concurrency_bounds_requests_and_preserves_output_order(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0

    def judge(state, *_args):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
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
    assert [record["state_ref"] for record in records[:-1]] == [
        "stdin#L1",
        "stdin#L2",
        "stdin#L3",
        "stdin#L4",
    ]


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
