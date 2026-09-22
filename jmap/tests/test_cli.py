from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

from jmap.answers import JudgeResponse, NoulAnswer
from jmap.cache import CacheStore
from jmap.cli import main

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
        ["jgrep", "launch"],
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
        ["jgrep", "launch", "--by", "line", "--filter=keep", "--format=pretty"],
        judge_fn=_judge,
        cache_store=CacheStore(tmp_path),
    )
    assert code == 0
    assert [record["record_type"] for record in records] == ["result", "coverage"]
    assert records[0]["meta"]["chunker"] == "line"
    assert "stdin#L1" in stderr


def test_incompatible_by_is_usage_error_without_coverage() -> None:
    code, records, stderr = _invoke(
        ["jgrep", "launch", "--by", "record"],
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
        ["jgrep", "launch"], judge_fn=_judge, cache_store=store
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
        [sys.executable, "-m", "jmap", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    script = subprocess.run(
        ["uv", "run", "jmap", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert module.returncode == script.returncode == 0
    assert "jmap run" in module.stdout or "run" in module.stdout
    assert "jmap run" in script.stdout or "run" in script.stdout


def test_module_accepts_file_path_arguments_before_judgment(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("launch decision\n", encoding="utf-8")
    commands = (
        [sys.executable, "-m", "jmap"],
        ["uv", "run", "jmap"],
    )
    for command in commands:
        result = subprocess.run(
            [*command, "jgrep", "launch", "--by", "file", str(path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={
                key: value
                for key, value in os.environ.items()
                if key != "JEV_API_KEY"
            },
        )
        assert result.returncode == 2
        assert "JEV_API_KEY is not set" in result.stderr
        assert "unrecognized arguments" not in result.stderr
