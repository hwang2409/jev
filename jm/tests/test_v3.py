from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from jm.answers import JudgeResponse, NoulAnswer
from jm.cache import CacheStore, build_cache_preimage, cache_key
from jm.cli import main
from jm.presets import SCHEMA_V3, validate_preset
from jm.runner import State

ROOT = Path(__file__).parents[1]


def _preset(tmp_path: Path, *, context_field_bytes: int = 4096) -> Path:
    data = yaml.safe_load(
        (ROOT / "jm" / "presets" / "jgrep.yml").read_text(encoding="utf-8")
    )
    data["schema"] = SCHEMA_V3
    data["chunking"] = {
        "by": "state",
        "max_chunks": 512,
        "limits": {
            "focus_bytes": 16384,
            "context_field_bytes": context_field_bytes,
            "state_bytes": 32768,
        },
    }
    data["compatible_chunkers"] = ["state"]
    data["parameters"] = {"declared": ["predicate", "query"]}
    data["output"]["pretty_template"] = "{state_ref}"
    path = tmp_path / "state.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _invoke(
    preset: Path,
    input_text: str,
    *options: str,
    judge_fn=None,
    cache_store: CacheStore | None = None,
) -> tuple[int, list[dict[str, object]], str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "run",
            "--preset",
            str(preset),
            "--by",
            "state",
            *options,
        ],
        stdin=io.StringIO(input_text),
        stdout=stdout,
        stderr=stderr,
        judge_fn=judge_fn,
        cache_store=cache_store,
    )
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return code, records, stderr.getvalue()


def _answer(*_args) -> JudgeResponse:
    return JudgeResponse({"matches_query": NoulAnswer(0.9)})


def test_v3_raw_state_preserves_ref_and_context(tmp_path: Path) -> None:
    calls = []
    preset = _preset(tmp_path)
    code, records, _ = _invoke(
        preset,
        '{"state_ref":"case-17","focus":"evidence","context":{"heading":"release"}}\n',
        "--param",
        "query=launch",
        judge_fn=lambda state, *_args: (calls.append(state) or _answer()),
    )

    assert code == 0
    assert records[0]["state_ref"] == "case-17"
    assert calls[0].context["heading"] == "release"
    assert calls[0].context["query"] == "launch"
    assert calls[0].context["state_ref"] == "case-17"
    assert calls[0].api_payload == {
        "focus": "evidence",
        "context": {"query": "launch"},
    }


def test_v3_raw_state_input_errors_keep_partial_coverage(tmp_path: Path) -> None:
    preset = _preset(tmp_path, context_field_bytes=8)
    code, records, _ = _invoke(
        preset,
        '{"state_ref":"ok","focus":"evidence","context":{}}\n'
        '{"state_ref":"bad","focus":"evidence"}\n'
        "not json\n"
        '{"state_ref":"large","focus":"evidence","context":{"extra":"123456789"}}\n',
        "--param",
        "query=q",
        judge_fn=_answer,
    )

    assert code == 2
    assert [record["record_type"] for record in records].count("result") == 1
    assert [record["record_type"] for record in records].count("error") == 3
    coverage = records[-1]
    assert coverage["coverage"] == "partial"
    assert coverage["coverage_counts"] == {
        "discovered": 1,
        "judged": 1,
        "emitted": 1,
        "skipped": 0,
        "failed": 0,
    }
    assert coverage["coverage_reasons"] == ["input_error"]


def test_v3_duplicate_refs_fail_before_judge_or_cache(tmp_path: Path) -> None:
    preset = _preset(tmp_path)
    cache = CacheStore(tmp_path / "cache")
    calls = []
    code, records, _ = _invoke(
        preset,
        '{"state_ref":"same","focus":"one","context":{}}\n'
        '{"state_ref":"same","focus":"two","context":{}}\n',
        "--param",
        "query=q",
        judge_fn=lambda *args: (calls.append(args) or _answer()),
        cache_store=cache,
    )

    assert code == 2
    assert calls == []
    assert records[-1]["record_type"] == "coverage"
    assert records[-1]["coverage_reasons"] == ["input_error"]
    assert list((tmp_path / "cache").rglob("*.json")) == []


def test_v3_parameters_validate_before_input_and_aliases_share_path(
    tmp_path: Path,
) -> None:
    preset = _preset(tmp_path)
    for options, message in (
        ((), "missing required parameter 'query'"),
        (("--param", "unknown=value"), "unknown parameter 'unknown'"),
        (("--param", "query=one", "--param", "query=two"), "more than once"),
        (("--param", "query=one", "--query", "two"), "conflicts"),
        (
            ("--param", "query=one", "--predicate", "two", "--param", "predicate=two"),
            "conflicts",
        ),
    ):
        code, records, stderr = _invoke(
            preset,
            "not json\n",
            *options,
            judge_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("judge called")
            ),
        )
        assert code == 64, records
        assert message in stderr
        assert records == []


def test_v3_cache_projection_includes_declared_context_only() -> None:
    questions = {
        "query_match": {
            "type": "noul",
            "instructions": {"state_fields": ["focus", "context.query"]},
        }
    }
    common = {
        "model": "jev-custom",
        "preset": "custom",
        "preset_version": "1",
        "chunking": {
            "by": "state",
            "max_chunks": 512,
            "limits": {
                "focus_bytes": 16384,
                "context_field_bytes": 4096,
                "state_bytes": 32768,
            },
        },
        "questions": questions,
        "preset_schema": SCHEMA_V3,
    }
    base = build_cache_preimage(
        state=State("case", "focus", {"query": "one", "ignored": "a"}),
        **common,
    )
    ignored_change = build_cache_preimage(
        state=State("case", "focus", {"query": "one", "ignored": "b"}),
        **common,
    )
    declared_change = build_cache_preimage(
        state=State("case", "focus", {"query": "two", "ignored": "a"}),
        **common,
    )

    assert cache_key(base) == cache_key(ignored_change)
    assert cache_key(base) != cache_key(declared_change)
    assert base["state"]["context"] == {"query": "one"}


def test_v3_preset_requires_the_parameter_block(tmp_path: Path) -> None:
    preset = yaml.safe_load(
        (ROOT / "jm" / "presets" / "jgrep.yml").read_text(encoding="utf-8")
    )
    preset["schema"] = SCHEMA_V3
    try:
        validate_preset(preset)
    except ValueError as exc:
        assert "parameters" in str(exc)
    else:
        raise AssertionError("missing v3 parameter block was accepted")
