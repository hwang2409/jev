from __future__ import annotations

import hashlib
import io
import json
import shutil
from copy import deepcopy
from pathlib import Path

import httpx
import yaml

from jm._transport import _GatewayTransport
from jm.answers import JudgeResponse, NoulAnswer, ResultRecord
from jm.cache import CACHE_SCHEMA, CacheStore, battery_hash, cache_key
from jm.cli import main
from jm.client import JevClient, build_canonical_request
from jm.presets import SCHEMA_V3, Preset, resolve_preset
from jm.runner import FormationReport, State, judge

QUESTIONS = {
    "match": {
        "type": "noul",
        "instructions": {"state_fields": ["focus", "context.query", "context.heading"]},
        "criteria": {"true": {}, "false": {}},
    }
}


def _preset(name: str = "test", version: str = "1") -> Preset:
    return Preset(
        {
            "schema": SCHEMA_V3,
            "name": name,
            "version": version,
            "model": "typesafe-ai/jev",
            "chunking": {
                "by": "state",
                "limits": {
                    "focus_bytes": 1000,
                    "context_field_bytes": 1000,
                    "state_bytes": 2000,
                },
            },
            "compatible_chunkers": ["state"],
            "questions": QUESTIONS,
            "thresholds": {},
            "output": {"record_type": "result", "meta": ["cache"]},
        },
        Path("<runtime>"),
    )


def _report() -> FormationReport:
    return FormationReport((), ())


def _answer(*_args: object) -> JudgeResponse:
    return JudgeResponse({"match": NoulAnswer(0.9)})


def _main_answer(*_args: object) -> JudgeResponse:
    return JudgeResponse({"matches_query": NoulAnswer(0.9)})


def _write_main_preset(
    tmp_path: Path,
    *,
    filename: str,
    model: str = "typesafe-ai/jev",
    by: str = "state",
    state_fields: tuple[str, ...] = ("focus",),
    limits: dict[str, int] | None = None,
) -> Path:
    data = deepcopy(resolve_preset("jgrep").data)
    data["schema"] = SCHEMA_V3
    data["name"] = "v4-main-test"
    data["model"] = model
    data["parameters"] = {"declared": []}
    data["chunking"] = {
        "by": by,
        "limits": limits
        or {
            "focus_bytes": 16_384,
            "context_field_bytes": 4_096,
            "state_bytes": 32_768,
        },
    }
    data["compatible_chunkers"] = [by]
    data["questions"]["matches_query"]["instructions"]["state_fields"] = list(
        state_fields
    )
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _run_main(
    argv: list[str],
    input_text: str,
    store: CacheStore,
    judge_fn: object,
) -> tuple[int, list[dict[str, object]], str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        argv,
        stdin=io.StringIO(input_text),
        stdout=stdout,
        stderr=stderr,
        judge_fn=judge_fn,
        cache_store=store,
    )
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return code, records, stderr.getvalue()


def _result_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if record["record_type"] == "result"]


def test_position_and_unnamed_context_do_not_change_key(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    calls: list[State] = []

    def judge_fn(state: State, *_args: object) -> JudgeResponse:
        calls.append(state)
        return _answer()

    states = (
        State("paragraph-1", "same", {"query": "q", "subunit": 1}),
        State("paragraph-2", "same", {"query": "q", "subunit": 2}),
    )
    records = tuple(
        judge(
            _preset(),
            states,
            formation_report=_report(),
            cache_store=store,
            judge_fn=judge_fn,
        )
    )

    assert len(calls) == 1
    assert len([record for record in records if isinstance(record, ResultRecord)]) == 2
    coverage = records[-1]
    assert coverage.coverage_counts["judged"] == 2
    entry = next(store.entries())
    assert entry.provenance[0]["state_refs"] == ["paragraph-1", "paragraph-2"]
    assert "state_ref" not in entry.wire_state["context"]


def test_named_context_and_wire_battery_change_key(tmp_path: Path) -> None:
    state = State("ref", "focus", {"query": "one", "heading": "h"})
    first = build_canonical_request(state, QUESTIONS, model="typesafe-ai/jev")
    changed = build_canonical_request(
        State("other", "focus", {"query": "two", "heading": "h"}),
        QUESTIONS,
        model="typesafe-ai/jev",
    )
    assert cache_key(
        {
            "cache_schema": CACHE_SCHEMA,
            "wire_request": first.payload,
            "transport_identity": first.transport_identity,
        }
    ) != cache_key(
        {
            "cache_schema": CACHE_SCHEMA,
            "wire_request": changed.payload,
            "transport_identity": changed.transport_identity,
        }
    )
    assert battery_hash(QUESTIONS) == battery_hash(dict(QUESTIONS))


def test_named_metadata_changes_cache_key_but_unselected_context_does_not() -> None:
    questions = {
        "match": {
            "type": "noul",
            "instructions": {
                "state_fields": ["focus", "context.metadata"]
            },
            "criteria": {"true": {}, "false": {}},
        }
    }

    def key(state: State) -> str:
        request = build_canonical_request(state, questions, model="typesafe-ai/jev")
        return cache_key(
            {
                "cache_schema": CACHE_SCHEMA,
                "wire_request": request.payload,
                "transport_identity": request.transport_identity,
            }
        )

    base = key(
        State(
            "one",
            "focus",
            {"metadata": {"kind": "payment"}},
            wire_context_keys=frozenset({"metadata"}),
        )
    )
    ignored = key(
        State(
            "two",
            "focus",
            {"metadata": {"kind": "payment"}, "secret": "different"},
            wire_context_keys=frozenset({"metadata"}),
        )
    )
    changed = key(
        State(
            "three",
            "focus",
            {"metadata": {"kind": "refund"}},
            wire_context_keys=frozenset({"metadata"}),
        )
    )

    assert base == ignored
    assert base != changed


def test_model_identity_prevents_cross_model_hits() -> None:
    state = {"focus": "focus", "context": {"query": "q"}}
    first = build_canonical_request(state, QUESTIONS, model="model-a")
    second = build_canonical_request(state, QUESTIONS, model="model-b")
    assert first.request_bytes == second.request_bytes
    assert first.transport_identity != second.transport_identity
    assert cache_key(
        {
            "cache_schema": CACHE_SCHEMA,
            "wire_request": first.payload,
            "transport_identity": first.transport_identity,
        }
    ) != cache_key(
        {
            "cache_schema": CACHE_SCHEMA,
            "wire_request": second.payload,
            "transport_identity": second.transport_identity,
        }
    )


def test_entry_has_no_embedded_battery_or_envelope(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    tuple(
        judge(
            _preset("p"),
            (State("ref", "focus", {"query": "q"}),),
            formation_report=_report(),
            cache_store=store,
            judge_fn=lambda *_args: _answer(),
        )
    )
    entry = next(store.entries())
    payload = json.loads(store.path_for(entry.cache_key).read_text())
    assert set(payload) == {
        "cache_key",
        "wire_state",
        "battery_hash",
        "transport_identity",
        "provenance",
        "configured_model",
        "served_model",
        "usage",
        "created_at",
        "response",
    }
    assert "question_battery" not in payload
    assert "wire_request" not in payload


def test_noul_battery_round_trip_and_export(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    records = tuple(
        judge(
            _preset("p"),
            (
                State("ref", "focus", {"query": "q"}),
                State("other", "other", {"query": "q"}),
            ),
            formation_report=_report(),
            cache_store=store,
            judge_fn=lambda *_args: _answer(),
        )
    )
    assert len([record for record in records if isinstance(record, ResultRecord)]) == 2
    assert next(store.export_triples("p"))["question"]["type"] == "noul"
    assert len(list((tmp_path / "batteries").rglob("*.json"))) == 1


def test_limits_do_not_change_key_when_wire_state_is_unchanged(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    calls: list[str] = []

    def judge_fn(state: State, *_args: object) -> JudgeResponse:
        calls.append(state.state_ref)
        return _answer()

    first = _preset("small")
    second_data = dict(_preset("large").data)
    second_chunking = dict(second_data["chunking"])
    second_chunking["limits"] = {
        "focus_bytes": 2000,
        "context_field_bytes": 2000,
        "state_bytes": 4000,
    }
    second_data["chunking"] = second_chunking
    second = Preset(second_data, Path("<runtime>"))
    for preset, state_ref in ((first, "small-ref"), (second, "large-ref")):
        tuple(
            judge(
                preset,
                (State(state_ref, "focus", {"query": "q"}),),
                formation_report=_report(),
                cache_store=store,
                judge_fn=judge_fn,
            )
        )
    assert calls == ["small-ref"]
    assert len(list(store.entries())) == 1


def test_inserting_a_line_or_paragraph_preserves_unchanged_cache_hits(
    tmp_path: Path,
) -> None:
    for chunker, first_input, second_input in (
        ("line", "unchanged\n", "inserted\nunchanged\n"),
        ("para", "unchanged\n", "inserted\n\nunchanged\n"),
    ):
        store = CacheStore(tmp_path / chunker)
        calls: list[str] = []

        def judge_fn(state: State, *_args: object) -> JudgeResponse:
            calls.append(state.focus)
            return _main_answer()

        first = _run_main(
            ["jgrep", "--query", "unchanged", "--by", chunker],
            first_input,
            store,
            judge_fn,
        )
        second = _run_main(
            ["jgrep", "--query", "unchanged", "--by", chunker],
            second_input,
            store,
            judge_fn,
        )

        assert first[0] == second[0] == 0
        assert first[2] == second[2] == ""
        assert calls == ["unchanged", "inserted"]
        assert [record["meta"]["cache"] for record in _result_records(second[1])] == [
            "miss",
            "hit",
        ]
        assert second[1][-1]["coverage_counts"] == {
            "discovered": 2,
            "judged": 2,
            "emitted": 2,
            "skipped": 0,
            "failed": 0,
        }


def test_limits_only_change_keys_when_the_formed_wire_state_changes(
    tmp_path: Path,
) -> None:
    store = CacheStore(tmp_path / "cache")
    calls: list[str] = []

    def judge_fn(state: State, *_args: object) -> JudgeResponse:
        calls.append(state.focus)
        return _main_answer()

    unchanged_limits = {
        "focus_bytes": 100,
        "context_field_bytes": 1_000,
        "state_bytes": 1_000,
    }
    split_limits = {
        "focus_bytes": 3,
        "context_field_bytes": 1_000,
        "state_bytes": 1_000,
    }
    runs = (
        ("large.yml", unchanged_limits, ["miss"]),
        (
            "larger.yml",
            {
                "focus_bytes": 200,
                "context_field_bytes": 2_000,
                "state_bytes": 2_000,
            },
            ["hit"],
        ),
        ("split.yml", split_limits, ["miss", "miss"]),
        ("large-again.yml", unchanged_limits, ["hit"]),
    )
    input_text = "abcdef\n"

    for filename, limits, expected_cache in runs:
        preset = _write_main_preset(
            tmp_path,
            filename=filename,
            by="line",
            limits=limits,
        )
        result = _run_main(
            [
                "run",
                "--preset",
                str(preset),
                "--by",
                "line",
                "--concurrency",
                "1",
            ],
            input_text,
            store,
            judge_fn,
        )
        assert result[0] == 0
        assert result[2] == ""
        assert [record["meta"]["cache"] for record in _result_records(result[1])] == (
            expected_cache
        )

    assert calls == ["abcdef", "abc", "def"]


def test_line_defaults_are_key_material_when_questions_name_them(
    tmp_path: Path,
) -> None:
    preset = _write_main_preset(
        tmp_path,
        filename="line-context.yml",
        by="line",
        state_fields=("focus", "context.line", "context.surrounding"),
    )
    store = CacheStore(tmp_path / "cache")
    calls: list[tuple[str, int]] = []

    def judge_fn(state: State, *_args: object) -> JudgeResponse:
        calls.append((state.focus, state.context["line"]))
        return _main_answer()

    first = _run_main(
        [
            "run",
            "--preset",
            str(preset),
            "--by",
            "line",
            "--concurrency",
            "1",
        ],
        "same\n",
        store,
        judge_fn,
    )
    second = _run_main(
        [
            "run",
            "--preset",
            str(preset),
            "--by",
            "line",
            "--concurrency",
            "1",
        ],
        "inserted\nsame\n",
        store,
        judge_fn,
    )

    assert first[0] == second[0] == 0
    assert first[2] == second[2] == ""
    assert calls == [("same", 1), ("inserted", 1), ("same", 2)]
    assert [record["meta"]["cache"] for record in _result_records(second[1])] == [
        "miss",
        "miss",
    ]


def test_different_models_do_not_cross_hit_through_main(tmp_path: Path) -> None:
    model_a = _write_main_preset(
        tmp_path,
        filename="model-a.yml",
        model="model-a",
    )
    model_b = _write_main_preset(
        tmp_path,
        filename="model-b.yml",
        model="model-b",
    )
    store = CacheStore(tmp_path / "cache")
    calls: list[str] = []
    input_text = '{"state_ref":"same","focus":"value","context":{}}\n'

    def judge_fn(_state: State, _questions: object, model: str) -> JudgeResponse:
        calls.append(model)
        return _main_answer()

    results = [
        _run_main(
            ["run", "--preset", str(preset), "--by", "state"],
            input_text,
            store,
            judge_fn,
        )
        for preset in (model_a, model_b, model_a)
    ]

    assert [result[0] for result in results] == [0, 0, 0]
    assert [result[2] for result in results] == ["", "", ""]
    assert [
        _result_records(result[1])[0]["meta"]["cache"] for result in results
    ] == ["miss", "miss", "hit"]
    assert calls == ["model-a", "model-b"]


def test_mixed_cache_results_report_exact_coverage_counts(tmp_path: Path) -> None:
    preset = _write_main_preset(tmp_path, filename="mixed.yml")
    store = CacheStore(tmp_path / "cache")
    calls: list[str] = []

    def judge_fn(state: State, *_args: object) -> JudgeResponse:
        calls.append(state.focus)
        return _main_answer()

    first = _run_main(
        [
            "run",
            "--preset",
            str(preset),
            "--by",
            "state",
            "--concurrency",
            "1",
        ],
        '{"state_ref":"first","focus":"alpha","context":{}}\n'
        '{"state_ref":"second","focus":"beta","context":{}}\n',
        store,
        judge_fn,
    )
    second = _run_main(
        [
            "run",
            "--preset",
            str(preset),
            "--by",
            "state",
            "--concurrency",
            "1",
        ],
        '{"state_ref":"renamed","focus":"beta","context":{}}\n'
        '{"state_ref":"new","focus":"gamma","context":{}}\n',
        store,
        judge_fn,
    )

    assert first[0] == second[0] == 0
    assert first[2] == second[2] == ""
    assert calls == ["alpha", "beta", "gamma"]
    assert [record["meta"]["cache"] for record in _result_records(second[1])] == [
        "hit",
        "miss",
    ]
    assert second[1][-1]["coverage_counts"] == {
        "discovered": 2,
        "judged": 2,
        "emitted": 2,
        "skipped": 0,
        "failed": 0,
    }


def test_cross_preset_hit_appends_provenance(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    calls: list[str] = []

    def judge_fn(state: State, *_args: object) -> JudgeResponse:
        calls.append(state.state_ref)
        return _answer()

    first_records = tuple(
        judge(
            _preset("first"),
            (State("a", "focus", {"query": "q"}),),
            formation_report=_report(),
            cache_store=store,
            judge_fn=judge_fn,
        )
    )
    second_records = tuple(
        judge(
            _preset("second"),
            (State("b", "focus", {"query": "q"}),),
            formation_report=_report(),
            cache_store=store,
            judge_fn=judge_fn,
        )
    )
    assert calls == ["a"]
    first_result = next(
        record for record in first_records if isinstance(record, ResultRecord)
    )
    second_result = next(
        record for record in second_records if isinstance(record, ResultRecord)
    )
    assert first_result.to_dict()["meta"]["cache"] == "miss"
    assert second_result.to_dict()["meta"]["cache"] == "hit"
    loaded = next(store.entries("first"))
    assert loaded is not None
    assert [(group["preset"], group["state_refs"]) for group in loaded.provenance] == [
        ("first", ["a"]),
        ("second", ["b"]),
    ]
    exported = list(store.export_triples("second"))
    assert len(exported) == 1
    assert exported[0]["question"]["type"] == "noul"
    battery = store.batteries.get(
        "second", "1", battery_hash(QUESTIONS)
    )
    assert battery is not None
    assert battery.effective == _preset("second").data


def test_entries_reject_a_copy_at_the_wrong_digest_path(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    tuple(
        judge(
            _preset(),
            (State("ref", "focus", {"query": "q"}),),
            formation_report=_report(),
            cache_store=store,
            judge_fn=lambda *_args: _answer(),
        )
    )
    original = next((tmp_path / "answers").rglob("*.json"))
    payload = json.loads(original.read_text())
    wrong = tmp_path / "answers" / "ff" / "ee" / ("f" * 64)
    wrong = wrong.with_suffix(".json")
    wrong.parent.mkdir(parents=True)
    shutil.copyfile(original, wrong)
    original.unlink()
    assert list(store.entries()) == []
    assert store.get(payload["cache_key"]) is None


def test_old_entries_are_invalidated_once(tmp_path: Path) -> None:
    old = tmp_path / "answers" / "aa" / "bb" / ("0" * 64 + ".json")
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"cache_schema": "jm-answer/v2"}))
    CacheStore(tmp_path)
    assert not old.exists()


def test_transmitted_bytes_match_canonical_builder(tmp_path: Path, monkeypatch) -> None:
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(
            200,
            json={"answers": {"match": {"type": "boolean", "probability": 0.9}}},
            request=request,
        )

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "key")
    state = State(
        "ref", "focus", {"query": "q"}, wire_context_keys=frozenset({"query"})
    )
    request = build_canonical_request(state, QUESTIONS)
    client = JevClient(
        _transport=_GatewayTransport(
            http_client=httpx.Client(transport=httpx.MockTransport(handler))
        )
    )
    try:
        client.evaluate(state, QUESTIONS)
    finally:
        client.close()
    assert seen == [request.request_bytes]
    transmitted = json.loads(seen[0])
    derived_key = cache_key(
        {
            "cache_schema": CACHE_SCHEMA,
            "wire_request": transmitted,
            "transport_identity": request.transport_identity,
        }
    )
    assert derived_key == cache_key(
        {
            "cache_schema": CACHE_SCHEMA,
            "wire_request": request.payload,
            "transport_identity": request.transport_identity,
        }
    )
    assert derived_key == "sha256:" + hashlib.sha256(
        json.dumps(
            {
                "cache_schema": CACHE_SCHEMA,
                "wire_request": transmitted,
                "transport_identity": request.transport_identity,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
