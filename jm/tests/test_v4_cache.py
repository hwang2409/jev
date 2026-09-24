from __future__ import annotations

import json
from pathlib import Path

import httpx

from jm._transport import _GatewayTransport
from jm.answers import JudgeResponse, NoulAnswer, ResultRecord
from jm.cache import CACHE_SCHEMA, CacheStore, battery_hash, cache_key
from jm.client import JevClient, build_canonical_request
from jm.presets import SCHEMA_V3, Preset
from jm.runner import FormationReport, State, judge

QUESTIONS = {
    "match": {
        "type": "noul",
        "instructions": {"state_fields": ["focus", "context.query", "context.heading"]},
        "criteria": {"true": {}, "false": {}},
    }
}


def _preset(name: str = "test") -> Preset:
    return Preset(
        {
            "schema": SCHEMA_V3,
            "name": name,
            "version": "1",
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
    request = build_canonical_request(
        {"focus": "focus", "context": {"query": "q"}},
        QUESTIONS,
    )
    entry = store.publish(
        request.payload["state"],
        _answer(),
        battery=QUESTIONS,
        preset="p",
        preset_version="1",
        configured_model="typesafe-ai/jev",
        transport_identity=request.transport_identity,
        state_ref="ref",
    )
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
    request = build_canonical_request(
        {"focus": "focus", "context": {"query": "q"}}, QUESTIONS
    )
    entry = store.publish(
        request.payload["state"],
        _answer(),
        battery=QUESTIONS,
        preset="p",
        preset_version="1",
        configured_model="typesafe-ai/jev",
        transport_identity=request.transport_identity,
        state_ref="ref",
    )
    second_request = build_canonical_request(
        {"focus": "other", "context": {"query": "q"}}, QUESTIONS
    )
    store.publish(
        second_request.payload["state"],
        _answer(),
        battery=QUESTIONS,
        preset="p",
        preset_version="1",
        configured_model="typesafe-ai/jev",
        transport_identity=second_request.transport_identity,
        state_ref="other",
    )
    assert store.get(entry.cache_key) is not None
    assert next(store.export_triples("p"))["question"]["type"] == "noul"
    assert len(list((tmp_path / "batteries").rglob("*.json"))) == 1


def test_limits_do_not_change_key_when_wire_state_is_unchanged() -> None:
    state = {"focus": "focus", "context": {"query": "q"}}
    keys = []
    for _limits in ({"state_bytes": 100}, {"state_bytes": 1000}):
        request = build_canonical_request(state, QUESTIONS)
        keys.append(
            cache_key(
                {
                    "cache_schema": CACHE_SCHEMA,
                    "wire_request": request.payload,
                    "transport_identity": request.transport_identity,
                }
            )
        )
    assert keys[0] == keys[1]


def test_cross_preset_hit_appends_provenance(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    request = build_canonical_request(
        {"focus": "focus", "context": {"query": "q"}}, QUESTIONS
    )
    first = store.publish(
        request.payload["state"],
        _answer(),
        battery=QUESTIONS,
        preset="first",
        preset_version="1",
        configured_model="typesafe-ai/jev",
        transport_identity=request.transport_identity,
        state_ref="a",
    )
    store.publish(
        request.payload["state"],
        _answer(),
        battery=QUESTIONS,
        preset="second",
        preset_version="1",
        configured_model="typesafe-ai/jev",
        transport_identity=request.transport_identity,
        state_ref="b",
    )
    loaded = store.get(first.cache_key)
    assert loaded is not None
    assert [(group["preset"], group["state_refs"]) for group in loaded.provenance] == [
        ("first", ["a"]),
        ("second", ["b"]),
    ]


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
