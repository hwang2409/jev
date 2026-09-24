from __future__ import annotations

import copy
import hashlib
import io
import json

import pytest

from jm.answers import ErrorResponse, JudgeResponse, NoulAnswer, ScoreAnswer
from jm.cache import (
    CACHE_SCHEMA,
    CacheStore,
    build_cache_preimage,
    cache_key,
    canonical_json_bytes,
)
from jm.presets import resolve_preset
from jm.runner import FakeJudge, Runner, State, StateLimits

QUESTIONS = {
    "matches_query": {
        "type": "noul",
        "instructions": "judge the focus",
        "criteria": {"true": {"what": "direct evidence"}},
    },
    "risk": {
        "type": "score",
        "instructions": "score the risk",
        "criteria": {"0": {"what": "low"}, "3": {"what": "high"}},
    },
}

V3_QUESTIONS = {
    "matches_query": {
        "type": "noul",
        "instructions": {"state_fields": ["focus", "context.query"]},
        "criteria": {"true": {"what": "direct evidence"}},
    }
}


def _preimage() -> dict[str, object]:
    return build_cache_preimage(
        model="typesafe-ai/jev",
        preset="jgrep",
        preset_version="1",
        chunking={"by": "para", "context_paragraphs": 0, "max_chunks": 512},
        questions=QUESTIONS,
        state=State(
            "notes/intro.md#p3",
            "the launch decision was approved",
            {"file": "notes/intro.md", "query": "launch decision"},
        ),
        limits=StateLimits(),
    )


def test_cache_key_has_the_exact_canonical_preimage() -> None:
    preimage = _preimage()
    assert preimage["cache_schema"] == CACHE_SCHEMA
    expected = (
        b'{"cache_schema":"jm-answer/v2","chunking":{"by":"para",'
        b'"context_paragraphs":0,"limits":{"context_field_bytes":4096,'
        b'"focus_bytes":16384,"state_bytes":32768},"max_chunks":512},'
        b'"model":"typesafe-ai/jev",'
        b'"preset":"jgrep","preset_version":"1","question_battery":'
        b'{"matches_query":{"criteria":{"true":{"what":"direct evidence"}},'
        b'"instructions":"judge the focus","type":"noul"},"risk":{"criteria":'
        b'{"0":{"what":"low"},"3":{"what":"high"}},"instructions":"score '
        b'the risk","type":"score"}},"state":{"context":{"file":"notes/'
        b'intro.md","query":"launch decision","state_ref":"notes/intro.md#p3"},'
        b'"focus":"the launch decision was approved"}}'
    )

    assert canonical_json_bytes(preimage) == expected
    assert cache_key(preimage) == "sha256:" + hashlib.sha256(expected).hexdigest()


def test_cache_key_ignores_object_insertion_order() -> None:
    preimage = _preimage()
    reordered = json.loads(json.dumps(preimage, sort_keys=False))
    reordered["chunking"] = {
        "limits": reordered["chunking"]["limits"],
        "context_paragraphs": 0,
        "max_chunks": 512,
        "by": "para",
    }
    assert cache_key(preimage) == cache_key(reordered)


def test_each_key_input_perturbation_changes_the_digest() -> None:
    baseline = _preimage()
    paths = [
        ("cache_schema", "jm-answer/v1"),
        ("model", "jev-2.0.0"),
        ("preset", "jfilter"),
        ("preset_version", "2"),
        ("chunking.by", "line"),
        ("chunking.context_paragraphs", 1),
        ("chunking.max_chunks", 1),
        ("chunking.limits.focus_bytes", 1),
        ("chunking.limits.context_field_bytes", 1),
        ("chunking.limits.state_bytes", 1),
        ("question_battery.matches_query.type", "score"),
        ("question_battery.matches_query.instructions", "changed"),
        ("question_battery.matches_query.criteria.true.what", "changed"),
        ("question_battery.risk.type", "noul"),
        ("question_battery.risk.instructions", "changed"),
        ("question_battery.risk.criteria.0.what", "changed"),
        ("question_battery.risk.criteria.3.what", "changed"),
        ("state.focus", "changed"),
        ("state.context.file", "other.md"),
        ("state.context.query", "other query"),
        ("state.context.state_ref", "other#p3"),
    ]

    for path, value in paths:
        changed = copy.deepcopy(baseline)
        target = changed
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        assert cache_key(changed) != cache_key(baseline), path


def test_consistency_uids_produce_distinct_cache_keys() -> None:
    keys = set()
    for uid in ("one", "two", "three"):
        state = State("notes/intro.md#p3", "focus", {"uid": uid})
        keys.add(
            cache_key(
                build_cache_preimage(
                    model="typesafe-ai/jev",
                    preset="jgrep",
                    preset_version="1",
                    chunking={
                        "by": "para",
                        "context_paragraphs": 0,
                        "max_chunks": 512,
                    },
                    questions=QUESTIONS,
                    state=state,
                    limits=StateLimits(),
                )
            )
        )
    assert len(keys) == 3


def test_distinct_preset_schema_versions_have_distinct_keys() -> None:
    v1 = _preimage()
    v2 = build_cache_preimage(
        model="typesafe-ai/jev",
        preset="jgrep",
        preset_version="1",
        preset_schema="jm.preset/v2",
        chunking={"by": "para", "context_paragraphs": 0, "max_chunks": 512},
        questions=QUESTIONS,
        state=State(
            "notes/intro.md#p3",
            "the launch decision was approved",
            {"file": "notes/intro.md", "query": "launch decision"},
        ),
        limits=StateLimits(),
    )
    assert cache_key(v2) != cache_key(v1)
    assert v2["preset_schema"] == "jm.preset/v2"


def test_chunking_requires_exact_resolved_fields() -> None:
    with pytest.raises(ValueError, match="chunking"):
        build_cache_preimage(
            model="typesafe-ai/jev",
            preset="jgrep",
            preset_version="1",
            chunking={"by": "para"},
            questions=QUESTIONS,
            state=State("notes/intro.md#p3", "focus"),
            limits=StateLimits(),
        )

    with pytest.raises(ValueError, match="chunking"):
        build_cache_preimage(
            model="typesafe-ai/jev",
            preset="jgrep",
            preset_version="1",
            chunking={
                "by": "para",
                "context_paragraphs": 0,
                "max_chunks": 512,
                "future": True,
            },
            questions=QUESTIONS,
            state=State("notes/intro.md#p3", "focus"),
            limits=StateLimits(),
        )


@pytest.mark.parametrize("name", ("jgrep", "jfilter", "diff-risk-heat"))
def test_real_presets_have_cacheable_chunking_and_stable_key_inputs(name: str) -> None:
    preset = resolve_preset(name)
    state = State("stdin#L1", "focus", {"source": "stdin"})
    baseline = build_cache_preimage(
        model=preset.model,
        preset=preset.name,
        preset_version=preset.version,
        chunking=preset.chunking,
        questions=preset.questions,
        state=state,
    )

    changed_version = copy.deepcopy(baseline)
    changed_version["preset_version"] = "2"
    changed_criterion = copy.deepcopy(baseline)
    question_id = next(iter(changed_criterion["question_battery"]))
    question = changed_criterion["question_battery"][question_id]
    if isinstance(question["criteria"], list):
        question["criteria"][0]["what"] += " changed"
    else:
        first_criterion = next(iter(question["criteria"].values()))
        first_criterion["what"] += " changed"

    assert cache_key(changed_version) != cache_key(baseline)
    assert cache_key(changed_criterion) != cache_key(baseline)


def test_cache_store_uses_two_level_paths_and_round_trips_typed_answers(
    tmp_path,
) -> None:
    store = CacheStore(tmp_path)
    response = JudgeResponse(
        {
            "matches_query": NoulAnswer(0.93),
            "risk": ScoreAnswer(1.5, confidence=0.8),
        }
    )
    entry = store.publish(_preimage(), response)
    assert entry.cache_key == cache_key(_preimage())
    assert store.get(cache_key(_preimage()), QUESTIONS) is not None

    path = store.path_for(entry.cache_key)
    assert (
        path
        == tmp_path
        / "answers"
        / entry.cache_key[7:9]
        / entry.cache_key[9:11]
        / f"{entry.cache_key[7:]}.json"
    )
    assert list(path.parent.glob("*.tmp")) == []
    loaded = store.get(entry.cache_key, QUESTIONS)
    assert loaded is not None
    assert loaded.response.answers == response.answers
    assert loaded.response.served_model == "unknown"


def test_v3_cache_round_trip_uses_projected_state_identity(tmp_path) -> None:
    store = CacheStore(tmp_path)
    chunking = {
        "by": "state",
        "limits": {
            "focus_bytes": 16_384,
            "context_field_bytes": 4_096,
            "state_bytes": 32_768,
        },
    }

    first = build_cache_preimage(
        model="typesafe-ai/jev",
        preset="raw",
        preset_version="1",
        preset_schema="jm.preset/v3",
        chunking=chunking,
        questions=V3_QUESTIONS,
        state=State("first", "focus", {"query": "same", "ignored": "one"}),
        limits=StateLimits(),
    )
    equivalent = build_cache_preimage(
        model="typesafe-ai/jev",
        preset="raw",
        preset_version="1",
        preset_schema="jm.preset/v3",
        chunking=chunking,
        questions=V3_QUESTIONS,
        state=State("second", "focus", {"query": "same", "ignored": "two"}),
        limits=StateLimits(),
    )
    response = JudgeResponse({"matches_query": NoulAnswer(0.93)})
    entry = store.publish(first, response)

    assert cache_key(first) == cache_key(equivalent)
    assert store.get(cache_key(equivalent), V3_QUESTIONS) == entry

    changed_parameter = build_cache_preimage(
        model="typesafe-ai/jev",
        preset="raw",
        preset_version="1",
        preset_schema="jm.preset/v3",
        chunking=chunking,
        questions=V3_QUESTIONS,
        state=State("third", "focus", {"query": "changed"}),
        limits=StateLimits(),
    )
    assert store.get(cache_key(changed_parameter), V3_QUESTIONS) is None

    named_ref_questions = {
        "matches_query": {
            **V3_QUESTIONS["matches_query"],
            "instructions": {
                "state_fields": ["focus", "context.state_ref"]
            },
        }
    }
    named_ref_first = build_cache_preimage(
        model="typesafe-ai/jev",
        preset="raw",
        preset_version="1",
        preset_schema="jm.preset/v3",
        chunking=chunking,
        questions=named_ref_questions,
        state=State("first", "focus", {"query": "same"}),
        limits=StateLimits(),
    )
    named_ref_second = build_cache_preimage(
        model="typesafe-ai/jev",
        preset="raw",
        preset_version="1",
        preset_schema="jm.preset/v3",
        chunking=chunking,
        questions=named_ref_questions,
        state=State("second", "focus", {"query": "same"}),
        limits=StateLimits(),
    )
    store.publish(named_ref_first, response)
    assert cache_key(named_ref_first) != cache_key(named_ref_second)
    assert store.get(cache_key(named_ref_second), named_ref_questions) is None


@pytest.mark.parametrize(
    ("names_state_ref", "state_ref_shape", "valid"),
    (
        (True, "string", True),
        (True, "null", False),
        (True, "missing", False),
        (False, "string", False),
    ),
)
def test_v3_cache_validates_named_state_ref_shape(
    tmp_path, names_state_ref, state_ref_shape, valid
) -> None:
    store = CacheStore(tmp_path)
    questions = {
        "matches_query": {
            "type": "noul",
            "instructions": {
                "state_fields": [
                    "focus",
                    "context.state_ref" if names_state_ref else "context.query",
                ]
            },
            "criteria": {"true": {"what": "direct evidence"}},
        }
    }
    preimage = build_cache_preimage(
        model="typesafe-ai/jev",
        preset="raw",
        preset_version="1",
        preset_schema="jm.preset/v3",
        chunking={
            "by": "state",
            "limits": {
                "focus_bytes": 16_384,
                "context_field_bytes": 4_096,
                "state_bytes": 32_768,
            },
        },
        questions=questions,
        state=State("case-1", "focus", {"state_ref": "case-1"}),
        limits=StateLimits(),
    )
    context = preimage["state"]["context"]
    if state_ref_shape == "null":
        context["state_ref"] = None
    elif state_ref_shape == "missing":
        del context["state_ref"]
    elif not names_state_ref:
        context["state_ref"] = "case-1"

    entry = store.publish(
        preimage,
        JudgeResponse({"matches_query": NoulAnswer(0.93)}),
    )
    assert (store.get(entry.cache_key, questions) is not None) is valid


def test_publish_refuses_silently_missing_answers(tmp_path) -> None:
    store = CacheStore(tmp_path)
    response = JudgeResponse({"matches_query": NoulAnswer(0.93)})

    with pytest.raises(ValueError, match="answer"):
        store.publish(_preimage(), response)

    assert list(store.entries()) == []


def test_publish_accepts_matching_answer_ids(tmp_path) -> None:
    store = CacheStore(tmp_path)
    response = JudgeResponse(
        {
            "matches_query": NoulAnswer(0.93),
            "risk": ScoreAnswer(1.5, confidence=0.8),
        }
    )

    entry = store.publish(_preimage(), response)

    assert entry.response.answers == response.answers
    assert entry.response.served_model == "unknown"


def test_cache_preserves_configured_and_served_model_provenance(tmp_path) -> None:
    store = CacheStore(tmp_path)
    preimage = _preimage()
    preimage["model"] = "jev-custom"
    response = JudgeResponse(
        {
            "matches_query": NoulAnswer(0.93),
            "risk": ScoreAnswer(1.5, confidence=0.8),
        },
        served_model="jev-served",
    )

    entry = store.publish(preimage, response)
    loaded = store.get(entry.cache_key, QUESTIONS)

    assert loaded is not None
    assert loaded.model == "jev-custom"
    assert loaded.response.served_model == "jev-served"
    payload = entry.to_dict()
    assert payload["model"] == "jev-custom"
    assert payload["served_model"] == "jev-served"


def test_malformed_and_partial_files_are_cache_misses(tmp_path) -> None:
    store = CacheStore(tmp_path)
    response = JudgeResponse(
        {
            "matches_query": NoulAnswer(0.93),
            "risk": ScoreAnswer(1.5, confidence=0.8),
        }
    )
    entry = store.publish(_preimage(), response)
    path = store.path_for(entry.cache_key)

    path.write_text("not json", encoding="utf-8")
    assert store.get(entry.cache_key, QUESTIONS) is None

    entry = store.publish(_preimage(), response)
    path = store.path_for(entry.cache_key)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["answers"]["risk"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.get(entry.cache_key, QUESTIONS) is None

    entry = store.publish(_preimage(), response)
    path = store.path_for(entry.cache_key)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cache_schema"] = "jm-answer/v0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.get(entry.cache_key, QUESTIONS) is None


def test_failed_response_is_never_published(tmp_path) -> None:
    store = CacheStore(tmp_path)
    runner = Runner(lambda *_: ErrorResponse("failed"))
    runner.run(
        [State("stdin#L1", "focus")],
        QUESTIONS,
        chunker="para",
        cache_store=store,
        chunking={"by": "para", "context_paragraphs": 0, "max_chunks": 512},
    )

    assert list(store.entries()) == []


def test_partial_response_is_never_published(tmp_path) -> None:
    store = CacheStore(tmp_path)
    runner = Runner(
        lambda *_: JudgeResponse({"matches_query": NoulAnswer(0.93)}, ("risk",))
    )
    runner.run(
        [State("stdin#L1", "focus")],
        QUESTIONS,
        chunker="para",
        cache_store=store,
        chunking={"by": "para", "context_paragraphs": 0, "max_chunks": 512},
    )

    assert list(store.entries()) == []


def test_incomplete_runtime_chunking_cannot_reach_cache(tmp_path) -> None:
    store = CacheStore(tmp_path)

    with pytest.raises(ValueError, match="chunking"):
        Runner(FakeJudge()).run(
            [State("stdin#L1", "focus")],
            QUESTIONS,
            chunker="para",
            cache_store=store,
            chunking={"by": "para", "context_paragraphs": 0},
        )

    assert list(store.entries()) == []


def test_implicitly_partial_response_is_never_published(tmp_path) -> None:
    store = CacheStore(tmp_path)
    runner = Runner(
        lambda *_: JudgeResponse({"matches_query": NoulAnswer(0.93)})
    )

    with pytest.raises(ValueError, match="answer"):
        runner.run(
            [State("stdin#L1", "focus")],
            QUESTIONS,
            chunker="para",
            cache_store=store,
            chunking={"by": "para", "context_paragraphs": 0, "max_chunks": 512},
        )

    assert list(store.entries()) == []


def test_runner_replays_a_complete_answer_from_cache(tmp_path) -> None:
    calls = 0

    def judge(*_):
        nonlocal calls
        calls += 1
        return JudgeResponse(
            {
                "matches_query": NoulAnswer(0.93),
                "risk": ScoreAnswer(1.5, confidence=0.8),
            },
            served_model="jev-1.13.0",
        )

    state = State("stdin#L1", "focus")
    store = CacheStore(tmp_path)
    first = Runner(judge).run(
        [state],
        QUESTIONS,
        chunker="para",
        cache_store=store,
        chunking={"by": "para", "context_paragraphs": 0, "max_chunks": 512},
    )
    second = Runner(lambda *_: (_ for _ in ()).throw(AssertionError("cache miss"))).run(
        [state],
        QUESTIONS,
        chunker="para",
        cache_store=store,
        chunking={"by": "para", "context_paragraphs": 0, "max_chunks": 512},
    )

    assert calls == 1
    assert first.records[0].to_dict()["meta"]["cache"] == "miss"
    assert second.records[0].to_dict()["meta"]["cache"] == "hit"
    assert first.records[0].to_dict()["meta"]["model"] == "typesafe-ai/jev"
    assert first.records[0].to_dict()["meta"]["served_model"] == "jev-1.13.0"
    assert second.records[0].to_dict()["meta"]["model"] == "typesafe-ai/jev"
    assert second.records[0].to_dict()["meta"]["served_model"] == "jev-1.13.0"
    assert second.responses[0] == first.responses[0]
    assert second.records[-1].to_dict() == first.records[-1].to_dict()


def test_clear_only_removes_the_requested_preset(tmp_path) -> None:
    store = CacheStore(tmp_path)
    response = JudgeResponse(
        {
            "matches_query": NoulAnswer(0.93),
            "risk": ScoreAnswer(1.5, confidence=0.8),
        }
    )
    store.publish(_preimage(), response)
    other = copy.deepcopy(_preimage())
    other["preset"] = "jfilter"
    store.publish(other, response)

    assert store.clear("jgrep") == 1
    assert [entry.preset for entry in store.entries()] == ["jfilter"]


def test_export_emits_exact_triples_and_preserves_scores(tmp_path, monkeypatch) -> None:
    sentinels = {
        "VERCEL_AI_GATEWAY": "gateway-sentinel",
        "AI_GATEWAY_API_KEY": "api-key-sentinel",
        "VERCEL_JEV_KEY": "jev-key-sentinel",
    }
    for name, value in sentinels.items():
        monkeypatch.setenv(name, value)

    store = CacheStore(tmp_path)
    response = JudgeResponse(
        {
            "matches_query": NoulAnswer(0.93),
            "risk": ScoreAnswer(1.5, confidence=0.8),
        }
    )
    entry = store.publish(_preimage(), response)
    output = io.StringIO()

    assert store.export_jsonl("jgrep", output) == 2
    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    expected_keys = {
        "state",
        "question_id",
        "question",
        "answer",
        "model",
        "preset",
        "preset_version",
        "cache_key",
    }
    assert all(set(line) == expected_keys for line in lines)
    assert {line["question_id"] for line in lines} == {"matches_query", "risk"}
    assert all(line["cache_key"] == entry.cache_key for line in lines)
    assert {line["answer"]["type"] for line in lines} == {"noul", "score"}
    score = next(line["answer"] for line in lines if line["question_id"] == "risk")
    assert score["score"] == 1.5
    assert all("coverage" not in line for line in lines)
    cache_files = list(tmp_path.rglob("*.json"))
    assert cache_files
    for value in sentinels.values():
        assert all(value not in json.dumps(line) for line in lines)
        assert all(
            value not in path.read_text(encoding="utf-8") for path in cache_files
        )
