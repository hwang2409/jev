from __future__ import annotations

import copy
import hashlib
import io
import json

import pytest

from jmap.answers import ErrorResponse, JudgeResponse, NoulAnswer, ScoreAnswer
from jmap.cache import (
    CacheStore,
    build_cache_preimage,
    cache_key,
    canonical_json_bytes,
)
from jmap.presets import resolve_preset
from jmap.runner import FakeJudge, Runner, State, StateLimits

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


def _preimage() -> dict[str, object]:
    return build_cache_preimage(
        model="jev-1.13.0",
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
    expected = (
        b'{"cache_schema":"jmap-answer/v1","chunking":{"by":"para",'
        b'"context_paragraphs":0,"limits":{"context_field_bytes":4096,'
        b'"focus_bytes":16384,"state_bytes":32768},"max_chunks":512},'
        b'"model":"jev-1.13.0",'
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
        ("cache_schema", "jmap-answer/v2"),
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


def test_chunking_requires_exact_resolved_fields() -> None:
    with pytest.raises(ValueError, match="chunking"):
        build_cache_preimage(
            model="jev-1.13.0",
            preset="jgrep",
            preset_version="1",
            chunking={"by": "para"},
            questions=QUESTIONS,
            state=State("notes/intro.md#p3", "focus"),
            limits=StateLimits(),
        )

    with pytest.raises(ValueError, match="chunking"):
        build_cache_preimage(
            model="jev-1.13.0",
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
    assert loaded.response == response


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

    assert entry.response == response


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
    payload["cache_schema"] = "jmap-answer/v0"
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
            }
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


def test_export_emits_exact_triples_and_preserves_scores(tmp_path) -> None:
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
    assert all("JEV_API_KEY" not in json.dumps(line) for line in lines)
