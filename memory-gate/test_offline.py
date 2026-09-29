"""Offline tests for the memory-gate runner; no test opens a network socket."""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("memory_gate_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(run)
FIXTURE = HERE / "runs" / "fixture-dev"


def test_fixture_schema_and_replay():
    run.validate_artifact(FIXTURE)
    candidates = run.read_jsonl(FIXTURE / "candidates.jsonl")
    all_scores = {row["candidate_id"]: row["score"] for row in run.read_jsonl(FIXTURE / "scores.jsonl")}
    labels = {row["candidate_id"]: row["label"] for row in run.read_jsonl(FIXTURE / "labels.jsonl")}
    grouped = {}
    for row in candidates:
        grouped.setdefault(row["case_id"], []).append(row)
    assert {len(v) for v in grouped.values()} <= {1, 2}
    scores = {
        cid: {row["candidate_id"]: all_scores[row["candidate_id"]] for row in rows}
        for cid, rows in grouped.items()
    }
    assert run.select_blocks(grouped["case-1"], scores["case-1"], 0.6)[0]["candidate_id"] == "case-1:candidate-0"
    assert run.metrics([{"case_id": cid, "candidates": rows, "scores": scores[cid], "answerable": True} for cid, rows in grouped.items()], .6, labels)["packet_precision"] == 1.0


def test_candidate_rows_namespaces_ids_and_two_case_generated_shape_validates(tmp_path):
    provenance = {
        "case_set_fingerprint": "cases", "corpus_fingerprint": "corpus",
        "pausanias_revision": "p", "retrieval_config": {"mode": "test"},
        "production_builder_hash": run.artifacts.authoritative_production_builder_hash(),
        "configured_model_id": "model", "served_model_id": "model",
        "harness_revision": "h", "jm_revision": "j",
    }
    cases = [
        {"case_id": "one", "query": "q1", "retrieved": [{"excerpt": "one", "path": "one.md", "heading": []}]},
        {"case_id": "two", "query": "q2", "retrieved": [{"excerpt": "two", "path": "two.md", "heading": []}]},
    ]
    candidates = run.candidate_rows(cases, provenance)
    assert [row["candidate_id"] for row in candidates] == ["one:candidate-0", "two:candidate-0"]
    for row in candidates:
        row["canonical_request_hash"] = run.artifacts._canonical_hash([row])
        row["rank"] = 0
    scores = [{**row, "score": 0.9, "request_error": None, "coverage": True} for row in candidates]
    labels = [{**row, "label": "positive"} for row in candidates]
    run.artifacts.write_artifact(tmp_path, candidates, scores, labels, "report")
    run.validate_artifact(tmp_path)


def test_candidate_rows_preserves_first_whitespace_equivalent_source():
    first = "a" * 600 + " first tail"
    second = "a" * 600 + " second tail"
    case = {"case_id": "case", "query": "q", "retrieved": [
        {"excerpt": first, "path": "one.md", "heading": []},
        {"excerpt": second, "path": "two.md", "heading": []},
    ]}
    rows = run.candidate_rows([case], {"case_set_fingerprint": "x"})
    assert len(rows) == 2
    assert rows[0]["path"] == "one.md"


def test_partial_score_artifact_aborts_metrics():
    candidates = [
        {"candidate_id": "a", "path": "a", "heading": [], "excerpt": "a"},
        {"candidate_id": "b", "path": "b", "heading": [], "excerpt": "b"},
    ]
    with pytest.raises(run.pipeline.ScoreValidationError):
        run.metrics([{"candidates": candidates, "scores": {"a": 0.9}}], 0.6, {})


def test_locomo_category5_filter_and_assertion():
    tiny = [{"id": "conversation-1", "qa": [{"category": 5, "question": "q"}, {"category": 4, "question": "other"}]}]
    assert len(run.filter_locomo_category5(tiny, expected_conversations=1, expected_questions=1)) == 1
    with pytest.raises(ValueError, match="assertion"):
        run.filter_locomo_category5(tiny)


def test_selection_boundaries_and_order():
    rows = [
        {"candidate_id": "a", "path": "a", "heading": [], "excerpt": "x"},
        {"candidate_id": "b", "path": "b", "heading": [], "excerpt": "y"},
    ]
    assert [x["candidate_id"] for x in run.select_blocks(rows, {"a": .6, "b": .9}, .6)] == ["b"]
    assert [x["candidate_id"] for x in run.select_blocks(rows, {"a": .6, "b": .9}, .6, no_gate=True)] == ["a", "b"]


def test_golden_request_equivalence_and_parser():
    """Eval and production are deliberately the same pinned builder/parser."""
    candidates = [{"id": "candidate-0", "excerpt": "SQLite is used."}]
    eval_request = run.production_request("Where is the cache?", candidates)
    # The committed response is adapter wire shape, not a live service response.
    response = {"answers": {"memory_relevance_0": {"noul": .75}}}
    assert eval_request == run._production_adapter()[0]("Where is the cache?", candidates)
    assert run.parse_production_scores(response, candidates) == {"candidate-0": .75}


def test_no_network_guard(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("offline eval attempted network")
    monkeypatch.setattr(socket, "socket", refused)
    run.validate_artifact(FIXTURE)


def git_run(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_lock_requires_exact_published_witness(tmp_path):
    work, bare = tmp_path / "work", tmp_path / "remote.git"
    work.mkdir()
    bare.mkdir()
    git_run(work, "init", "-b", "main")
    git_run(bare, "init", "--bare")
    git_run(work, "remote", "add", "origin", str(bare))
    lock = work / "memory-gate" / "runs" / "x" / "LOCK.json"
    lock.parent.mkdir(parents=True)
    import shutil
    fixture = HERE / "runs" / "fixture-dev"
    for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md"):
        shutil.copy(fixture / name, lock.parent / name)
    payload = run.lock_witness(lock, [lock.parent / name for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")], .6)
    git_run(work, "add", ".")
    git_run(work, "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-m", "lock")
    witness = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work, text=True).strip()
    git_run(work, "push", "origin", "main")
    run.verify_witness(lock, witness, repo=work)
    lock.write_text(lock.read_text() + "tampered")
    with pytest.raises(ValueError, match="exact"):
        run.verify_witness(lock, witness, repo=work)
    lock.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    (work / "memory-gate" / "runs" / "x" / "safety.json").write_text("done")
    with pytest.raises(ValueError, match="already exist"):
        run.verify_witness(lock, witness, repo=work, safety_outputs=[work / "memory-gate" / "runs" / "x" / "safety.json"])
