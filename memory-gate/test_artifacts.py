from __future__ import annotations

import json
from pathlib import Path

import artifacts
import pytest

HERE = Path(__file__).parent
FIXTURE = HERE / "runs" / "fixture-dev"


def rows(name):
    return [json.loads(line) for line in (FIXTURE / name).read_text().splitlines() if line]


@pytest.mark.parametrize("name,field,value", [
    ("scores.jsonl", "score", None),
    ("scores.jsonl", "score", 2),
    ("scores.jsonl", "coverage", False),
    ("scores.jsonl", "request_error", "timeout"),
    ("labels.jsonl", "label", "bad"),
])
def test_each_bad_record_invalidates_the_whole_run(tmp_path, name, field, value):
    for filename in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
        (tmp_path / filename).write_text((FIXTURE / filename).read_text())
    (tmp_path / "report.md").write_text("synthetic")
    data = rows(name)
    data[0][field] = value
    (tmp_path / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in data))
    assert not artifacts.is_valid_for_gating(tmp_path)


def test_missing_duplicate_and_extra_relationships_are_invalid(tmp_path):
    for filename in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
        (tmp_path / filename).write_text((FIXTURE / filename).read_text())
    (tmp_path / "report.md").write_text("synthetic")
    scores = rows("scores.jsonl")[:-1]
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores))
    assert not artifacts.is_valid_for_gating(tmp_path)


def copy_fixture(tmp_path):
    for filename in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
        (tmp_path / filename).write_text((FIXTURE / filename).read_text())
    (tmp_path / "report.md").write_text("synthetic")


def test_untruncated_hashes_allow_same_excerpt_with_distinct_sources(tmp_path):
    copy_fixture(tmp_path)
    candidates = rows("candidates.jsonl")
    candidates[1]["presented_excerpt"] = candidates[0]["presented_excerpt"]
    candidates[1]["untruncated_excerpt_hash"] = "b" * 64
    expected = artifacts._canonical_hash([row for row in candidates if row["case_id"] == "case-1"])
    for row in candidates:
        if row["case_id"] == "case-1":
            row["canonical_request_hash"] = expected
    (tmp_path / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    scores = rows("scores.jsonl")
    labels = rows("labels.jsonl")
    for row in scores + labels:
        if row["case_id"] == "case-1":
            row["canonical_request_hash"] = expected
    scores[1]["presented_excerpt"] = candidates[1]["presented_excerpt"]
    scores[1]["untruncated_excerpt_hash"] = "b" * 64
    labels[1]["presented_excerpt"] = candidates[1]["presented_excerpt"]
    labels[1]["untruncated_excerpt_hash"] = "b" * 64
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores))
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    artifacts.validate_run(tmp_path)


@pytest.mark.parametrize("stream,field,value", [
    ("scores.jsonl", "case_id", "wrong"),
    ("labels.jsonl", "query", "wrong"),
    ("labels.jsonl", "path", "wrong.md"),
    ("labels.jsonl", "heading", ["wrong"]),
])
def test_duplicated_candidate_identity_mismatch_invalidates(stream, field, value, tmp_path):
    copy_fixture(tmp_path)
    data = rows(stream)
    data[0][field] = value
    (tmp_path / stream).write_text("".join(json.dumps(row) + "\n" for row in data))
    assert not artifacts.is_valid_for_gating(tmp_path)


def test_stale_production_builder_hash_invalidates(tmp_path):
    copy_fixture(tmp_path)
    candidates = rows("candidates.jsonl")
    candidates[0]["production_builder_hash"] = "0" * 64
    (tmp_path / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    assert not artifacts.is_valid_for_gating(tmp_path)


def test_fixture_is_valid_and_provenance_drift_is_not(tmp_path):
    assert artifacts.is_valid_for_gating(FIXTURE)
    for filename in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
        (tmp_path / filename).write_text((FIXTURE / filename).read_text())
    (tmp_path / "report.md").write_text("synthetic")
    candidates = rows("candidates.jsonl")
    candidates[0]["served_model_id"] = "different-model"
    (tmp_path / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    assert not artifacts.is_valid_for_gating(tmp_path)


def test_score_missing_identity_field_is_rejected(tmp_path):
    copy_fixture(tmp_path)
    score = rows("scores.jsonl")
    del score[0]["path"]
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in score))
    with pytest.raises(artifacts.ArtifactValidationError, match="missing"):
        artifacts.validate_run(tmp_path)


def test_score_wrong_path_is_rejected_with_specific_drift_error(tmp_path):
    copy_fixture(tmp_path)
    score = rows("scores.jsonl")
    score[0]["path"] = "wrong.md"
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in score))
    with pytest.raises(artifacts.ArtifactValidationError, match="candidate drift in path"):
        artifacts.validate_run(tmp_path)


def test_label_wrong_heading_is_rejected_with_specific_drift_error(tmp_path):
    copy_fixture(tmp_path)
    labels = rows("labels.jsonl")
    labels[0]["heading"] = ["wrong"]
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    with pytest.raises(artifacts.ArtifactValidationError, match="candidate drift in heading"):
        artifacts.validate_run(tmp_path)


def test_provenance_drift_serialization_reports_validation_error_not_json_error(tmp_path):
    copy_fixture(tmp_path)
    candidates = rows("candidates.jsonl")
    candidates[0]["served_model_id"] = "different-model"
    (tmp_path / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    with pytest.raises(artifacts.ArtifactValidationError, match="mixed provenance/model identity"):
        artifacts.validate_run(tmp_path)


def test_authoritative_builder_fingerprint_tracks_imported_source(monkeypatch):
    import inspect
    original = artifacts.build_memory_relevance_request

    def substitute(query, candidates):
        return {"query": query, "candidates": candidates, "changed": True}

    monkeypatch.setattr(artifacts, "build_memory_relevance_request", substitute)
    assert artifacts.authoritative_production_builder_hash() != artifacts.sha256(inspect.getsource(original).encode())
