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


def test_fixture_is_valid_and_provenance_drift_is_not(tmp_path):
    assert artifacts.is_valid_for_gating(FIXTURE)
    for filename in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
        (tmp_path / filename).write_text((FIXTURE / filename).read_text())
    (tmp_path / "report.md").write_text("synthetic")
    candidates = rows("candidates.jsonl")
    candidates[0]["served_model_id"] = "different-model"
    (tmp_path / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    assert not artifacts.is_valid_for_gating(tmp_path)
