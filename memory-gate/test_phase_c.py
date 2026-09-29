from __future__ import annotations

import json

import pytest

import run


def test_fake_client_scoring_records_complete_coverage_and_cache(tmp_path):
    calls = []

    def fake(request):
        calls.append(request)
        return {"answers": {f"memory_relevance_{i}": {"noul": .7 + i / 10}
                              for i in range(len(request["state"]["memory_candidates"]))},
                "configured_model": "m", "served_model": "m"}

    cases = [{"case_id": "c", "query": "q", "retrieved": [
        {"path": "a.md", "heading": [], "excerpt": "a"},
        {"path": "b.md", "heading": [], "excerpt": "b"},
    ]}]
    first = run.score_cases(cases, fake, model="m", cache_dir=tmp_path)
    second = run.score_cases(cases, fake, model="m", cache_dir=tmp_path)
    assert [row["score"] for row in first] == pytest.approx([.7, .8])
    assert second == first
    assert len(calls) == 1
    assert all(row["coverage"] for row in first)


def test_repeatability_identity_and_crossing_fraction():
    rows = [{"candidate_id": "a", "score": score, "configured_model_id": "m",
             "served_model_id": "m"} for score in (.5, .6, .7)]
    replicates = [{"configured_model_id": "m", "served_model_id": "m", "scores": [row]}
                  for row in rows]
    result = run.repeatability(replicates, .6)
    assert result["per_candidate"]["a"]["stddev"] == pytest.approx((1 / 150) ** .5)
    assert result["per_candidate"]["a"]["worst_spread"] == pytest.approx(.2)
    assert result["fraction_crossing_tau"] == 1
    with pytest.raises(ValueError, match="mixed"):
        run.repeatability([replicates[0], {**replicates[1], "served_model_id": "other"}], .6)


def test_repeatability_refuses_none_identity_in_tuples():
    """Fix 3: (None, None) identity tuples must be detected — the old predicate
    tested `None in {(tuple,...)}` which is always False for tuples."""
    row_missing = {"candidate_id": "a", "score": .5}  # no configured/served keys -> .get returns None
    replicate = {"configured_model_id": None, "served_model_id": None,
                 "scores": [row_missing]}
    with pytest.raises(ValueError, match="missing or mixed"):
        run.repeatability([replicate], .6)


def test_repeatability_refuses_two_served_models_within_one_replicate():
    """Fix 3: mixed model identities within a single replicate are refused."""
    rows = [
        {"candidate_id": "a", "score": .5, "configured_model_id": "m", "served_model_id": "gateway-1"},
        {"candidate_id": "b", "score": .6, "configured_model_id": "m", "served_model_id": "gateway-2"},
    ]
    replicate = {"configured_model_id": "m", "served_model_id": "gateway-1", "scores": rows}
    with pytest.raises(ValueError, match="mixed"):
        run.repeatability([replicate], .6)
def test_frozen_safety_wilson_and_non_authoritative_curve(tmp_path):
    path = tmp_path / "safety.json"
    result = run.freeze_safety(path, witness_commit="abc", tau=.6,
                               false_injections=1, total=446)
    assert result["point_estimate"] == pytest.approx(1 / 446)
    assert result["wilson_95"][1] == pytest.approx(.01258969, abs=1e-5)
    assert result["accepts"] is True
    with pytest.raises(FileExistsError):
        run.freeze_safety(path, witness_commit="abc", tau=.6, false_injections=0, total=446)
    curve = tmp_path / "posthoc-safety-curve.json"
    run.write_posthoc_curve(curve, [{"tau": .6}])
    assert json.loads(curve.read_text())["authoritative"] is False


def test_safety_filter_refuses_subset():
    with pytest.raises(ValueError, match="assertion"):
        run.filter_locomo_category5([{"qa": [{"category": 5}]}])
