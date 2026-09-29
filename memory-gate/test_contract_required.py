"""Required memory-gate contract tests; keep these tests close to the seams they lock."""
from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("required_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(run)
import pipeline


def _case(case_id="case", text="memory"):
    return {"case_id": case_id, "query": "where?", "retrieved": [{"excerpt": text, "path": "m.md", "heading": []}]}


def _response(request, *, configured="requested-model", served="served-model"):
    return {"answers": {"memory_relevance_0": {"noul": 0.9}},
            "configured_model": configured, "served_model": served}


def test_t1_response_missing_served_identity_is_rejected(monkeypatch):
    monkeypatch.setattr(run, "_client_response", lambda *a, **k: {"answers": {"candidate-0": {"score": .9}}, "configured_model": "requested-model"})
    rows = run.score_cases([_case()], object(), model="requested-model")
    assert rows and rows[0]["request_error"]
    assert rows[0]["coverage"] is False


def test_t1_configured_model_must_equal_requested_model(monkeypatch):
    monkeypatch.setattr(run, "_client_response", lambda *a, **k: {"answers": {"candidate-0": {"score": .9}}, "configured_model": "other", "served_model": "other"})
    rows = run.score_cases([_case()], object(), model="requested-model")
    assert "configured" in rows[0]["request_error"] or "requested" in rows[0]["request_error"]


def test_t1_score_rows_carry_verified_identities(monkeypatch):
    monkeypatch.setattr(run, "_client_response", lambda request, client, model: _response(request, configured=model, served="served-by-gateway"))
    rows = run.score_cases([_case()], object(), model="requested-model")
    assert rows[0]["configured_model_id"] == "requested-model"
    assert rows[0]["served_model_id"] == "served-by-gateway"


def test_t2_eval_cache_key_matches_jm_projected_wire_request_and_protocol_miss():
    request = pipeline.build_request("where?", pipeline.prepare_candidates({}, [{"excerpt": "memory", "path": "m.md", "heading": []}]))
    from jm.cache import build_cache_preimage, cache_key
    expected = cache_key(build_cache_preimage(model="m", questions=request["questions"], state=run._formed_state(request)))
    assert run._cache_key(request, "m") == expected
    changed = {**request, "questions": {**request["questions"], "protocol_version": "changed"}}
    assert run._cache_key(changed, "m") != expected


def _score_row(case_id, candidate_id, score=.9, coverage=True):
    return {"case_id": case_id, "candidate_id": candidate_id, "path": "m.md", "heading": [],
            "presented_excerpt": "memory", "score": score, "coverage": coverage}


def test_t3_zero_candidate_case_remains_in_safety_denominator():
    rows = [_score_row("present", "present:candidate-0")]
    rows.extend({"case_id": f"case-{i}", "candidates": []} for i in range(1, 446))
    artifact = run.safety_result(rows, .6, witness_commit="w", output=Path("/tmp/jev-safety-test.json"), expected_case_ids={f"case-{i}" for i in range(1, 446)} | {"present"})
    assert artifact["total"] == 446
    Path("/tmp/jev-safety-test.json").unlink(missing_ok=True)


def test_t3_freeze_refuses_missing_case_ids(tmp_path):
    with pytest.raises(ValueError, match="case IDs"):
        run.safety_result([_score_row("only", "only:candidate-0")], .6, witness_commit="w", output=tmp_path / "safety.json", expected_case_ids={"only", "missing"})


def test_t4_adapter_error_classes_are_distinct_and_preserve_context(monkeypatch):
    class Record:
        def __init__(self, payload): self.payload = payload
        def to_dict(self): return self.payload
    seen = []
    for records in ([Record({"record_type": "coverage", "coverage": "complete"})],
                    [Record({"record_type": "coverage", "coverage": "complete"}), Record({"record_type": "error", "error": "bad"})],
                    [Record({"record_type": "coverage", "coverage": "partial"})]):
        with pytest.raises(pipeline.AdapterError) as exc:
            pipeline._decode_records(records)
        seen.append(str(exc.value))
    assert len(set(seen)) == 3
    assert all(text for text in seen)


def test_t4_partial_and_absent_coverage_are_distinguishable(monkeypatch):
    class Record:
        def __init__(self, payload): self.payload = payload
        def to_dict(self): return self.payload
    with pytest.raises(pipeline.AdapterError, match="missing coverage"):
        pipeline._decode_records([])
    with pytest.raises(pipeline.AdapterError, match="partial coverage"):
        pipeline._decode_records([Record({"record_type": "coverage", "coverage": "partial"})])


def test_t5_evaluate_protocol_is_called_once():
    class Client:
        def __init__(self): self.calls = 0
        def evaluate(self, *args, **kwargs): self.calls += 1; return {"answers": {}, "served_model": "m"}
    client = Client()
    pipeline.evaluate_production({"questions": {}, "state": {}}, client, model="m")
    assert client.calls == 1


def test_t5_callable_transport_typeerror_is_not_retried():
    calls = []
    def transport(*args):
        calls.append(args)
        raise TypeError("transport internals")
    with pytest.raises(pipeline.AdapterError, match="partial coverage"):
        pipeline.evaluate_production({"questions": {}, "state": {}}, transport, model="m")
    assert len(calls) == 1


def test_t5_legacy_callable_protocol_still_works():
    calls = []
    def legacy(request):
        calls.append(request)
        return {"answers": {}, "served_model": "m"}
    pipeline.evaluate_production({"questions": {}, "state": {}}, legacy, model="m")
    assert len(calls) == 1


def test_t6_posthoc_requires_validated_frozen_safety(tmp_path):
    with pytest.raises((ValueError, SystemExit), match="frozen|safety"):
        run.posthoc_curve(tmp_path, [{"tau": .6}])
    (tmp_path / "safety.json").write_text(json.dumps({"authoritative": True, "accepts": True, "witness_commit": "w"}))
    (tmp_path / "safety-scores.jsonl").write_text("")
    run.posthoc_curve(tmp_path, [{"tau": .6}])
    assert json.loads((tmp_path / "posthoc-safety-curve.json").read_text())["authoritative"] is False


def test_t7_failed_safety_validation_publishes_no_outputs(tmp_path):
    with pytest.raises(ValueError):
        run.safety_result([_score_row("x", "x:candidate-0", coverage=False)], .6, witness_commit="w", output=tmp_path / "safety.json", scores_output=tmp_path / "safety-scores.jsonl")
    assert not (tmp_path / "safety.json").exists()
    assert not (tmp_path / "safety-scores.jsonl").exists()


def test_t8_candidate_subprocess_fixture_and_locomo_boundaries(tmp_path):
    executable = tmp_path / "fake-pausanias"
    executable.write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps([{'excerpt':'x','path':'x.md','heading':[]}]))\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    cases = tmp_path / "cases.json"; cases.write_text(json.dumps([_case()]))
    output = tmp_path / "candidates.jsonl"
    assert run.generate_candidates(cases, output, pausanias_executable=str(executable))
    conversation = {"qa": [{"category": "5", "question": "q", "retrieved": []}]}
    conversation["qa"] = [{"category": "5", "question": "q", "retrieved": []}] * 447
    with pytest.raises(ValueError): run.filter_locomo_category5([conversation] * 10)
    conversation["qa"] = [{"category": "5", "question": "q", "retrieved": []}]
    with pytest.raises(ValueError): run.filter_locomo_category5([conversation] * 9)


def test_t9_calibration_and_witnessed_safety_happy_paths(monkeypatch, tmp_path):
    def fake(request, client, model): return _response(request, configured=model, served="gateway-model")
    monkeypatch.setattr(run, "_client_response", fake)
    cases = [_case(f"case-{i}") for i in range(446)]
    rows = run.score_cases(cases, object(), model="requested-model")
    assert rows[0]["coverage"]
    artifact = run.safety_result(rows, .6, witness_commit="w", output=tmp_path / "safety.json", expected_case_ids={f"case-{i}" for i in range(446)})
    assert artifact["authoritative"] and artifact["witness_commit"] == "w"
    assert artifact["tau"] == .6 and "wilson_95" in artifact and isinstance(artifact["accepts"], bool)
