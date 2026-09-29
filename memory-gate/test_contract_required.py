"""Required memory-gate contract tests; keep these tests close to the seams they lock."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("required_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(run)
import pipeline


def _case(case_id="case", text="memory"):
    return {"case_id": case_id, "query": "where?", "scope": {"project": "test"},
            "retrieved": [{"excerpt": text, "path": "m.md", "heading": []}]}


def _response(request, *, configured="requested-model", served="served-model"):
    return {"answers": {"memory_relevance_0": {"noul": 0.9}},
            "configured_model": configured, "served_model": served}


def test_t1_response_missing_served_identity_is_rejected(monkeypatch):
    monkeypatch.setattr(run, "_client_response", lambda *a, **k: {"answers": {"memory_relevance_0": {"noul": .9}}, "configured_model": "requested-model"})
    rows = run.score_cases([_case()], object(), model="requested-model")
    assert rows and rows[0]["request_error"]
    assert rows[0]["coverage"] is False


def test_t1_configured_model_must_equal_requested_model(monkeypatch):
    monkeypatch.setattr(run, "_client_response", lambda *a, **k: {"answers": {"memory_relevance_0": {"noul": .9}}, "configured_model": "other", "served_model": "other"})
    rows = run.score_cases([_case()], object(), model="requested-model")
    assert "configured" in rows[0]["request_error"] or "requested" in rows[0]["request_error"]


def test_t1_score_rows_carry_verified_identities(monkeypatch):
    monkeypatch.setattr(run, "_client_response", lambda client, request, model="", **kw: _response(request, configured=model, served="served-by-gateway"))
    rows = run.score_cases([_case()], object(), model="requested-model")
    assert rows[0]["configured_model_id"] == "requested-model"
    assert rows[0]["served_model_id"] == "served-by-gateway"


def test_t2_eval_cache_key_matches_jm_projected_wire_request_and_protocol_miss(monkeypatch, tmp_path):
    """Obtain the expected key from jm's OWN runner/CacheStore (the authority),
    not via shared helpers.  A real judge() call with a fake transport and a
    tmpdir CacheStore writes one entry; the key that store recorded is the
    ground truth.  The miss test bumps jm's real CACHE_SCHEMA constant."""
    import jm.cache as jm_cache
    from jm.answers import JudgeResponse, NoulAnswer
    from jm.cache import CacheStore

    request = pipeline.build_request("where?", pipeline.prepare_candidates({}, [{"excerpt": "memory", "path": "m.md", "heading": []}]))

    # Run the real runner/CacheStore path with a fake transport
    store = CacheStore(tmp_path / "cache")

    def fake_transport(state, questions, model):
        return JudgeResponse({"memory_relevance_0": NoulAnswer(0.9)}, served_model="test-served")

    pipeline.evaluate_production(request, fake_transport, model="m", cache_store=store)
    entries = list(store.entries())
    assert len(entries) == 1, f"expected exactly one cached entry, got {len(entries)}"
    expected = entries[0].cache_key

    # _cache_key must agree with the runner's stored key
    assert run._cache_key(request, "m") == expected

    # The miss test bumps the REAL jm cache schema constant, proving the key
    # is sensitive to jm's protocol version, not a synthetic question mutation.
    monkeypatch.setattr(jm_cache, "CACHE_SCHEMA", "jm-answer/v999-bumped")
    bumped_key = run._cache_key(request, "m")
    assert bumped_key != expected, "bumping CACHE_SCHEMA must invalidate the cache key"


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
    # (a) coverage=None -> exact "Jev judgment did not produce terminal coverage"
    with pytest.raises(pipeline.AdapterError, match="^Jev judgment did not produce terminal coverage$"):
        pipeline._decode_records([])
    # (b) partial coverage, no result, no error -> result-is-None branch raises "Jev judgment failed"
    with pytest.raises(pipeline.AdapterError, match="Jev judgment failed"):
        pipeline._decode_records([Record({"record_type": "coverage", "coverage": "partial"})])
    # (c) "request failed" error preserved verbatim (no coverage suffix)
    with pytest.raises(pipeline.AdapterError, match="^request failed$") as exc_info:
        pipeline._decode_records(
            [Record({"record_type": "coverage", "coverage": "partial"}),
             Record({"record_type": "error", "error": {"message": "request failed", "http_status": 502}})],
        )
    assert exc_info.value.http_status == 502


def test_t4_partial_and_absent_coverage_are_distinguishable(monkeypatch):
    class Record:
        def __init__(self, payload): self.payload = payload
        def to_dict(self): return self.payload
    with pytest.raises(pipeline.AdapterError, match="terminal coverage"):
        pipeline._decode_records([])
    # partial coverage WITH a result -> partial-coverage message
    with pytest.raises(pipeline.AdapterError, match="partial coverage"):
        pipeline._decode_records([Record({"record_type": "coverage", "coverage": "partial"}),
                                  Record({"record_type": "result", "answers": {}})])


def test_t4_partial_coverage_with_error_surfaces_error_with_http_status():
    """Fix 2(c): error records in the no-result branch surface verbatim error."""
    class Record:
        def __init__(self, payload): self.payload = payload
        def to_dict(self): return self.payload
    with pytest.raises(pipeline.AdapterError) as exc:
        pipeline._decode_records([
            Record({"record_type": "coverage", "coverage": "partial"}),
            Record({"record_type": "error", "error": {"message": "gateway timeout", "http_status": 504}}),
        ])
    assert str(exc.value) == "gateway timeout"
    assert exc.value.http_status == 504


def test_t4_result_with_partial_coverage_raises_partial():
    """Fix 2: result+partial -> partial-coverage message (coverage checked LAST)."""
    class Record:
        def __init__(self, payload): self.payload = payload
        def to_dict(self): return self.payload
    with pytest.raises(pipeline.AdapterError, match="^Jev judgment returned partial coverage$"):
        pipeline._decode_records([
            Record({"record_type": "coverage", "coverage": "partial"}),
            Record({"record_type": "result", "answers": {}}),
        ])


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
    with pytest.raises(pipeline.AdapterError, match="request failed"):
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
    expected_wilson = run.wilson(0, 446)
    (tmp_path / "LOCK.json").write_text(json.dumps({"tau": .731}))
    (tmp_path / "safety.json").write_text(json.dumps({
        "authoritative": True, "witness_commit": "abc1234", "tau": .731,
        "false_injections": 0, "total": 446, "point_estimate": 0.0,
        "wilson_95": expected_wilson, "accepts": True, "repeatability": None,
    }))
    (tmp_path / "safety-scores.jsonl").write_text("")
    run.posthoc_curve(tmp_path, [{"tau": .6}])
    assert json.loads((tmp_path / "posthoc-safety-curve.json").read_text())["authoritative"] is False


def test_t6_posthoc_rejects_malformed_point_estimate(tmp_path):
    """Fix 5: point_estimate must be numeric and consistent with counts."""
    expected_wilson = run.wilson(0, 446)
    (tmp_path / "LOCK.json").write_text(json.dumps({"tau": .6}))
    (tmp_path / "safety.json").write_text(json.dumps({
        "authoritative": True, "witness_commit": "abc1234", "tau": .6,
        "false_injections": 0, "total": 446, "point_estimate": "zero",
        "wilson_95": expected_wilson, "accepts": True, "repeatability": None,
    }))
    (tmp_path / "safety-scores.jsonl").write_text("")
    with pytest.raises(ValueError, match="point_estimate"):
        run.posthoc_curve(tmp_path, [{"tau": .6}])


def test_t6_posthoc_rejects_inconsistent_wilson_vs_counts(tmp_path):
    """Fix 5: Wilson bounds must match recomputed values from counts."""
    (tmp_path / "LOCK.json").write_text(json.dumps({"tau": .6}))
    (tmp_path / "safety.json").write_text(json.dumps({
        "authoritative": True, "witness_commit": "abc1234", "tau": .6,
        "false_injections": 0, "total": 446, "point_estimate": 0.0,
        "wilson_95": [0.0, 0.5],  # deliberately wrong
        "accepts": True, "repeatability": None,
    }))
    (tmp_path / "safety-scores.jsonl").write_text("")
    with pytest.raises(ValueError, match="Wilson.*inconsistent"):
        run.posthoc_curve(tmp_path, [{"tau": .6}])


def test_t6_posthoc_safety_run_with_retrieval_miss_markers(tmp_path):
    """Fix 5: safety run containing zero-candidate marker rows must not crash
    in the posthoc-safety-curve CLI path."""
    # 1 scored case + 445 marker rows (zero candidates) = 446 total
    total = 446
    fi = 1
    expected_wilson = run.wilson(fi, total)
    pe = fi / total
    accepts = (fi / total <= .02 and expected_wilson[1] <= .04)
    (tmp_path / "LOCK.json").write_text(json.dumps({"tau": .6}))
    (tmp_path / "safety.json").write_text(json.dumps({
        "authoritative": True, "witness_commit": "abc1234", "tau": .6,
        "false_injections": fi, "total": total, "point_estimate": pe,
        "wilson_95": expected_wilson, "accepts": accepts, "repeatability": None,
    }))
    # One scored case, 445 marker rows (zero candidates)
    scored = {"case_id": "case-0", "candidate_id": "case-0:candidate-0",
              "path": "m.md", "heading": [], "presented_excerpt": "memory",
              "score": 0.9, "coverage": True}
    lines = [json.dumps(scored, sort_keys=True)]
    for i in range(1, total):
        lines.append(json.dumps({"case_id": f"case-{i}", "candidates": []}, sort_keys=True))
    (tmp_path / "safety-scores.jsonl").write_text("\n".join(lines) + "\n")
    # This should succeed — the CLI iterates and must skip marker rows
    assert run.main(["posthoc-safety-curve", "--run", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "posthoc-safety-curve.json").read_text())
    assert result["authoritative"] is False
    # The total in the curve should count all cases (scored + markers)
    assert result["curve"][0]["total"] == total


def test_t7_failed_safety_validation_publishes_no_outputs(tmp_path):
    with pytest.raises(ValueError):
        run.safety_result([_score_row("x", "x:candidate-0", coverage=False)], .6, witness_commit="w", output=tmp_path / "safety.json", scores_output=tmp_path / "safety-scores.jsonl")
    assert not (tmp_path / "safety.json").exists()
    assert not (tmp_path / "safety-scores.jsonl").exists()


def test_t7_second_rename_failure_leaves_no_partial_outputs(tmp_path, monkeypatch):
    """Fix 4: if the second artifact's publication fails, the first must be
    cleaned up — no partial safety outputs may remain."""
    # Build enough valid rows for freeze_safety (>=446 cases)
    rows = [_score_row(f"case-{i}", f"case-{i}:candidate-0") for i in range(446)]
    expected_ids = {f"case-{i}" for i in range(446)}
    output = tmp_path / "safety.json"
    scores_output = tmp_path / "safety-scores.jsonl"

    original_replace = os.replace
    call_count = [0]

    def failing_replace(src, dst):
        call_count[0] += 1
        # Let the first replace (scores) succeed, fail the second (safety.json)
        if call_count[0] >= 2:
            raise OSError("simulated disk failure on second rename")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated disk failure"):
        run.safety_result(rows, .6, witness_commit="w", output=output,
                          scores_output=scores_output,
                          expected_case_ids=expected_ids)
    # The invariant: NO partial outputs remain
    assert not output.exists(), "safety.json should not exist after failed second rename"
    assert not scores_output.exists(), "safety-scores.jsonl should be cleaned up after failed second rename"


def test_t8_candidate_subprocess_fixture_and_locomo_boundaries(tmp_path):
    executable = tmp_path / "fake-pausanias"
    executable.write_text(
        "#!/usr/bin/env python3\nimport json, sys\n"
        "args = sys.argv[1:]\n"
        "assert '--retrieval-mode' in args, f'missing --retrieval-mode in {args}'\n"
        "assert '--all-projects' in args or '--project' in args or '--root' in args, f'missing scope flag in {args}'\n"
        "print(json.dumps([{'excerpt':'x','path':'x.md','heading':[]}]))\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    cases = tmp_path / "cases.json"; cases.write_text(json.dumps([_case()]))
    output = tmp_path / "candidates.jsonl"
    assert run.generate_candidates(cases, output, pausanias_executable=str(executable))
    conversation = {"qa": [{"category": "5", "question": "q", "retrieved": []}]}
    conversation["qa"] = [{"category": "5", "question": "q", "retrieved": []}] * 447
    with pytest.raises(ValueError): run.filter_locomo_category5([conversation] * 10)
    conversation["qa"] = [{"category": "5", "question": "q", "retrieved": []}]
    with pytest.raises(ValueError): run.filter_locomo_category5([conversation] * 9)


def _make_locomo_dataset(questions_per_conv=None):
    """Build a synthetic-but-locomo-SHAPED dataset with 10 conversations and 446 category-5 questions."""
    if questions_per_conv is None:
        questions_per_conv = [45] * 9 + [41]
    assert len(questions_per_conv) == 10 and sum(questions_per_conv) == 446
    dataset = []
    for conv_idx, n_questions in enumerate(questions_per_conv):
        questions = []
        for q_idx in range(n_questions):
            questions.append({
                "category": "5",
                "question": f"conversation-{conv_idx}-question-{q_idx}",
                "retrieved": [{"excerpt": f"excerpt-{conv_idx}-{q_idx}", "path": f"memory-{conv_idx}.md", "heading": []}],
            })
        dataset.append({"id": f"conversation-{conv_idx}", "qa": questions})
    return dataset


def test_t9_flat_case_file_is_refused_by_safety_pipeline(monkeypatch, tmp_path):
    """Fix 1: an arbitrary flat 446-case file must be refused by the safety CLI."""
    repo, bare = tmp_path / "work", tmp_path / "remote.git"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=repo, check=True)
    run_dir = repo / "run"
    shutil.copytree(HERE / "runs" / "fixture-dev", run_dir)
    tau = 0.731
    run.lock_witness(run_dir / "LOCK.json",
                     [run_dir / name for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")], tau)
    # Write a flat (non-locomo-shaped) case file with 446 entries
    cases_path = repo / "cases.json"
    cases_path.write_text(json.dumps([_case(f"case-{i}") for i in range(446)]))
    responses_path = repo / "responses.json"
    responses_path.write_text(json.dumps([_response({}, configured="requested-model", served="gateway-model") for _ in range(446)]))
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=test", "commit", "-m", "lock"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=repo, check=True, capture_output=True)
    witness = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    monkeypatch.chdir(repo)
    # The flat file must be refused — it's not locomo-shaped
    with pytest.raises((ValueError, TypeError, SystemExit)):
        run.main(["score", "--lane", "safety", "--run", str(run_dir), "--cases", str(cases_path),
                  "--responses", str(responses_path), "--model", "requested-model", "--witness", witness])
    # No safety artifacts should have been created
    assert not (run_dir / "safety.json").exists()


def test_t9_locomo_shaped_dataset_through_real_filter_succeeds(monkeypatch, tmp_path):
    """Fix 1: a locomo-shaped synthetic dataset through the real filter succeeds."""
    repo, bare = tmp_path / "work", tmp_path / "remote.git"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=repo, check=True)
    run_dir = repo / "run"
    shutil.copytree(HERE / "runs" / "fixture-dev", run_dir)
    tau = 0.731
    run.lock_witness(run_dir / "LOCK.json",
                     [run_dir / name for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")], tau)
    # Build a locomo-shaped dataset (10 conversations, 446 category-5 questions)
    dataset = _make_locomo_dataset()
    cases_path = repo / "cases.json"
    cases_path.write_text(json.dumps(dataset))
    # Each case produces 1 candidate, so we need 446 responses
    responses_path = repo / "responses.json"
    responses_path.write_text(json.dumps([_response({}, configured="requested-model", served="gateway-model") for _ in range(446)]))
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=test", "commit", "-m", "lock"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=repo, check=True, capture_output=True)
    witness = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    monkeypatch.chdir(repo)
    assert run.main(["score", "--lane", "safety", "--run", str(run_dir), "--cases", str(cases_path),
                     "--responses", str(responses_path), "--model", "requested-model", "--witness", witness]) == 0
    artifact = json.loads((run_dir / "safety.json").read_text())
    assert artifact["authoritative"] and artifact["witness_commit"] == witness
    assert artifact["tau"] == json.loads((run_dir / "LOCK.json").read_text())["tau"] == tau


# ---------------------------------------------------------------------------
# T10: Scope flags + retrieval-mode fidelity (Fix 1, Fix 2, Fix 5)
# ---------------------------------------------------------------------------

def test_t10_resolve_scope_flags():
    """_resolve_scope_flags must produce --root <id>, --project <name>, or
    --all-projects; missing/empty/malformed scope raises ScopeResolutionError."""
    assert run._resolve_scope_flags({"scope": {"project": "phoebe"}}) == ["--project", "phoebe"]
    assert run._resolve_scope_flags({"scope": {"all_projects": True}}) == ["--all-projects"]
    assert run._resolve_scope_flags({"scope": {"root": "pausanias"}}) == ["--root", "pausanias"]
    assert run._resolve_scope_flags({"scope": {"root": "atlas"}}) == ["--root", "atlas"]
    # Missing, empty, or malformed scope must raise (not silently broaden)
    with pytest.raises(run.ScopeResolutionError):
        run._resolve_scope_flags({"scope": {}})
    with pytest.raises(run.ScopeResolutionError):
        run._resolve_scope_flags({})
    with pytest.raises(run.ScopeResolutionError):
        run._resolve_scope_flags({"scope": {"project": ""}})
    with pytest.raises(run.ScopeResolutionError):
        run._resolve_scope_flags({"scope": {"all_projects": False}})
    with pytest.raises(run.ScopeResolutionError):
        run._resolve_scope_flags({"scope": {"root": ""}})


def test_t10_resolve_scope_real_root_shapes():
    """The two real root-scoped cases from cases.json parse correctly."""
    # cases.json:156 — {"root": "pausanias"}
    assert run._resolve_scope_flags({"id": "packet-budget-root", "scope": {"root": "pausanias"}}) == ["--root", "pausanias"]
    assert run._resolve_scope({"scope": {"root": "pausanias"}}) == {"root": "pausanias"}
    # cases.json:493 — {"root": "atlas"}
    assert run._resolve_scope_flags({"id": "global-security-root", "scope": {"root": "atlas"}}) == ["--root", "atlas"]
    assert run._resolve_scope({"scope": {"root": "atlas"}}) == {"root": "atlas"}


def _fake_pausanias_asserting_script() -> str:
    """Fake pausanias that asserts scope and retrieval-mode flags, then
    echoes the received args as JSON metadata alongside a dummy result."""
    return (
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "# Strict seam: must receive --retrieval-mode fused\n"
        "assert '--retrieval-mode' in args, f'missing --retrieval-mode in {args}'\n"
        "rm_idx = args.index('--retrieval-mode')\n"
        "assert args[rm_idx + 1] == 'fused', f'expected fused, got {args[rm_idx + 1]}'\n"
        "# Strict seam: must receive a scope flag\n"
        "has_project = '--project' in args\n"
        "has_all = '--all-projects' in args\n"
        "has_root = '--root' in args\n"
        "assert has_project or has_all or has_root, f'missing scope flag in {args}'\n"
        "assert sum([has_project, has_all, has_root]) == 1, f'multiple scope flags in {args}'\n"
        "# Encode which scope we received for test assertion\n"
        "if has_root:\n"
        "    scope_kind = 'root'\n"
        "    scope_value = args[args.index('--root') + 1]\n"
        "elif has_project:\n"
        "    scope_kind = 'project'\n"
        "    scope_value = args[args.index('--project') + 1]\n"
        "else:\n"
        "    scope_kind = 'all_projects'\n"
        "    scope_value = None\n"
        "print(json.dumps([{"
        "'excerpt': f'scope={scope_kind}:{scope_value}', "
        "'path': 'test.md', 'heading': []}]))\n"
    )


def test_t10_project_scoped_case_passes_project_flag(tmp_path):
    """A case with scope.project='phoebe' must produce --project phoebe in
    the pausanias subprocess; the fake asserts and echoes."""
    executable = tmp_path / "fake-pausanias"
    executable.write_text(_fake_pausanias_asserting_script())
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"case_id": "scoped-1", "query": "where?",
         "scope": {"project": "phoebe"}},
    ]))
    output = tmp_path / "candidates.jsonl"
    rows = run.generate_candidates(cases, output, pausanias_executable=str(executable))
    assert len(rows) == 1
    assert "scope=project:phoebe" in rows[0]["presented_excerpt"]


def test_t10_all_projects_case_passes_all_projects_flag(tmp_path):
    """A case with scope.all_projects=true must produce --all-projects."""
    executable = tmp_path / "fake-pausanias"
    executable.write_text(_fake_pausanias_asserting_script())
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"case_id": "global-1", "query": "anything",
         "scope": {"all_projects": True}},
    ]))
    output = tmp_path / "candidates.jsonl"
    rows = run.generate_candidates(cases, output, pausanias_executable=str(executable))
    assert len(rows) == 1
    assert "scope=all_projects:None" in rows[0]["presented_excerpt"]


def test_t10_mixed_scope_cases(tmp_path):
    """Both a project-scoped and an all-projects case in one generation."""
    executable = tmp_path / "fake-pausanias"
    executable.write_text(_fake_pausanias_asserting_script())
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"case_id": "proj-1", "query": "q1", "scope": {"project": "phoebe"}},
        {"case_id": "glob-1", "query": "q2", "scope": {"all_projects": True}},
    ]))
    output = tmp_path / "candidates.jsonl"
    rows = run.generate_candidates(cases, output, pausanias_executable=str(executable))
    proj_rows = [r for r in rows if r["case_id"] == "proj-1"]
    glob_rows = [r for r in rows if r["case_id"] == "glob-1"]
    assert proj_rows and "scope=project:phoebe" in proj_rows[0]["presented_excerpt"]
    assert glob_rows and "scope=all_projects:None" in glob_rows[0]["presented_excerpt"]


def test_t10_root_scoped_case_passes_root_flag(tmp_path):
    """A case with scope.root='pausanias' must produce --root pausanias."""
    executable = tmp_path / "fake-pausanias"
    executable.write_text(_fake_pausanias_asserting_script())
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"case_id": "root-1", "query": "packet budget?",
         "scope": {"root": "pausanias"}},
    ]))
    output = tmp_path / "candidates.jsonl"
    rows = run.generate_candidates(cases, output, pausanias_executable=str(executable))
    assert len(rows) == 1
    assert "scope=root:pausanias" in rows[0]["presented_excerpt"]
    assert rows[0]["retrieval_scope"] == {"root": "pausanias"}


def test_t10_mixed_scope_with_root(tmp_path):
    """Root, project, and all-projects cases in one generation."""
    executable = tmp_path / "fake-pausanias"
    executable.write_text(_fake_pausanias_asserting_script())
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"case_id": "root-1", "query": "q1", "scope": {"root": "atlas"}},
        {"case_id": "proj-1", "query": "q2", "scope": {"project": "phoebe"}},
        {"case_id": "glob-1", "query": "q3", "scope": {"all_projects": True}},
    ]))
    output = tmp_path / "candidates.jsonl"
    rows = run.generate_candidates(cases, output, pausanias_executable=str(executable))
    root_rows = [r for r in rows if r["case_id"] == "root-1"]
    proj_rows = [r for r in rows if r["case_id"] == "proj-1"]
    glob_rows = [r for r in rows if r["case_id"] == "glob-1"]
    assert root_rows and "scope=root:atlas" in root_rows[0]["presented_excerpt"]
    assert root_rows[0]["retrieval_scope"] == {"root": "atlas"}
    assert proj_rows and "scope=project:phoebe" in proj_rows[0]["presented_excerpt"]
    assert proj_rows[0]["retrieval_scope"] == {"project": "phoebe"}
    assert glob_rows and "scope=all_projects:None" in glob_rows[0]["presented_excerpt"]
    assert glob_rows[0]["retrieval_scope"] == {"all_projects": True}


def test_t10_missing_scope_raises_in_generate_candidates(tmp_path):
    """Cases without scope must raise ScopeResolutionError, not silently broaden."""
    executable = tmp_path / "fake-pausanias"
    executable.write_text(_fake_pausanias_asserting_script())
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"case_id": "no-scope", "query": "q1"},
    ]))
    output = tmp_path / "candidates.jsonl"
    with pytest.raises(run.ScopeResolutionError, match="missing or empty"):
        run.generate_candidates(cases, output, pausanias_executable=str(executable))


# ---------------------------------------------------------------------------
# T10b: Per-case retrieval_scope provenance
# ---------------------------------------------------------------------------

def test_t10b_candidate_rows_carry_per_case_scope():
    """candidate_rows must attach retrieval_scope from each case's scope."""
    provenance = {
        "case_set_fingerprint": "x", "corpus_fingerprint": "x",
        "pausanias_revision": "x", "retrieval_config": {"mode": "test"},
        "production_builder_hash": "x",
        "configured_model_id": "x", "served_model_id": "x",
        "harness_revision": "x", "jm_revision": "x",
    }
    cases = [
        {"case_id": "proj-a", "query": "q1", "scope": {"project": "alpha"},
         "retrieved": [{"excerpt": "data", "path": "a.md", "heading": []}]},
        {"case_id": "root-b", "query": "q2", "scope": {"root": "beta"},
         "retrieved": [{"excerpt": "data2", "path": "b.md", "heading": []}]},
        {"case_id": "all-c", "query": "q3", "scope": {"all_projects": True},
         "retrieved": [{"excerpt": "data3", "path": "c.md", "heading": []}]},
    ]
    rows = run.candidate_rows(cases, provenance)
    assert rows[0]["retrieval_scope"] == {"project": "alpha"}
    assert rows[1]["retrieval_scope"] == {"root": "beta"}
    assert rows[2]["retrieval_scope"] == {"all_projects": True}


def test_t10b_validator_rejects_mismatched_intra_case_scope(tmp_path):
    """Rows from the same case_id must agree on retrieval_scope."""
    import artifacts
    provenance = {
        "case_set_fingerprint": "x", "corpus_fingerprint": "x",
        "pausanias_revision": "x", "retrieval_config": {"mode": "test"},
        "production_builder_hash": artifacts.authoritative_production_builder_hash(),
        "configured_model_id": "model", "served_model_id": "model",
        "harness_revision": "x", "jm_revision": "x",
    }
    cases = [
        {"case_id": "c1", "query": "q", "scope": {"project": "alpha"},
         "retrieved": [
             {"excerpt": "one", "path": "a.md", "heading": []},
             {"excerpt": "two", "path": "b.md", "heading": []},
         ]},
    ]
    rows = run.candidate_rows(cases, provenance)
    assert len(rows) == 2
    # Tamper: give the two rows different scopes
    rows[0]["retrieval_scope"] = {"project": "alpha"}
    rows[1]["retrieval_scope"] = {"project": "beta"}
    for row in rows:
        row["canonical_request_hash"] = artifacts._canonical_hash(rows)
    scores = [{**row, "score": 0.9, "request_error": None, "coverage": True} for row in rows]
    labels = [{**row, "label": "positive"} for row in rows]
    artifacts.write_artifact(tmp_path, rows, scores, labels, "report")
    with pytest.raises(artifacts.ArtifactValidationError, match="mixed retrieval_scope"):
        artifacts.validate_run(tmp_path)


def test_t10b_validator_allows_different_scope_across_cases(tmp_path):
    """Different cases may (and should) have different retrieval_scope values."""
    import artifacts
    provenance = {
        "case_set_fingerprint": "x", "corpus_fingerprint": "x",
        "pausanias_revision": "x", "retrieval_config": {"mode": "test"},
        "production_builder_hash": artifacts.authoritative_production_builder_hash(),
        "configured_model_id": "model", "served_model_id": "model",
        "harness_revision": "x", "jm_revision": "x",
    }
    cases = [
        {"case_id": "proj", "query": "q1", "scope": {"project": "alpha"},
         "retrieved": [{"excerpt": "data", "path": "a.md", "heading": []}]},
        {"case_id": "root", "query": "q2", "scope": {"root": "beta"},
         "retrieved": [{"excerpt": "data2", "path": "b.md", "heading": []}]},
    ]
    rows = run.candidate_rows(cases, provenance)
    for row in rows:
        row["canonical_request_hash"] = artifacts._canonical_hash(
            [r for r in rows if r["case_id"] == row["case_id"]]
        )
    scores = [{**row, "score": 0.9, "request_error": None, "coverage": True} for row in rows]
    labels = [{**row, "label": "positive"} for row in rows]
    artifacts.write_artifact(tmp_path, rows, scores, labels, "report")
    artifacts.validate_run(tmp_path)


# ---------------------------------------------------------------------------
# T11: Zero-candidate guard (Fix 4)
# ---------------------------------------------------------------------------

def test_t11_mass_empty_retrieval_raises(tmp_path):
    """If >20% of cases return zero candidates, generation must fail loudly."""
    executable = tmp_path / "fake-pausanias"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "# Accept any args but return empty\n"
        "print(json.dumps([]))\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"case_id": f"case-{i}", "query": f"query-{i}", "scope": {"all_projects": True}} for i in range(10)
    ]))
    output = tmp_path / "candidates.jsonl"
    with pytest.raises(run.EmptyRetrievalError, match="10/10.*100%.*threshold"):
        run.generate_candidates(cases, output, pausanias_executable=str(executable))


def test_t11_below_threshold_succeeds(tmp_path):
    """If <=20% of cases return zero candidates, generation proceeds."""
    # 1/5 = 20% -> exactly at threshold, should pass (> not >=)
    executable = tmp_path / "fake-pausanias"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "query = sys.argv[-1]\n"
        "if 'empty' in query:\n"
        "    print(json.dumps([]))\n"
        "else:\n"
        "    print(json.dumps([{'excerpt':'data','path':'f.md','heading':[]}]))\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    cases = tmp_path / "cases.json"
    case_list = [{"case_id": f"case-{i}", "query": f"query-{i}", "scope": {"project": "test"}} for i in range(4)]
    case_list.append({"case_id": "case-empty", "query": "empty", "scope": {"project": "test"}})
    cases.write_text(json.dumps(case_list))
    output = tmp_path / "candidates.jsonl"
    rows = run.generate_candidates(cases, output, pausanias_executable=str(executable))
    # 4 cases have 1 candidate each, 1 case has 0 -> 4 rows total
    assert len(rows) == 4


# ---------------------------------------------------------------------------
# T12: Provenance placeholders are real values (Fix 3)
# ---------------------------------------------------------------------------

def test_t12_corpus_fingerprint_is_not_unknown(tmp_path):
    """corpus_fingerprint must be computed from config file, not 'unknown'."""
    config = tmp_path / "corpus.toml"
    config.write_text("[corpus]\npath = '/tmp'\n")
    fp = run._corpus_fingerprint(config)
    assert fp != "unknown"
    assert fp != "unavailable"
    assert len(fp) == 64  # sha256 hex
    # Deterministic
    assert fp == run._corpus_fingerprint(config)


def test_t12_corpus_fingerprint_unavailable_without_config():
    """No config -> 'unavailable', not 'unknown'."""
    assert run._corpus_fingerprint(None) == "unavailable"
    assert run._corpus_fingerprint(Path("/nonexistent")) == "unavailable"


def test_t12_git_revision_returns_hash_in_real_repo():
    """In this checkout (wherever it lives), _git_revision returns a commit hash."""
    rev = run._git_revision(Path(run.__file__).resolve().parents[1])
    assert rev != "unavailable"
    assert len(rev) == 40  # full SHA-1


def test_t12_git_revision_unavailable_for_nonexistent():
    assert run._git_revision(None) == "unavailable"
    assert run._git_revision(Path("/nonexistent")) == "unavailable"


def test_t12_retrieval_config_records_mode_in_candidates(tmp_path):
    """retrieval_config in provenance must include retrieval_mode: fused."""
    executable = tmp_path / "fake-pausanias"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "assert '--retrieval-mode' in sys.argv, f'missing --retrieval-mode in {sys.argv}'\n"
        "assert '--all-projects' in sys.argv or '--project' in sys.argv or '--root' in sys.argv\n"
        "print(json.dumps([{'excerpt':'x','path':'x.md','heading':[]}]))\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"case_id": "c1", "query": "q", "scope": {"project": "test"}}]))
    output = tmp_path / "candidates.jsonl"
    rows = run.generate_candidates(cases, output, pausanias_executable=str(executable))
    assert rows[0]["retrieval_config"]["retrieval_mode"] == "fused"


# ---------------------------------------------------------------------------
# T13: Repeatability CLI — subset construction, overwrite refusal, output schema
# ---------------------------------------------------------------------------

def _make_candidate_row(case_id, rank=0, excerpt="memory"):
    """Build a minimal candidates.jsonl row."""
    return {
        "case_id": case_id,
        "candidate_id": f"{case_id}:candidate-{rank}",
        "query": f"query for {case_id}",
        "path": "m.md",
        "heading": [],
        "rank": rank,
        "presented_excerpt": excerpt,
        "untruncated_excerpt_hash": "abc123",
        "retrieval_scope": {"project": "test"},
        "case_set_fingerprint": "x",
        "corpus_fingerprint": "x",
        "pausanias_revision": "x",
        "retrieval_config": {"mode": "test"},
        "production_builder_hash": "x",
        "configured_model_id": "x",
        "served_model_id": "x",
        "harness_revision": "x",
        "jm_revision": "x",
        "canonical_request_hash": "x",
    }


def _make_safety_cases_json():
    """Build a minimal locomo-shaped safety-cases.json with 10 convs / 446 cat-5 questions."""
    questions_per_conv = [45] * 9 + [41]
    dataset = []
    for conv_idx, n in enumerate(questions_per_conv):
        questions = []
        for q_idx in range(n):
            questions.append({
                "category": "5",
                "question": f"conv-{conv_idx}-q-{q_idx}",
                "retrieved": [{"excerpt": f"excerpt-{conv_idx}-{q_idx}", "path": f"m-{conv_idx}.md", "heading": []}],
                "retrieval_provenance": {"pipeline": "locomo-pinned"},
            })
        dataset.append({"qa": questions, "sample_id": f"conv-{conv_idx}"})
    return dataset


def _write_repeatability_run_dir(run_dir):
    """Set up a run dir with enough verbatim/paraphrase candidates + safety-cases.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    verbatim_ids = [f"prior-case-{i}" for i in range(4)] + [f"exact-case-{i}" for i in range(2)]
    paraphrase_ids = [f"paraphrase-case-{i}" for i in range(5)]
    other_ids = ["other-case-0"]
    rows = []
    for cid in verbatim_ids + paraphrase_ids + other_ids:
        rows.append(_make_candidate_row(cid))
    (run_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    (run_dir / "safety-cases.json").write_text(json.dumps(_make_safety_cases_json()))


def test_t13_subset_construction_stratification(tmp_path):
    """build_repeatability_subset produces 4 verbatim + 4 paraphrase + 4 abstain."""
    run_dir = tmp_path / "run"
    _write_repeatability_run_dir(run_dir)
    subset = run.build_repeatability_subset(run_dir)
    assert len(subset) == 12
    cats = [c["category"] for c in subset]
    assert cats.count("verbatim") == 4
    assert cats.count("paraphrase") == 4
    assert cats.count("abstain") == 4
    # Verbatim cases come from prior-/exact- prefix
    verbatim_ids = [c["case_id"] for c in subset if c["category"] == "verbatim"]
    assert all(cid.startswith(("prior-", "exact-")) for cid in verbatim_ids)
    # Paraphrase from paraphrase- prefix
    para_ids = [c["case_id"] for c in subset if c["category"] == "paraphrase"]
    assert all(cid.startswith("paraphrase-") for cid in para_ids)
    # Abstain from locomo
    abstain_ids = [c["case_id"] for c in subset if c["category"] == "abstain"]
    assert all(cid.startswith("locomo-") for cid in abstain_ids)


def test_t13_subset_abstain_sourced_from_safety_cases(tmp_path):
    """Abstain stratum is sourced from safety-cases.json, not calibration candidates."""
    run_dir = tmp_path / "run"
    _write_repeatability_run_dir(run_dir)
    subset = run.build_repeatability_subset(run_dir)
    abstain = [c for c in subset if c["category"] == "abstain"]
    assert len(abstain) == 4
    # Each abstain case must have the answerable=False marker from safety_cases()
    for c in abstain:
        assert c["answerable"] is False
    # Without safety-cases.json, must fail
    run_dir2 = tmp_path / "run2"
    run_dir2.mkdir(parents=True)
    # Copy candidates but not safety-cases.json
    import shutil
    shutil.copy(run_dir / "candidates.jsonl", run_dir2 / "candidates.jsonl")
    with pytest.raises(ValueError, match="abstain"):
        run.build_repeatability_subset(run_dir2)


def test_t13_subset_insufficient_verbatim_raises(tmp_path):
    """Must raise if fewer than 4 verbatim cases available."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    # Only 2 verbatim, enough paraphrase
    rows = []
    for i in range(2):
        rows.append(_make_candidate_row(f"exact-case-{i}"))
    for i in range(4):
        rows.append(_make_candidate_row(f"paraphrase-case-{i}"))
    (run_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    (run_dir / "safety-cases.json").write_text(json.dumps(_make_safety_cases_json()))
    with pytest.raises(ValueError, match="verbatim"):
        run.build_repeatability_subset(run_dir)


def test_t13_overwrite_refusal(tmp_path):
    """repeatability CLI must refuse to overwrite existing repeatability.json."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "repeatability.json").write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run.main(["repeatability", "--run", str(run_dir), "--tau", "0.6"])


def test_t13_output_schema(tmp_path, monkeypatch):
    """repeatability CLI writes a well-formed repeatability.json."""
    run_dir = tmp_path / "run"
    _write_repeatability_run_dir(run_dir)

    call_count = [0]
    def fake_client(request):
        call_count[0] += 1
        # Return a plausible response for any number of candidates
        answers = {}
        for key in request.get("questions", {}):
            answers[key] = {"noul": 0.7}
        return {"answers": answers, "configured_model": "test-model", "served_model": "test-served"}

    monkeypatch.setattr(run, "build_live_client", lambda: (fake_client, None))
    monkeypatch.setattr(run, "_client_response",
                        lambda client, req, model="", **kw: client(req))

    exit_code = run.main(["repeatability", "--run", str(run_dir), "--tau", "0.58", "--model", "test-model"])
    assert exit_code == 0

    output = json.loads((run_dir / "repeatability.json").read_text())
    assert output["replicates"] == 6
    assert output["tau"] == 0.58
    assert "strata" in output
    assert set(output["strata"].keys()) == {"abstain", "verbatim", "paraphrase"}
    for stratum_cases in output["strata"].values():
        assert len(stratum_cases) == 4
    assert "worst_spread" in output
    assert "fraction_crossing_tau" in output
    assert "model_identity" in output
    assert "per_candidate" in output


def test_t13_categorize_case_id():
    """_categorize_case_id correctly maps prefixes to strata."""
    assert run._categorize_case_id("prior-sandbox") == "verbatim"
    assert run._categorize_case_id("exact-pho-123") == "verbatim"
    assert run._categorize_case_id("paraphrase-cache") == "paraphrase"
    assert run._categorize_case_id("locomo-0") == "abstain"
    assert run._categorize_case_id("other-case") is None
    assert run._categorize_case_id("concept-query") is None
