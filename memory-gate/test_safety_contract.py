from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("memory_gate_contract_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(run)
import lock


def git(cwd: Path, *args: str, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def repo_with_lock(tmp_path: Path):
    repo, bare = tmp_path / "repo", tmp_path / "remote.git"
    repo.mkdir(); bare.mkdir()
    git(repo, "init", "-b", "main"); git(bare, "init", "--bare")
    git(repo, "remote", "add", "origin", str(bare))
    run_dir = repo / "run"
    shutil.copytree(HERE / "runs" / "fixture-dev", run_dir)
    lock.write_lock(run_dir / "LOCK.json", [run_dir / n for n in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")], .6)
    git(repo, "add", ".")
    git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-m", "lock")
    witness = git(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, run_dir, witness


def test_safety_cli_main_refuses_when_witness_absent():
    with pytest.raises(SystemExit, match="requires --witness"):
        run.main(["score", "--lane", "safety", "--run", "run"])


def test_unpushed_local_witness_rejected_after_prune_even_with_stale_tracking_ref(tmp_path):
    repo, run_dir, witness = repo_with_lock(tmp_path)
    git(repo, "update-ref", "refs/remotes/origin/main", witness)
    with pytest.raises(ValueError, match="reachable"):
        run.verify_witness(run_dir / "LOCK.json", witness, repo=repo)
    assert git(repo, "show-ref", "--verify", "--quiet", "refs/remotes/origin/main", check=False).returncode != 0


def test_safety_refuses_verification_before_phase_c_stub(monkeypatch, capsys):
    calls = []
    def verify(*args, **kwargs):
        calls.append("verify")
        print("verified")
        return "abc"
    monkeypatch.setattr(run, "verify_witness", verify)
    with pytest.raises(SystemExit, match="phase C not implemented"):
        run.main(["score", "--lane", "safety", "--run", "run", "--witness", "abc"])
    assert calls == ["verify"]
    assert capsys.readouterr().out == "verified\n"


def test_absent_lock_json_is_distinct_rejection(tmp_path):
    with pytest.raises(ValueError, match="LOCK.json does not exist"):
        run.verify_witness(tmp_path / "LOCK.json", "HEAD", repo=tmp_path)


def test_verify_witness_returns_resolved_witness_commit_hash(tmp_path):
    repo, run_dir, witness = repo_with_lock(tmp_path)
    git(repo, "push", "origin", "main")
    assert run.verify_witness(run_dir / "LOCK.json", witness, repo=repo) == witness


def test_non_default_remote_is_passed_to_verification(monkeypatch, tmp_path):
    seen = {}
    def verify(lock_path, witness, remote="origin", **kwargs):
        seen["remote"] = remote
        return "resolved"
    monkeypatch.setattr(run, "verify_witness", verify)
    with pytest.raises(SystemExit, match="phase C not implemented"):
        run.main(["score", "--lane", "safety", "--remote", "backup", "--run", str(tmp_path), "--witness", "abc"])
    assert seen["remote"] == "backup"


def test_lock_subcommand_refuses_invalid_calibration_run(tmp_path):
    run_dir = tmp_path / "run"; run_dir.mkdir()
    (run_dir / "candidates.jsonl").write_text("not json\n")
    for name in ("scores.jsonl", "labels.jsonl", "report.md"):
        (run_dir / name).write_text("")
    with pytest.raises(run.artifacts.ArtifactValidationError):
        run.main(["lock", "--run", str(run_dir)])


def test_report_drift_after_lock_is_rejected(tmp_path):
    repo, run_dir, witness = repo_with_lock(tmp_path)
    git(repo, "push", "origin", "main")
    (run_dir / "report.md").write_text("drift")
    with pytest.raises(ValueError, match="hash mismatch"):
        run.verify_witness(run_dir / "LOCK.json", witness, repo=repo)


@pytest.mark.parametrize("hashes", [{}, {"candidates.jsonl": "x"}, {"foreign.jsonl": "x"}])
def test_lock_hashes_require_exact_run_local_stream_set(tmp_path, hashes):
    lock_path = tmp_path / "LOCK.json"
    lock_path.write_text(json.dumps({"schema_version": 1, "tau": .6, "calibration_artifact_hashes": hashes}))
    assert not lock.lock_hashes_match(lock_path)
