from __future__ import annotations

import shutil
from pathlib import Path

import lock

HERE = Path(__file__).parent


def test_lock_hash_predicate_and_safety_output_predicate(tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(HERE / "runs" / "fixture-dev", run_dir)
    artifact_paths = [run_dir / name for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")]
    lock_path = run_dir / "LOCK.json"
    lock.write_lock(lock_path, artifact_paths, 0.6)
    assert lock.lock_hashes_match(lock_path)
    assert lock.no_preexisting_safety_outputs([run_dir / "safety.json"])
    (run_dir / "safety.json").write_text("already frozen")
    assert not lock.no_preexisting_safety_outputs([run_dir / "safety.json"])
    artifact_paths[2].write_text("tampered")
    assert not lock.lock_hashes_match(lock_path)
