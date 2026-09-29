from __future__ import annotations

import lock


def test_lock_hash_predicate_and_safety_output_predicate(tmp_path):
    artifact = tmp_path / "scores.jsonl"
    artifact.write_text("synthetic")
    lock_path = tmp_path / "LOCK.json"
    lock.write_lock(lock_path, [artifact], 0.6)
    assert lock.lock_hashes_match(lock_path)
    assert lock.no_preexisting_safety_outputs([tmp_path / "safety.json"])
    (tmp_path / "safety.json").write_text("already frozen")
    assert not lock.no_preexisting_safety_outputs([tmp_path / "safety.json"])
    artifact.write_text("tampered")
    assert not lock.lock_hashes_match(lock_path)
