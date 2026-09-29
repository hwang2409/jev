"""LOCK.json creation and the four independent §8.3 witness predicates."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path


class WitnessError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_lock(lock_path: Path, calibration_artifacts: Sequence[Path], tau: float) -> dict:
    payload = {"schema_version": 1, "tau": tau, "calibration_artifact_hashes": {str(p): sha256_file(p) for p in calibration_artifacts}}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def lock_hashes_match(lock_path: Path) -> bool:
    try:
        data = json.loads(lock_path.read_bytes())
        hashes = data["calibration_artifact_hashes"]
        required = {"candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md"}
        if not isinstance(hashes, dict) or len(hashes) != len(required):
            return False
        run_dir = lock_path.parent.resolve()
        normalized = {}
        for raw_path, digest in hashes.items():
            path = Path(raw_path).resolve()
            if path.parent != run_dir or path.name not in required or path.name in normalized:
                return False
            normalized[path.name] = path
            if not path.is_file() or sha256_file(path) != digest:
                return False
        if set(normalized) != required:
            return False
        from artifacts import validate_run
        validate_run(run_dir)
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def lock_contained_in_witness(lock_path: Path, witness: str, repo: Path) -> bool:
    try:
        rel = lock_path.resolve().relative_to(repo.resolve()).as_posix()
        result = subprocess.run(["git", "show", f"{witness}:{rel}"], cwd=repo, capture_output=True, check=False)
        return result.returncode == 0 and result.stdout == lock_path.read_bytes()
    except (OSError, ValueError):
        return False


def witness_reachable_after_fresh_fetch(witness: str, repo: Path, remote: str = "origin") -> bool:
    fetched = subprocess.run(["git", "fetch", "--prune", remote], cwd=repo, capture_output=True, check=False)
    if fetched.returncode:
        return False
    refs = subprocess.run(["git", "for-each-ref", "--format=%(refname)", f"refs/remotes/{remote}/"], cwd=repo, text=True, capture_output=True, check=False)
    if refs.returncode:
        return False
    return any(subprocess.run(["git", "merge-base", "--is-ancestor", witness, ref], cwd=repo, capture_output=True, check=False).returncode == 0 for ref in refs.stdout.splitlines())


def no_preexisting_safety_outputs(outputs: Sequence[Path]) -> bool:
    return not any(path.exists() for path in outputs)


def verify_witness(lock_path: Path, witness: str, *, remote: str = "origin", safety_outputs: Sequence[Path] = (), repo: Path | None = None) -> str:
    """Run predicates in order and return the verified witness commit hash."""
    repo = repo or Path.cwd()
    if not lock_path.exists():
        raise WitnessError("LOCK.json does not exist")
    resolved = subprocess.run(["git", "rev-parse", "--verify", f"{witness}^{{commit}}"], cwd=repo, text=True, capture_output=True, check=False)
    if resolved.returncode:
        raise WitnessError("witness is not a valid commit")
    witness_commit = resolved.stdout.strip()
    if not lock_contained_in_witness(lock_path, witness_commit, repo):
        raise WitnessError("witness does not contain exact LOCK.json bytes")
    if not lock_hashes_match(lock_path):
        raise WitnessError("calibration artifact hash mismatch (exact LOCK.json bytes are present)")
    if not witness_reachable_after_fresh_fetch(witness_commit, repo, remote):
        raise WitnessError("witness is not reachable from a remote-tracking ref after fresh fetch")
    if not no_preexisting_safety_outputs(safety_outputs):
        raise WitnessError("safety outputs already exist")
    return witness_commit
