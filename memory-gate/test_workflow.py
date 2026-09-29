"""W1 and W2 mandatory workflow acceptance tests.

These tests are the definition of done for the composition round.
W1: full calibration workflow through the CLI.
W2: full safety workflow with nested run-dir layout.
"""
from __future__ import annotations

import json
import shutil
import stat
import subprocess
from pathlib import Path

import artifacts
import lock as lock_module
import pytest
import run

HERE = Path(__file__).parent
FIXTURE = HERE / "runs" / "fixture-dev"


def _git(cwd: Path, *args: str, check: bool = True):
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def _build_fake_response(n_candidates: int, scores: list[float] | None = None):
    """Build a fake Jev response with the right structure."""
    if scores is None:
        scores = [0.7] * n_candidates
    return {
        "answers": {
            f"memory_relevance_{i}": {"noul": scores[i]}
            for i in range(n_candidates)
        },
        "configured_model": "test-model",
        "served_model": "test-served",
    }


# ---------------------------------------------------------------------------
# W1: Full calibration workflow
# ---------------------------------------------------------------------------

class TestW1CalibrationWorkflow:
    """candidates CLI -> label-template CLI -> fill labels -> score CLI
    -> report CLI -> lock CLI, in ONE run dir under a git repo mimicking
    memory-gate/runs/<stamp>-<model>/ nesting.
    Candidates are generated via the real CLI path with a fake-pausanias
    executable; labels are derived from captured label-template output;
    candidates.jsonl bytes are proven immutable through scoring.
    artifacts.validate_run passes; the lock verifies."""

    def test_full_calibration_workflow(self, tmp_path, capsys):
        # Set up a git repo with nested run dir: memory-gate/runs/20250101-test-model/
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "test")
        bare = tmp_path / "remote.git"
        bare.mkdir()
        _git(bare, "init", "--bare")
        _git(repo, "remote", "add", "origin", str(bare))

        run_dir = repo / "memory-gate" / "runs" / "20250101-test-model"
        run_dir.mkdir(parents=True)

        # ---- 1. candidates CLI with fake-pausanias executable ----
        # The fake-pausanias script returns two retrieval results for every
        # query, exercising the same subprocess path as the real runner.
        fake_pausanias = tmp_path / "fake-pausanias"
        fake_pausanias.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps(["
            "{'excerpt': 'The cache is in SQLite.', 'path': 'docs/cache.md', 'heading': ['Storage']},"
            "{'excerpt': 'Old cache plan.', 'path': 'docs/old.md', 'heading': ['Old']}"
            "]))\n"
        )
        fake_pausanias.chmod(fake_pausanias.stat().st_mode | stat.S_IXUSR)

        cases = [
            {"case_id": "w1-case-1", "query": "Where is the cache?", "answerable": True},
            {"case_id": "w1-case-2", "query": "Unanswerable question", "answerable": False},
        ]
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(json.dumps(cases))

        candidates_path = run_dir / "candidates.jsonl"
        exit_code = run.main([
            "candidates",
            "--cases", str(cases_file),
            "--output", str(candidates_path),
            "--pausanias-python", str(fake_pausanias),
        ])
        assert exit_code == 0
        assert candidates_path.exists()
        written_candidates = run.read_jsonl(candidates_path)
        assert len(written_candidates) == 4  # 2 cases x 2 candidates each

        # ---- 2. label-template CLI: capture its stdout ----
        capsys.readouterr()  # drain
        exit_code = run.main(["label-template", "--candidates", str(run_dir)])
        assert exit_code == 0
        template_output = capsys.readouterr().out

        # ---- 3. Fill labels from captured template output ----
        template_rows = [json.loads(line) for line in template_output.strip().splitlines()]
        assert len(template_rows) == len(written_candidates)
        for row in template_rows:
            assert row["label"] is None  # template must emit null labels

        labels = []
        for row in template_rows:
            filled = dict(row)
            if "w1-case-1" in row["candidate_id"] and row["rank"] == 0:
                filled["label"] = "positive"
            else:
                filled["label"] = "negative"
            labels.append(filled)
        (run_dir / "labels.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in labels)
        )

        # ---- 4. Snapshot candidates.jsonl bytes before scoring ----
        candidates_bytes_before = candidates_path.read_bytes()

        # Build fake responses: one per case group
        by_case: dict[str, list[dict]] = {}
        for row in written_candidates:
            by_case.setdefault(row["case_id"], []).append(row)
        responses = []
        for group in by_case.values():
            n = len(group)
            scores_list = [0.8, 0.2][:n]
            responses.append(_build_fake_response(n, scores_list))

        response_file = tmp_path / "responses.json"
        response_file.write_text(json.dumps(responses))

        labels_before = (run_dir / "labels.jsonl").read_text()
        exit_code = run.main([
            "score", "--lane", "calibration",
            "--run", str(run_dir),
            "--responses", str(response_file),
            "--model", "test-model",
        ])
        assert exit_code == 0

        # ---- candidates.jsonl must be byte-identical after scoring ----
        assert candidates_path.read_bytes() == candidates_bytes_before, \
            "candidates.jsonl was modified by scoring — immutability violated"

        # Labels must NOT have been overwritten
        assert (run_dir / "labels.jsonl").read_text() == labels_before

        # Scores must exist and match candidates
        assert (run_dir / "scores.jsonl").exists()
        scores = run.read_jsonl(run_dir / "scores.jsonl")
        assert len(scores) == len(written_candidates)

        # Score rows must carry canonical_request_hash + verified identities
        for score_row in scores:
            assert "canonical_request_hash" in score_row
            assert score_row["configured_model_id"] == "test-model"
            assert score_row["served_model_id"] == "test-served"
            assert score_row["coverage"] is True

        # ---- 5. report CLI ----
        exit_code = run.main(["report", "--run", str(run_dir)])
        assert exit_code == 0
        assert (run_dir / "report.md").exists()
        report_content = (run_dir / "report.md").read_text()
        assert "tau" in report_content.lower() or "Threshold" in report_content or "RESULTS" in report_content

        # ---- 6. validate_run must pass ----
        artifacts.validate_run(run_dir)

        # ---- 7. lock CLI ----
        exit_code = run.main(["lock", "--run", str(run_dir), "--tau", "0.6"])
        assert exit_code == 0
        assert (run_dir / "LOCK.json").exists()
        lock_data = json.loads((run_dir / "LOCK.json").read_text())
        assert lock_data["tau"] == 0.6

        # Lock must hash report.md
        assert any("report.md" in str(k) for k in lock_data["calibration_artifact_hashes"])

        # ---- 8. Commit, push, verify witness ----
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "calibration run")
        witness = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _git(repo, "push", "origin", "main")

        verified = run.verify_witness(
            run_dir / "LOCK.json", witness, repo=repo
        )
        assert verified == witness

    def test_score_calibration_refuses_without_labels(self, tmp_path):
        """score --lane calibration must refuse if labels.jsonl is absent."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Write only candidates
        shutil.copy(FIXTURE / "candidates.jsonl", run_dir / "candidates.jsonl")
        response_file = tmp_path / "responses.json"
        response_file.write_text("[]")
        with pytest.raises(ValueError, match="calibration scoring requires labels.jsonl"):
            run.main([
                "score", "--lane", "calibration",
                "--run", str(run_dir),
                "--responses", str(response_file),
                "--model", "m",
            ])

    def test_score_calibration_refuses_template_empty_labels(self, tmp_path):
        """score --lane calibration must refuse labels that are still template (null)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        shutil.copy(FIXTURE / "candidates.jsonl", run_dir / "candidates.jsonl")
        # Write template labels (label: null)
        candidates = run.read_jsonl(run_dir / "candidates.jsonl")
        template_labels = [{**row, "label": None} for row in candidates]
        (run_dir / "labels.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in template_labels)
        )
        response_file = tmp_path / "responses.json"
        response_file.write_text("[]")
        with pytest.raises(ValueError, match="unfilled template labels"):
            run.main([
                "score", "--lane", "calibration",
                "--run", str(run_dir),
                "--responses", str(response_file),
                "--model", "m",
            ])


# ---------------------------------------------------------------------------
# W2: Full safety workflow with nested layout
# ---------------------------------------------------------------------------

def _locomo_fixture(n_conversations: int = 10, n_questions_per: int = 45,
                    last_extra: int = 1, with_retrieval: bool = True) -> list[dict]:
    """Build a locomo-shaped fixture with the correct 10/446 structure.

    10 conversations, 446 total category-5 questions:
    9 conversations x 45 questions + 1 conversation x 41 questions = 446
    Wait, we need exactly 446. 9*49 + 1*5 = 446? No.
    Let's do: 9 * 49 + 1 * 5 = 441+5 = 446. No: 9*49=441.
    Actually: first 9 have 45 each = 405, then 10th has 41 = 446.
    """
    conversations = []
    total = 0
    for i in range(10):
        if i < 9:
            n_q = 45
        else:
            n_q = 446 - total  # last conversation gets the remainder
        questions = []
        for j in range(n_q):
            q: dict = {
                "question": f"Question {j} of conversation {i}",
                "category": 5,
            }
            if with_retrieval:
                q["retrieved"] = [
                    {"excerpt": f"Retrieved text for q{j} conv{i} candidate 0",
                     "path": f"memories/conv{i}/doc{j}.md",
                     "heading": [f"Section {j}"],
                     "retrieval_provenance": {
                         "pipeline": "pausanias",
                         "pipeline_revision": "pinned",
                         "search_config": {"mode": "production"},
                     }},
                ]
            questions.append(q)
        conversations.append({"conversation_id": f"conv-{i}", "qa": questions})
        total += n_q
    assert total == 446
    return conversations


class TestW2SafetyWorkflow:
    """Full safety workflow: pushed lock -> score --lane safety --witness
    on a locomo-shaped fixture WITH production retrieval -> frozen artifact;
    witness verification must succeed with run dir at
    memory-gate/runs/<stamp>-<model>/ depth."""

    def test_full_safety_workflow(self, tmp_path):
        # Create repo with nested run dir
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "test")
        bare = tmp_path / "remote.git"
        bare.mkdir()
        _git(bare, "init", "--bare")
        _git(repo, "remote", "add", "origin", str(bare))

        run_dir = repo / "memory-gate" / "runs" / "20250101-test-model"
        run_dir.mkdir(parents=True)

        # Set up a valid calibration run first (required for lock)
        # Use fixture candidates/scores/labels as valid calibration run
        for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
            shutil.copy(FIXTURE / name, run_dir / name)
        (run_dir / "report.md").write_text("# Test report\n\nSynthetic calibration.\n")

        # Lock the calibration run
        lock_module.write_lock(
            run_dir / "LOCK.json",
            [run_dir / n for n in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")],
            0.55,
        )

        # Commit and push
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "calibration lock")
        witness = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _git(repo, "push", "origin", "main")

        # Build locomo-shaped safety dataset WITH retrieval provenance
        locomo_data = _locomo_fixture(with_retrieval=True)
        cases_file = tmp_path / "locomo-safety.json"
        cases_file.write_text(json.dumps(locomo_data))

        # Build fake responses: one per case (446 cases, each with up to 2 candidates)
        # Each case has 1 candidate, so 446 responses each with 1 score
        responses = [_build_fake_response(1, [0.3]) for _ in range(446)]
        response_file = tmp_path / "responses.json"
        response_file.write_text(json.dumps(responses))

        # score --lane safety --witness
        exit_code = run.main([
            "score", "--lane", "safety",
            "--run", str(run_dir),
            "--cases", str(cases_file),
            "--responses", str(response_file),
            "--model", "test-model",
            "--witness", witness,
        ])
        assert exit_code == 0

        # Frozen safety artifact must exist
        assert (run_dir / "safety.json").exists()
        safety = json.loads((run_dir / "safety.json").read_text())
        assert safety["witness_commit"] == witness
        assert safety["total"] == 446
        assert safety["authoritative"] is True

        # Witness verification must succeed at nested depth
        verified = run.verify_witness(
            run_dir / "LOCK.json", witness, repo=repo
        )
        assert verified == witness

    def test_safety_refuses_dataset_without_retrieval_provenance(self, tmp_path):
        """A locomo dataset WITHOUT retrieval fields must be refused.
        This is the 0/446 false-pass hole.

        The refusal must come from safety_cases (retrieval enforcement),
        NOT from verify_witness failing on a missing LOCK.json.
        """
        # ---- Build a valid pushed lock + witness (reuse W2 fixture machinery) ----
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "test")
        bare = tmp_path / "remote.git"
        bare.mkdir()
        _git(bare, "init", "--bare")
        _git(repo, "remote", "add", "origin", str(bare))

        run_dir = repo / "memory-gate" / "runs" / "20250101-test-model"
        run_dir.mkdir(parents=True)

        # Valid calibration run for the lock
        for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
            shutil.copy(FIXTURE / name, run_dir / name)
        (run_dir / "report.md").write_text("# Test report\n\nSynthetic calibration.\n")
        lock_module.write_lock(
            run_dir / "LOCK.json",
            [run_dir / n for n in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")],
            0.55,
        )
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "calibration lock")
        witness = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _git(repo, "push", "origin", "main")

        # ---- Dataset WITHOUT retrieval fields ----
        locomo_no_retrieval = _locomo_fixture(with_retrieval=False)
        cases_file = tmp_path / "locomo-no-retrieval.json"
        cases_file.write_text(json.dumps(locomo_no_retrieval))

        response_file = tmp_path / "responses.json"
        response_file.write_text("[]")

        # ---- Narrow assertion: must match retrieval-provenance error from safety_cases ----
        with pytest.raises(ValueError, match="no.*'retrieved' field.*retrieval never ran"):
            run.main([
                "score", "--lane", "safety",
                "--run", str(run_dir),
                "--cases", str(cases_file),
                "--responses", str(response_file),
                "--model", "m",
                "--witness", witness,
            ])

    def test_zero_candidate_without_retrieval_provenance_is_refused(self, tmp_path):
        """Cases with zero candidates must have retrieval provenance to prove
        retrieval ran and genuinely found nothing (not 'retrieval never ran')."""
        # A question that looks correctly shaped but has no retrieved field at all
        locomo = _locomo_fixture(with_retrieval=True)
        # Remove retrieved from first question to simulate "retrieval never ran"
        del locomo[0]["qa"][0]["retrieved"]
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(json.dumps(locomo))

        with pytest.raises(ValueError, match="no.*'retrieved' field.*retrieval never ran"):
            run.safety_cases(locomo)


# ---------------------------------------------------------------------------
# Blocker 3: Witness path tests
# ---------------------------------------------------------------------------

class TestWitnessNestedPath:
    """Witness verification must work with nested runs/<stamp>-<model>/ layouts
    by computing repo root via git rev-parse --show-toplevel."""

    def test_nested_run_dir_witness_verifies(self, tmp_path):
        """The witness path must not use run_dir.parent; it must find the repo root."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "test")
        bare = tmp_path / "remote.git"
        bare.mkdir()
        _git(bare, "init", "--bare")
        _git(repo, "remote", "add", "origin", str(bare))

        # Create deeply nested run dir
        run_dir = repo / "memory-gate" / "runs" / "20250101-test-model"
        run_dir.mkdir(parents=True)
        for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
            shutil.copy(FIXTURE / name, run_dir / name)
        (run_dir / "report.md").write_text("# Test\n")

        lock_module.write_lock(
            run_dir / "LOCK.json",
            [run_dir / n for n in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")],
            0.6,
        )

        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "nested lock")
        witness = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _git(repo, "push", "origin", "main")

        # This must succeed — repo root is repo/, not run_dir.parent
        verified = run.verify_witness(run_dir / "LOCK.json", witness, repo=repo)
        assert verified == witness


# ---------------------------------------------------------------------------
# Blocker 4: Report authority tests
# ---------------------------------------------------------------------------

class TestReportAuthority:
    """report.md is the authoritative report; lock must hash it."""

    def test_report_writes_report_md_not_just_results_txt(self, tmp_path):
        """The report command must write report.md (authoritative)."""
        run_dir = tmp_path / "run"
        shutil.copytree(FIXTURE, run_dir)
        exit_code = run.main(["report", "--run", str(run_dir)])
        assert exit_code == 0
        assert (run_dir / "report.md").exists()
        content = (run_dir / "report.md").read_text()
        # Must contain actual metrics/sweep data
        assert "tau" in content.lower() or "RESULTS" in content

    def test_lock_hashes_report_md(self, tmp_path):
        """LOCK.json must include report.md in its hashes."""
        run_dir = tmp_path / "run"
        shutil.copytree(FIXTURE, run_dir)
        lock_path = run_dir / "LOCK.json"
        lock_module.write_lock(
            lock_path,
            [run_dir / n for n in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")],
            0.6,
        )
        data = json.loads(lock_path.read_text())
        assert any("report.md" in str(k) for k in data["calibration_artifact_hashes"])

    def test_report_md_content_change_breaks_lock(self, tmp_path):
        """Changing report.md after lock must break verification."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "test")
        bare = tmp_path / "remote.git"
        bare.mkdir()
        _git(bare, "init", "--bare")
        _git(repo, "remote", "add", "origin", str(bare))

        run_dir = repo / "run"
        shutil.copytree(FIXTURE, run_dir)
        lock_module.write_lock(
            run_dir / "LOCK.json",
            [run_dir / n for n in ("candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md")],
            0.6,
        )
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "lock")
        witness = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _git(repo, "push", "origin", "main")

        # Modify report.md
        (run_dir / "report.md").write_text("TAMPERED REPORT\n")

        with pytest.raises(ValueError, match="hash mismatch"):
            run.verify_witness(run_dir / "LOCK.json", witness, repo=repo)


# ---------------------------------------------------------------------------
# Blocker 2: Retrieval enforcement tests
# ---------------------------------------------------------------------------

class TestRetrievalEnforcement:
    """Safety cases must REFUSE questions lacking retrieval output."""

    def test_safety_cases_refuses_missing_retrieved_field(self):
        """A question without 'retrieved' field must be refused."""
        dataset = _locomo_fixture(with_retrieval=True)
        # Remove all retrieved fields
        for conv in dataset:
            for q in conv["qa"]:
                q.pop("retrieved", None)
        with pytest.raises(ValueError, match="no.*'retrieved' field.*retrieval never ran"):
            run.safety_cases(dataset)

    def test_safety_cases_accepts_genuine_zero_retrieval_with_provenance(self):
        """A question with empty retrieved list + retrieval_provenance is OK
        (genuine zero-retrieval from the pinned pausanias pipeline)."""
        dataset = _locomo_fixture(with_retrieval=True)
        # Set first question to have empty retrieval but with provenance
        dataset[0]["qa"][0]["retrieved"] = []
        dataset[0]["qa"][0]["retrieval_provenance"] = {
            "pipeline": "pausanias",
            "pipeline_revision": "pinned",
        }
        # This should succeed (the question is a genuine zero-retrieval result)
        cases = run.safety_cases(dataset)
        assert len(cases) == 446
