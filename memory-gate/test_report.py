from __future__ import annotations

from pathlib import Path

import artifacts
import pytest
import run

FIXTURE = Path(__file__).parent / "runs" / "fixture-dev"


def test_report_snapshot_is_deterministic(tmp_path, capsys):
    import shutil
    target = tmp_path / "run"
    shutil.copytree(FIXTURE, target)
    assert run.main(["report", "--run", str(target)]) == 0
    first = (target / "results.txt").read_bytes()
    expected = (Path(__file__).with_name("testdata-report-fixture.txt")).read_bytes()
    assert first == expected
    # report.md is the authoritative copy (Blocker 4)
    assert (target / "report.md").read_bytes() == expected
    assert b"\\\\n" not in first
    assert b"Bootstrap 95% CIs (10,000 resamples; seed 20260929; percentile):" in first
    assert b"precision=null" in first
    capsys.readouterr()
    assert run.main(["report", "--run", str(target)]) == 0
    assert (target / "results.txt").read_bytes() == expected
    assert (target / "report.md").read_bytes() == expected


def test_report_refuses_invalid_run(tmp_path):
    with pytest.raises(artifacts.ArtifactValidationError):
        run.main(["report", "--run", str(tmp_path)])
