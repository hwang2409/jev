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
    first = (target / "results.txt").read_text()
    assert "RESULTS — memory-gate metrics" in first
    assert "tau | any-injection | recall | precision" in first
    capsys.readouterr()
    assert run.main(["report", "--run", str(target)]) == 0
    assert (target / "results.txt").read_text() == first


def test_report_refuses_invalid_run(tmp_path):
    with pytest.raises(artifacts.ArtifactValidationError):
        run.main(["report", "--run", str(tmp_path)])
