from __future__ import annotations

import importlib.util
from pathlib import Path

import artifacts
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("memory_gate_metrics_test_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(run)
import metrics

FIXTURE = HERE / "runs" / "fixture-dev"


def test_fixture_hand_computed_metrics_and_all_paths():
    cases = metrics.load_run(FIXTURE)
    row = metrics.metrics(cases, .6)
    assert row["any_injection_rate"] == 4 / 8
    assert row["packet_recall"] == 1
    assert row["packet_precision"] == 1
    assert row["exact_packet_rate"] == 1
    assert row["forbidden_injection_rate"] == 0
    assert row["forbidden_injection_blocks"] == 0
    assert row["retrieval_miss_rate"] == 1 / 4
    assert row["ambiguous_pairs"] == 1
    assert row["empty_injection_cases"] == 4
    assert row["retention"] == 1
    assert row["prevalence"] == 3 / 9
    assert row["brier"] == pytest.approx((.1**2 + .1**2 + .3**2 + .55**2 + .5**2 + 0**2 + .2**2 + .15**2 + .55**2) / 9)
    assert metrics.metrics(cases, .9)["packet_precision"] is None
    assert metrics.metrics(cases, 0)["any_injection_rate"] > 0
    assert metrics.metrics(cases, 0)["any_injection_rate"] < 1


def test_bootstrap_recomputes_ratios_and_is_deterministic():
    cases = metrics.load_run(FIXTURE)
    first = metrics.metrics(cases, .6)["bootstrap"]
    second = metrics.metrics(cases, .6)["bootstrap"]
    assert first == second
    assert all(x["replicates"] == 10_000 for x in first.values())
    assert first["packet_precision"]["null_replicates"] > 0
    result = metrics.bootstrap_cases([], lambda _: None, replicates=10_000)
    assert result == {"ci": None, "null_replicates": 10_000, "replicates": 10_000}


def test_wilson_known_safety_interval():
    assert metrics.wilson(6, 446) == pytest.approx([0.0061798, 0.0290359], abs=2e-5)


def test_invalid_run_is_rejected_before_computation(tmp_path):
    with pytest.raises(artifacts.ArtifactValidationError):
        metrics.compute_run(tmp_path, .6)
