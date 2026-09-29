from __future__ import annotations

from pathlib import Path

import metrics
import sweep

FIXTURE = Path(__file__).parent / "runs" / "fixture-dev"


def test_sweep_boundaries_frontier_and_no_qualifying_branch():
    cases = metrics.load_run(FIXTURE)
    result = sweep.sweep(cases)
    assert result["thresholds"] == [0, .1, .2, .5, .55, .6, .65, .7, .85, .9]
    assert [x["tau"] for x in result["pareto"]] == [.65, .7, .85, .9]
    assert result["selection"]["selected_tau"] is None
    assert result["selection"]["reason"] == "no qualifying tau"
    assert sweep.select_tau([{"tau": .4, "any_injection_rate": 0, "packet_recall": .95, "packet_recall_baseline": 1}])["selected_tau"] == .4
