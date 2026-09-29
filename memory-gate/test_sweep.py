from __future__ import annotations

from pathlib import Path

import metrics
import sweep

FIXTURE = Path(__file__).parent / "runs" / "fixture-dev"


def test_sweep_boundaries_use_abstain_stratum_and_explicit_frontier():
    cases = metrics.load_run(FIXTURE)
    result = sweep.sweep(cases)
    assert result["thresholds"] == [0, .1, .2, .5, .55, .6, .65, .7, .85, .9]
    assert [(x["tau"], x["any_injection_rate_by_stratum"]["abstain"]["rate"], x["packet_recall"]) for x in result["pareto"]] == [
        (.55, 0, 1),
        (.6, 0, 1),
        (.65, 0, 1),
    ]
    assert result["selection"] == {"selected_tau": .55, "reason": "smallest qualifying tau", "baseline_recall": 1}
    abstain_rates = [row["any_injection_rate_by_stratum"]["abstain"]["rate"] for row in result["table"]]
    assert abstain_rates == [.75, .75, .5, .25, 0, 0, 0, 0, 0, 0]
