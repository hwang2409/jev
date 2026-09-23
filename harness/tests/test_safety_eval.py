from __future__ import annotations

import argparse

import pytest

import evals.run_safety_eval as safety_eval


@pytest.mark.asyncio
async def test_live_safety_eval_calls_client_path_without_preflight(
    monkeypatch, capsys
) -> None:
    async def offline() -> dict[str, object]:
        return {
            "corpus_rows": 0,
            "safety_recall": 1.0,
            "safety_recall_counts": {"protected": 0, "dangerous": 0},
            "layer0_exact": 1.0,
            "layer0_exact_counts": {"exact": 0, "rows": 0},
            "benign_auto_approve_rate": 1.0,
            "false_escalate_rate": 0.0,
            "per_category": {},
            "dangerous_auto_approved": [],
        }

    monkeypatch.setattr(safety_eval, "run_offline", offline)
    called = False

    async def live_smoke() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(safety_eval, "run_live_smoke", live_smoke)

    assert await safety_eval._async_main(argparse.Namespace(live=True)) == 0
    assert called is True
    assert "live Jev smoke skipped" not in capsys.readouterr().out
