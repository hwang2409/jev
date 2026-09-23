from __future__ import annotations

import pytest

import evals.run_safety_eval as safety_eval
from evals.run_safety_eval import run_offline


@pytest.mark.asyncio
async def test_committed_safety_corpus_has_perfect_safety_recall(tmp_path) -> None:
    summary = await run_offline(output_path=tmp_path / "safety-acceptance.json")

    assert summary["safety_recall"] == 1.0
    assert summary["dangerous_auto_approved"] == []


@pytest.mark.asyncio
async def test_live_safety_eval_uses_gateway_key_resolution(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(safety_eval.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("JEV_API_KEY", "stale-native-key")
    for name in safety_eval.jev.GATEWAY_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)

    async def fake_run_offline() -> dict[str, object]:
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

    monkeypatch.setattr(safety_eval, "run_offline", fake_run_offline)
    live_calls = 0

    async def fake_live_smoke() -> None:
        nonlocal live_calls
        live_calls += 1

    monkeypatch.setattr(safety_eval, "run_live_smoke", fake_live_smoke)

    assert await safety_eval._async_main(safety_eval.argparse.Namespace(live=True)) == 0
    assert live_calls == 0
    assert "live Jev smoke skipped" in capsys.readouterr().out
