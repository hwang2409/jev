from __future__ import annotations

import pytest

from evals.run_safety_eval import run_offline


@pytest.mark.asyncio
async def test_committed_safety_corpus_has_perfect_safety_recall(tmp_path) -> None:
    summary = await run_offline(output_path=tmp_path / "safety-acceptance.json")

    assert summary["safety_recall"] == 1.0
    assert summary["dangerous_auto_approved"] == []
