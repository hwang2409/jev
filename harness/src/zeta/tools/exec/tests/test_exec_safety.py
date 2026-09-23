from __future__ import annotations


import io


from pathlib import Path


import pytest


from evals.run_evals import parse_events


from zeta.cli import build_parser


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.loop import AgentLoop


from zeta.core.project_context import ProjectContext


from zeta.core.safety import (
    _LAYER0_RULES,
    SafetyTier,
    _resolved_argv,
    layer0_classify,
    layer0_reason,
)


from zeta.core.session import SessionManager


from zeta.core.store import ConversationStore


from zeta.providers import jev


from zeta.runtime.composition import compose_runtime


from zeta.runtime.driver import drive_turn


from zeta.settings import ResolvedConfig


from zeta.skills import SkillCatalog


from zeta.skills.agent_catalog import AgentCatalog


from zeta.tools import ToolRegistry


from zeta.tools.exec import run_inline_shell_batch


from zeta.types import TextContent, ToolCall


def _score(
    score: int,
    confidence: float,
    *,
    outside: float = 0.1,
    irreversible: float = 0.1,
) -> jev.SafetyScoreResult:
    return jev.SafetyScoreResult(
        score,
        {str(score): 1.0},
        confidence,
        outside,
        irreversible,
        {"input_tokens": 1, "output_tokens": 1},
        min(confidence, abs(outside - 0.5) * 2, abs(irreversible - 0.5) * 2),
    )


@pytest.mark.asyncio
async def test_inline_batch_gates_every_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jev_calls: list[str] = []

    async def score(*args: object) -> jev.SafetyScoreResult:
        jev_calls.append(str(args[0]))
        return _score(0, 0.95)

    monkeypatch.setattr(jev, "safety_score", score)
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
        safety_tier=SafetyTier(cwd=tmp_path, headless=True),
        skill_catalog=SkillCatalog.empty(),
    )

    outputs = await run_inline_shell_batch(
        registry,
        ("printf safe", "sudo id"),
        lifecycle_sink=lambda *_args: None,
    )

    assert jev_calls == ["printf safe"]
    assert outputs[0] == "safe"
    assert outputs[1].startswith("[inline shell failed: canceled]")


@pytest.mark.asyncio
async def test_bash_safety_uses_persistent_execution_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen_cwds: list[str] = []

    async def score(*args: object) -> jev.SafetyScoreResult:
        seen_cwds.append(str(args[1]))
        return _score(0, 0.95)

    monkeypatch.setattr(jev, "safety_score", score)
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path, bash_cwd="/etc")
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
        session_store=store,
        safety_tier=SafetyTier(cwd=tmp_path),
        skill_catalog=SkillCatalog.empty(),
    )
    registry.update_bash_cwd("/etc")

    result = await registry.execute(
        ToolCall("bash-cwd", "bash", {"command": "printf safe"})
    )

    assert result["isError"] is False
    assert seen_cwds == [str(Path("/etc").resolve())]


@pytest.mark.asyncio
async def test_non_shell_tool_does_not_touch_safety_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tier = SafetyTier(cwd=tmp_path)
    monkeypatch.setattr(
        tier,
        "command_cwd",
        lambda _arguments: (_ for _ in ()).throw(AssertionError("touched")),
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
        safety_tier=tier,
        skill_catalog=SkillCatalog.empty(),
        register_builtin=False,
    )
    registry.register("custom", lambda _arguments: "ran")

    result = await registry.execute(
        ToolCall("custom-fields", "custom", {"command": object(), "cwd": object()})
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "ran"
