from __future__ import annotations


import io


from pathlib import Path


import pytest


from evals.run_evals import parse_events


from zeta.cli.main import build_parser


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.loop import AgentLoop


from zeta.core.project_context import ProjectContext


from zeta.core.safety import (
    _LAYER0_RULES,
    BrowserRiskEvidence,
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


from zeta.config.settings import ResolvedConfig


from zeta.skills import SkillCatalog


from zeta.skills.agent_catalog import AgentCatalog


from zeta.tools import ToolRegistry


from zeta.tools.exec import run_inline_shell_batch


from zeta.protocol.types import TextContent, ToolCall


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
@pytest.mark.parametrize(
    ("score", "confidence", "headless", "decision"),
    [
        (0, 0.9, False, "allow"),
        (1, 0.9, False, "allow"),
        (2, 0.9, False, "ask"),
        (3, 0.9, False, "ask"),
        (1, 0.7, False, "ask"),
        (1, 0.8, False, "allow"),
        (1, 0.799, False, "ask"),
        (2, 0.9, True, "deny"),
        (3, 0.9, True, "deny"),
    ],
)
async def test_decision_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    score: int,
    confidence: float,
    headless: bool,
    decision: str,
) -> None:
    async def score_result(*_args: object) -> jev.SafetyScoreResult:
        return _score(score, confidence)

    monkeypatch.setattr(jev, "safety_score", score_result)
    tier = SafetyTier(cwd=tmp_path, headless=headless)

    outcome = await tier.evaluate("bash", "printf safe", str(tmp_path))

    assert outcome.decision == decision


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [False, True])
async def test_jev_errors_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, headless: bool
) -> None:
    async def fail(*_args: object) -> jev.SafetyScoreResult:
        raise jev.JevRouterError("offline")

    monkeypatch.setattr(jev, "safety_score", fail)
    outcome = await SafetyTier(cwd=tmp_path, headless=headless).evaluate(
        "exec", "printf safe", str(tmp_path)
    )

    assert outcome.decision == ("deny" if headless else "ask")
    assert outcome.layer == "jev_error_failclosed"


@pytest.mark.asyncio
async def test_browser_risk_ignores_page_safe_label_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail(*_args: object, **_kwargs: object) -> jev.SafetyScoreResult:
        raise jev.JevRouterError("offline")

    monkeypatch.setattr(jev, "safety_score", fail)
    tier = SafetyTier(cwd=tmp_path, headless=True)
    outcome = await tier.evaluate_browser_action(
        BrowserRiskEvidence(
            action="click",
            role="button",
            text="safe approved click",
            current_origin="https://example.test",
            target_url="https://other.test/confirm",
            form_action_origin=None,
            payment_language=False,
            authentication_language=False,
            download=False,
            durable_state_change=False,
        )
    )

    assert outcome.decision == "deny"
    assert outcome.layer == "jev_error_failclosed"


@pytest.mark.asyncio
async def test_browser_external_origin_cannot_be_auto_approved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def safe(*_args: object, **_kwargs: object) -> jev.SafetyScoreResult:
        return _score(0, 0.99)

    monkeypatch.setattr(jev, "safety_score", safe)
    outcome = await SafetyTier(cwd=tmp_path).evaluate_browser_action(
        BrowserRiskEvidence(
            action="click",
            role="link",
            text="safe link",
            current_origin="https://example.test",
            target_url="https://other.test/next",
            form_action_origin=None,
            payment_language=False,
            authentication_language=False,
            download=False,
            durable_state_change=False,
        )
    )

    assert outcome.decision == "ask"
    assert outcome.layer == "layer0"
    assert outcome.reason == "external_origin"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("evidence_update", "reason"),
    [
        ({"payment_language": True}, "payment_or_financial_commitment"),
        ({"authentication_language": True}, "authentication_or_permission_change"),
        ({"download": True}, "download"),
        ({"durable_state_change": True}, "durable_state_change"),
        ({"action": "click", "text": "Delete this record"}, "destructive_action"),
    ],
)
async def test_browser_risky_evidence_cannot_be_auto_approved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evidence_update: dict[str, object],
    reason: str,
) -> None:
    async def safe(*_args: object, **_kwargs: object) -> jev.SafetyScoreResult:
        return _score(0, 0.99)

    monkeypatch.setattr(jev, "safety_score", safe)
    evidence = {
        "action": "click",
        "role": "button",
        "text": "safe page label",
        "current_origin": "https://example.test",
        "target_url": None,
        "form_action_origin": None,
        "payment_language": False,
        "authentication_language": False,
        "download": False,
        "durable_state_change": False,
    }
    evidence.update(evidence_update)

    outcome = await SafetyTier(cwd=tmp_path).evaluate_browser_action(
        BrowserRiskEvidence(**evidence)
    )

    assert outcome.decision == "ask"
    assert outcome.layer == "layer0"
    assert outcome.reason == reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), KeyError("answers"), ValueError("malformed")],
)
async def test_all_jev_failure_shapes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: BaseException,
) -> None:
    async def fail(*_args: object) -> jev.SafetyScoreResult:
        raise error

    monkeypatch.setattr(jev, "safety_score", fail)
    outcome = await SafetyTier(cwd=tmp_path, headless=True).evaluate(
        "exec", "printf safe", str(tmp_path)
    )

    assert outcome.decision == "deny"
    assert outcome.layer == "jev_error_failclosed"


@pytest.mark.asyncio
async def test_telemetry_has_named_skip_reason_and_trigger(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    tier = SafetyTier(cwd=tmp_path, telemetry=events.append)

    await tier.evaluate("bash", "sudo true", str(tmp_path))

    assert events[0]["skip_reason"] == "layer0_escalated"
    assert events[0]["trigger"] == "sudo"


@pytest.mark.asyncio
async def test_headless_teaching_error_contains_score_and_next_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def score_result(*_args: object) -> jev.SafetyScoreResult:
        return _score(2, 0.95)

    monkeypatch.setattr(jev, "safety_score", score_result)
    tier = SafetyTier(cwd=tmp_path, headless=True)

    outcome = await tier.evaluate("exec", "rm -rf build", str(tmp_path))

    message = tier.teaching_error(outcome)
    assert "score=2" in message
    assert "scoped destructive action" in message
    assert "narrow the command or ask the user" in message


@pytest.mark.asyncio
async def test_teaching_error_names_low_confidence_trigger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def score_result(*_args: object) -> jev.SafetyScoreResult:
        return _score(0, 0.02)

    monkeypatch.setattr(jev, "safety_score", score_result)
    tier = SafetyTier(cwd=tmp_path, headless=True)
    outcome = await tier.evaluate("exec", "printf safe", str(tmp_path))

    assert "trigger=low_confidence" in tier.teaching_error(outcome)


@pytest.mark.asyncio
async def test_teaching_error_names_score_trigger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def score_result(*_args: object) -> jev.SafetyScoreResult:
        return _score(2, 0.95)

    monkeypatch.setattr(jev, "safety_score", score_result)
    tier = SafetyTier(cwd=tmp_path, headless=True)
    outcome = await tier.evaluate("exec", "printf safe", str(tmp_path))

    assert "trigger=score_exceeds" in tier.teaching_error(outcome)


@pytest.mark.asyncio
async def test_safety_runs_only_after_yolo_would_allow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    async def score(*args: object) -> jev.SafetyScoreResult:
        calls.append(str(args[0]))
        return _score(0, 0.95)

    monkeypatch.setattr(jev, "safety_score", score)
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.DENY)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
        safety_tier=SafetyTier(cwd=tmp_path),
        skill_catalog=SkillCatalog.empty(),
        register_builtin=False,
    )
    registry.register("custom", lambda _arguments: "ran")

    result = await registry.execute(ToolCall("call", "custom", {}))

    assert result["isError"] is True
    assert calls == []


def test_safety_tier_flag_and_child_inheritance(tmp_path: Path) -> None:
    assert build_parser().parse_args(["--safety-tier"]).safety_tier is True
    assert build_parser().parse_args(["--no-safety-tier"]).safety_tier is False

    store = ConversationStore(tmp_path / "parent", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW)
    tier = SafetyTier(cwd=tmp_path)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
        safety_tier=tier,
        skill_catalog=SkillCatalog.empty(),
        register_builtin=False,
    )
    child_store = ConversationStore(tmp_path / "child", cwd=tmp_path)

    child = registry.clone_for_session(child_store)

    assert child.safety_tier is not tier
    assert child.safety_tier is not None
    tier.set_task_excerpt("parent task")
    assert child.safety_tier.task_excerpt == ""


@pytest.mark.asyncio
async def test_off_flag_preserves_pre_feature_provider_and_event_bytes(
    tmp_path: Path,
) -> None:
    turns = [
        ScriptedTurn(tool_calls=[ToolCall("call", "bash", {"command": "printf safe"})]),
        ScriptedTurn([TextContent("done")]),
    ]

    baseline_backend = FakeBackend(turns)
    baseline_store = ConversationStore(tmp_path / "baseline", cwd=tmp_path)
    baseline_policy = ApprovalPolicy(
        store=baseline_store, default=ApprovalDecision.ALLOW
    )
    baseline_registry = ToolRegistry(
        tmp_path,
        approval_policy=baseline_policy,
        approval_store=baseline_store,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
    )
    baseline_loop = AgentLoop(
        baseline_backend,
        baseline_store,
        registry=baseline_registry,
        approval_policy=baseline_policy,
        router_mode=False,
        router_style="tool",
        jev_compaction=False,
        memory_injection=False,
        system_prompt="",
        skill_catalog=SkillCatalog.empty(),
    )
    baseline_output = io.StringIO()
    await drive_turn(
        baseline_loop,
        "run it",
        format="json",
        stdout=baseline_output,
        stderr=io.StringIO(),
    )
    baseline_store.close()

    config = ResolvedConfig(
        provider="fake",
        model="offline",
        router=False,
        router_style="tool",
        jev_compaction=False,
        memory_injection=False,
        yolo=True,
        safety_tier=False,
        token_budget=None,
        theme=None,
        approval_allow=(),
        approval_deny=(),
        approval_ask=(),
        keybindings={},
    )
    off_backend = FakeBackend(turns)
    manager = SessionManager(tmp_path / "off-home")
    composition = compose_runtime(
        home=tmp_path / "off-home",
        cwd=tmp_path,
        manager=manager,
        config=config,
        provider="fake",
        model="offline",
        project_context=ProjectContext("", ()),
        backend_builder=lambda *_args, **_kwargs: (off_backend, "offline"),
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
    )
    off_output = io.StringIO()
    try:
        assert composition.loop.tool_registry.safety_tier is None
        await drive_turn(
            composition.loop,
            "run it",
            format="json",
            stdout=off_output,
            stderr=io.StringIO(),
        )
    finally:
        composition.opened.store.close()

    assert off_backend.request_bytes == baseline_backend.request_bytes
    assert off_output.getvalue().encode() == baseline_output.getvalue().encode()


@pytest.mark.asyncio
async def test_missing_jev_api_key_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("JEV_API_KEY", raising=False)

    outcome = await SafetyTier(cwd=tmp_path, headless=True).evaluate(
        "exec", "printf safe", str(tmp_path)
    )

    assert outcome.decision == "deny"
    assert outcome.layer == "jev_error_failclosed"


@pytest.mark.asyncio
async def test_runtime_composition_wires_safety_usage_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def score(*_args: object) -> jev.SafetyScoreResult:
        return _score(0, 0.95)

    monkeypatch.setattr(jev, "safety_score", score)
    events: list[object] = []
    config = ResolvedConfig(
        provider="fake",
        model="offline",
        router=False,
        router_style="tool",
        jev_compaction=False,
        memory_injection=False,
        yolo=True,
        safety_tier=True,
        token_budget=None,
        theme=None,
        approval_allow=(),
        approval_deny=(),
        approval_ask=(),
        keybindings={},
    )
    manager = SessionManager(tmp_path / "home")
    composition = compose_runtime(
        home=tmp_path / "home",
        cwd=tmp_path,
        manager=manager,
        config=config,
        provider="fake",
        model="offline",
        project_context=ProjectContext("", ()),
        backend_builder=lambda *_args, **_kwargs: (FakeBackend([]), "offline"),
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
    )
    try:
        composition.loop.set_background_event_sink(events.append)
        tier = composition.loop.tool_registry.safety_tier
        assert tier is not None
        await tier.evaluate("exec", "printf safe", str(tmp_path))
    finally:
        composition.opened.store.close()

    assert len(events) == 1
    event = events[0]
    assert event.type.value == "usage"
    assert event.data["service"] == "jev"
    assert event.data["usage"] == {"input_tokens": 1, "output_tokens": 1}
    assert parse_events([{"type": "usage", **event.data}])["jev_tokens"] == 2
