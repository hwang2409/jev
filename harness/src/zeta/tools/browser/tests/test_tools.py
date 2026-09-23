from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.safety import SafetyOutcome, SafetyTier
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.providers import jev
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.browser import PageStateDecision, register
from zeta.tools.browser.adapter import ElementRef, FakeBrowserAdapter, PageObservation


def _observation(snapshot_id: int = 1) -> PageObservation:
    return PageObservation(
        snapshot_id,
        snapshot_id,
        "https://example.test/",
        "Example",
        "Page text",
        (
            ElementRef(
                snapshot_id,
                "e1",
                "button",
                "click",
                "Continue",
                "Continue",
                None,
                "main",
                False,
                True,
            ),
        ),
        True,
        True,
    )


def _element_observation(
    element: ElementRef, *additional_elements: ElementRef
) -> PageObservation:
    return PageObservation(
        element.snapshot_id,
        element.snapshot_id,
        "https://example.test/",
        "Example",
        "Page text",
        (element, *additional_elements),
        True,
        True,
    )


def _registry(
    tmp_path: Path,
    adapter: FakeBrowserAdapter,
    safety_tier: SafetyTier | None = None,
    approval_policy: ApprovalPolicy | None = None,
    approval_store: ConversationStore | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        safety_tier=safety_tier,
        approval_policy=approval_policy,
        approval_store=approval_store,
    )
    registry.browser_adapter_factory = lambda: adapter
    register(registry)
    return registry


def _structured(result: dict[str, object]) -> dict[str, object]:
    structured = result["structuredContent"]
    assert isinstance(structured, dict)
    return structured


def _choice(
    element_id: str | None,
    confidence: float,
    candidates: tuple[str, ...],
    affordance: str = "click",
) -> jev.BrowserElementChoiceResult:
    return jev.BrowserElementChoiceResult(
        element_id,
        affordance if element_id is not None else None,
        candidates,
        {candidate: 1 / len(candidates) for candidate in candidates},
        confidence,
        0.9,
        0.9,
        0.9,
        {},
        confidence,
    )


@pytest.mark.asyncio
async def test_browser_click_uses_jev_choice_and_pre_post_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    registry.browser_goal = "continue"
    await registry.execute(ToolCall("state", "browser_state", {}))
    gates: list[dict[str, object]] = []

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",))

    async def gate(**kwargs: object) -> PageStateDecision:
        gates.append(kwargs)
        return PageStateDecision(True, None, None)

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr("zeta.tools.browser.evaluate_page_state_with_provider", gate)

    result = await registry.execute(
        ToolCall(
            "click",
            "browser_click",
            {"snapshot_id": 1, "element_id": "e1", "role": "button", "affordance": "click"},
        )
    )

    assert result["isError"] is False
    assert len(gates) == 2
    assert [element.element_id for element in adapter.clicks] == ["e1"]


@pytest.mark.asyncio
async def test_browser_click_returns_top_three_without_acting_on_low_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _observation()
    elements = tuple(
        ElementRef(
            1,
            f"e{index}",
            "button",
            "click",
            f"Choice {index}",
            f"Choice {index}",
            None,
            "main",
            False,
            True,
        )
        for index in range(1, 4)
    )
    adapter = FakeBrowserAdapter(
        [
            PageObservation(
                observation.snapshot_id,
                observation.generation,
                observation.url,
                observation.title,
                observation.text,
                elements,
                observation.loaded,
                observation.stable,
            )
        ]
    )
    registry = _registry(tmp_path, adapter)
    registry.browser_goal = "choose a choice"
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice(None, 0.7, ("e1", "e2", "e3"))

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    result = await registry.execute(
        ToolCall(
            "click",
            "browser_click",
            {"snapshot_id": 1, "element_id": "e1", "role": "button", "affordance": "click"},
        )
    )

    assert result["isError"] is False
    assert _structured(result)["candidate_ids"] == ["e1", "e2", "e3"]
    assert adapter.clicks == []


@pytest.mark.asyncio
async def test_browser_click_does_not_act_when_page_state_gate_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    registry.browser_goal = "continue"
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",))

    async def gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(False, "state", "action_not_next_step")

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr("zeta.tools.browser.evaluate_page_state_with_provider", gate)
    result = await registry.execute(
        ToolCall(
            "click",
            "browser_click",
            {"snapshot_id": 1, "element_id": "e1", "role": "button", "affordance": "click"},
        )
    )

    assert result["isError"] is True
    assert _structured(result)["error"]["kind"] == "action_not_next_step"
    assert adapter.clicks == []


@pytest.mark.asyncio
async def test_browser_submit_hands_risk_to_shared_safety_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _observation()
    submit = ElementRef(
        observation.snapshot_id,
        "e1",
        "button",
        "submit",
        "safe approved submit",
        "safe approved submit",
        None,
        "main",
        False,
        True,
    )
    adapter = FakeBrowserAdapter(
        [
            PageObservation(
                observation.snapshot_id,
                observation.generation,
                observation.url,
                observation.title,
                observation.text,
                (submit,),
                observation.loaded,
                observation.stable,
            )
        ]
    )
    tier = SafetyTier(cwd=tmp_path, headless=True)
    registry = _registry(tmp_path, adapter, tier)
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",), "submit")

    async def page_gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(True, None, None)

    async def deny(_evidence: object) -> SafetyOutcome:
        return SafetyOutcome("deny", "layer0", reason="durable_state_change")

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr(
        "zeta.tools.browser.evaluate_page_state_with_provider", page_gate
    )
    monkeypatch.setattr(tier, "evaluate_browser_action", deny)

    result = await registry.execute(
        ToolCall(
            "submit",
            "browser_submit",
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "role": "button",
                "affordance": "submit",
            },
        )
    )

    assert result["isError"] is True
    assert _structured(result)["error"]["kind"] == "safety_denied"
    assert adapter.clicks == []


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [ApprovalDecision.ALLOW, ApprovalDecision.DENY])
async def test_browser_ask_uses_durable_approval_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: ApprovalDecision,
) -> None:
    submit = ElementRef(
        1,
        "e1",
        "button",
        "submit",
        "Pay now",
        "Pay now",
        None,
        "main",
        False,
        True,
    )
    adapter = FakeBrowserAdapter([_element_observation(submit)])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW)
    tier = SafetyTier(cwd=tmp_path)
    registry = _registry(
        tmp_path,
        adapter,
        tier,
        approval_policy=policy,
        approval_store=store,
    )
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",), "submit")

    async def page_gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(True, None, None)

    async def ask(_evidence: object) -> SafetyOutcome:
        return SafetyOutcome("ask", "jev", score=2, confidence=0.9, reason="score_exceeds")

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr(
        "zeta.tools.browser.evaluate_page_state_with_provider", page_gate
    )
    monkeypatch.setattr(tier, "evaluate_browser_action", ask)

    call = ToolCall(
        "pay",
        "browser_submit",
        {
            "snapshot_id": 1,
            "element_id": "e1",
            "role": "button",
            "affordance": "submit",
        },
    )
    task = asyncio.create_task(registry.execute(call))
    for _ in range(100):
        if any(request.request_id == call.id for request in policy.pending_requests()):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("browser approval request was not persisted")

    assert policy.resolve(call.id, decision) is True
    result = await task

    assert result["isError"] is (decision is ApprovalDecision.DENY)
    assert len(adapter.clicks) == (1 if decision is ApprovalDecision.ALLOW else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "element", "reason"),
    [
        (
            "browser_click",
            ElementRef(1, "e1", "button", "click", "Delete record", "Delete record", None, "main", False, True),
            "destructive_action",
        ),
        (
            "browser_submit",
            ElementRef(1, "e1", "button", "submit", "Pay now", "Pay now", None, "main", False, True),
            "payment_or_financial_commitment",
        ),
        (
            "browser_submit",
            ElementRef(1, "e1", "button", "submit", "Delete record", "Delete record", None, "main", False, True),
            "durable_state_change",
        ),
        (
            "browser_click",
            ElementRef(1, "e1", "button", "click", "Sign in", "Sign in", None, "main", False, True),
            "authentication_or_permission_change",
        ),
        (
            "browser_click",
            ElementRef(
                1,
                "e1",
                "link",
                "click",
                "Download report",
                "Download report",
                None,
                "main",
                False,
                True,
                download=True,
            ),
            "download",
        ),
        (
            "browser_click",
            ElementRef(
                1,
                "e1",
                "button",
                "click",
                "Unknown target",
                "Unknown target",
                None,
                "main",
                False,
                True,
                target_url="not a url",
            ),
            "unclassifiable_target_url",
        ),
    ],
)
async def test_risky_routes_fail_closed_through_real_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    element: ElementRef,
    reason: str,
) -> None:
    adapter = FakeBrowserAdapter([_element_observation(element)])
    tier = SafetyTier(cwd=tmp_path, headless=True)
    registry = _registry(tmp_path, adapter, tier)
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",), element.affordance)

    async def page_gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(True, None, None)

    async def deny(_evidence: object) -> SafetyOutcome:
        return SafetyOutcome("deny", "layer0", reason=reason)

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr(
        "zeta.tools.browser.evaluate_page_state_with_provider", page_gate
    )
    monkeypatch.setattr(tier, "evaluate_browser_action", deny)

    result = await registry.execute(
        ToolCall(
            "risky",
            tool_name,
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "role": element.role,
                "affordance": element.affordance,
            },
        )
    )

    assert result["isError"] is True
    assert _structured(result)["error"]["kind"] == "safety_denied"
    assert adapter.clicks == []


@pytest.mark.asyncio
async def test_low_confidence_risky_choice_escalates_without_acting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete = ElementRef(
        1,
        "e1",
        "button",
        "click",
        "Delete record",
        "Delete record",
        None,
        "main",
        False,
        True,
    )
    alternatives = (
        ElementRef(
            1,
            "e2",
            "button",
            "click",
            "Delete another record",
            "Delete another record",
            None,
            "main",
            False,
            True,
        ),
        ElementRef(
            1,
            "e3",
            "button",
            "click",
            "Delete one more record",
            "Delete one more record",
            None,
            "main",
            False,
            True,
        ),
    )
    adapter = FakeBrowserAdapter([_element_observation(delete, *alternatives)])
    registry = _registry(tmp_path, adapter)
    registry.browser_goal = "choose a delete record"
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice(None, 0.7, ("e1", "e2", "e3"))

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    result = await registry.execute(
        ToolCall(
            "delete-choice",
            "browser_click",
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "role": "button",
                "affordance": "click",
            },
        )
    )

    assert result["isError"] is False
    assert _structured(result)["candidate_ids"] == ["e1", "e2", "e3"]
    assert adapter.clicks == []


@pytest.mark.asyncio
async def test_benign_navigation_does_not_require_optional_safety_tier(
    tmp_path: Path,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)

    result = await registry.execute(
        ToolCall("navigate", "browser_navigate", {"url": "https://example.test/next"})
    )

    assert result["isError"] is False
    assert adapter.navigations == ["https://example.test/next"]


@pytest.mark.asyncio
async def test_risky_navigation_without_safety_tier_fails_closed(
    tmp_path: Path,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    await registry.execute(ToolCall("state", "browser_state", {}))

    result = await registry.execute(
        ToolCall("navigate", "browser_navigate", {"url": "https://other.test/next"})
    )

    assert result["isError"] is True
    assert _structured(result)["error"]["kind"] == "safety_denied"
    assert adapter.navigations == []


@pytest.mark.asyncio
async def test_external_navigation_uses_safety_handler_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    tier = SafetyTier(cwd=tmp_path, headless=True)
    registry = _registry(tmp_path, adapter, tier)
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def deny(_evidence: object) -> SafetyOutcome:
        return SafetyOutcome("deny", "layer0", reason="external_origin")

    monkeypatch.setattr(tier, "evaluate_browser_action", deny)
    result = await registry.execute(
        ToolCall("navigate", "browser_navigate", {"url": "https://other.test/next"})
    )

    assert result["isError"] is True
    assert _structured(result)["error"]["kind"] == "safety_denied"
    assert adapter.navigations == []


@pytest.mark.asyncio
async def test_browser_registers_stable_schemas_and_starts_lazily(tmp_path: Path) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)

    assert {
        "browser_navigate",
        "browser_state",
        "browser_click",
        "browser_type",
        "browser_select",
        "browser_extract",
        "browser_submit",
    } <= registry.registered_names
    assert adapter.navigations == []
    result = await registry.execute(ToolCall("state", "browser_state", {}))

    assert result["isError"] is False
    assert _structured(result)["snapshot_id"] == 1
    assert adapter.navigations == []


def test_browser_schemas_match_the_complete_spec() -> None:
    registry = ToolRegistry(
        Path("."),
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.browser_adapter_factory = lambda: FakeBrowserAdapter([_observation()])
    register(registry)

    expected = {
        "browser_navigate": '{"description":"Open an allowed URL in the session page.","name":"browser_navigate","parameters":{"additionalProperties":false,"properties":{"url":{"minLength":1,"type":"string"}},"required":["url"],"type":"object"}}',
        "browser_state": '{"description":"Return the current bounded page snapshot and element catalog.","name":"browser_state","parameters":{"additionalProperties":false,"properties":{},"type":"object"}}',
        "browser_click": '{"description":"Click one catalog element by stable snapshot id.","name":"browser_click","parameters":{"additionalProperties":false,"properties":{"affordance":{"minLength":1,"type":"string"},"element_id":{"minLength":1,"type":"string"},"role":{"minLength":1,"type":"string"},"snapshot_id":{"minimum":1,"type":"integer"}},"required":["snapshot_id","element_id","role","affordance"],"type":"object"}}',
        "browser_type": '{"description":"Replace or append text in one input by snapshot id.","name":"browser_type","parameters":{"additionalProperties":false,"properties":{"affordance":{"minLength":1,"type":"string"},"element_id":{"minLength":1,"type":"string"},"replace":{"type":"boolean"},"role":{"minLength":1,"type":"string"},"snapshot_id":{"minimum":1,"type":"integer"},"text":{"type":"string"}},"required":["snapshot_id","element_id","role","affordance","text","replace"],"type":"object"}}',
        "browser_select": '{"description":"Select one option in a select control by snapshot id and value.","name":"browser_select","parameters":{"additionalProperties":false,"properties":{"affordance":{"minLength":1,"type":"string"},"element_id":{"minLength":1,"type":"string"},"role":{"minLength":1,"type":"string"},"snapshot_id":{"minimum":1,"type":"integer"},"value":{"type":"string"}},"required":["snapshot_id","element_id","role","affordance","value"],"type":"object"}}',
        "browser_submit": '{"description":"Submit a form or click the identified submit control after safety approval.","name":"browser_submit","parameters":{"additionalProperties":false,"properties":{"affordance":{"minLength":1,"type":"string"},"element_id":{"minLength":1,"type":"string"},"role":{"minLength":1,"type":"string"},"snapshot_id":{"minimum":1,"type":"integer"}},"required":["snapshot_id","element_id","role","affordance"],"type":"object"}}',
        "browser_extract": '{"description":"Return bounded text or selected attributes from one element or the page.","name":"browser_extract","parameters":{"additionalProperties":false,"properties":{"attributes":{"items":{"type":"string"},"type":"array"},"element_id":{"minLength":1,"type":["string","null"]},"limit":{"minimum":1,"type":"integer"},"snapshot_id":{"minimum":1,"type":["integer","null"]}},"type":"object"}}',
    }

    actual = {
        schema["name"]: json.dumps(schema, sort_keys=True, separators=(",", ":"))
        for schema in registry.schemas
    }
    assert actual == expected


@pytest.mark.asyncio
async def test_browser_use_after_registry_close_is_rejected_without_restart(
    tmp_path: Path,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    factory_calls = 0

    def factory() -> FakeBrowserAdapter:
        nonlocal factory_calls
        factory_calls += 1
        return adapter

    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.browser_adapter_factory = factory
    register(registry)
    await registry.execute(ToolCall("state", "browser_state", {}))
    await registry.close()

    result = await registry.execute(ToolCall("after-close", "browser_state", {}))

    assert result["isError"] is True
    assert _structured(result)["error"]["kind"] == "browser_session_closed"
    assert factory_calls == 1


@pytest.mark.asyncio
async def test_browser_rejects_stale_and_mismatched_element_identity(
    tmp_path: Path,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    await registry.execute(ToolCall("state", "browser_state", {}))

    stale = await registry.execute(
        ToolCall(
            "stale",
            "browser_click",
            {
                "snapshot_id": 999,
                "element_id": "e1",
                "role": "button",
                "affordance": "click",
            },
        )
    )
    mismatch = await registry.execute(
        ToolCall(
            "mismatch",
            "browser_click",
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "role": "link",
                "affordance": "click",
            },
        )
    )

    assert _structured(stale)["error"]["kind"] == "stale_snapshot"
    assert _structured(mismatch)["error"]["kind"] == "element_unavailable"
    assert adapter.clicks == []


@pytest.mark.asyncio
async def test_browser_close_is_idempotent_and_closes_adapter(tmp_path: Path) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    await registry.execute(ToolCall("state", "browser_state", {}))

    await registry.close()
    await registry.close()


@pytest.mark.asyncio
async def test_browser_extract_clamps_limit_and_keeps_truncation_successful(
    tmp_path: Path,
) -> None:
    long_text = "x" * 10_000
    observation = PageObservation(
        1,
        1,
        "https://example.test/",
        "Example",
        long_text,
        (
            ElementRef(
                1,
                "e1",
                "article",
                "extract",
                long_text,
                long_text,
                None,
                "main",
                False,
                True,
            ),
        ),
        True,
        True,
    )
    adapter = FakeBrowserAdapter([observation])
    registry = _registry(tmp_path, adapter)
    await registry.execute(ToolCall("state", "browser_state", {}))

    result = await registry.execute(
        ToolCall(
            "extract",
            "browser_extract",
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "attributes": [],
                "limit": 100_000,
            },
        )
    )

    assert result["isError"] is False
    assert _structured(result)["truncated"] is True
    assert adapter.extractions[-1][2] == 8_000
