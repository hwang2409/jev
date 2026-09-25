from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zeta.core.abort import AbortSignal
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.safety import BrowserRiskEvidence, SafetyOutcome, SafetyTier
from zeta.core.store import ConversationStore
from zeta.protocol import jev
from zeta.protocol.types import ToolCall
from zeta.runtime.execution import ToolExecutionContext
from zeta.tools.browser import PageStateDecision, _navigation_interceptor
from zeta.tools.browser.adapter import (
    FakeBrowserAdapter,
    NavigationBlockedError,
    PageObservation,
    SnapshotLimits,
)
from zeta.tools.browser.tests.test_tools import (
    _choice,
    _element_observation,
    _observation,
    _registry,
    _structured,
)


@pytest.mark.asyncio
async def test_navigation_guard_classifies_every_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    tier = SafetyTier(cwd=tmp_path, headless=True)
    registry = _registry(tmp_path, adapter, tier)
    seen: list[BrowserRiskEvidence] = []

    async def classify(evidence: BrowserRiskEvidence) -> SafetyOutcome:
        seen.append(evidence)
        return SafetyOutcome("deny", "layer0", reason="external_origin")

    monkeypatch.setattr(tier, "evaluate_browser_action", classify)
    interceptor = _navigation_interceptor(
        registry,
        {},
        abort_signal=None,
        execution_context=None,
        fallback_url="https://example.test/",
        operation_token=1,
    )

    with pytest.raises(NavigationBlockedError):
        await interceptor("https://other.test/submit", "https://example.test/")

    assert seen and seen[0].target_url == "https://other.test/submit"


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [False, True], ids=["interactive", "headless"])
@pytest.mark.parametrize("navigation_kind", ["redirect", "javascript"])
async def test_actual_top_level_navigation_uses_shared_safety_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
    navigation_kind: str,
) -> None:
    first = _observation()
    second = PageObservation(
        2,
        2,
        "https://other.test/redirected",
        "Other",
        "Other page",
        (),
        True,
        True,
    )
    adapter = FakeBrowserAdapter([first, second])
    tier = SafetyTier(cwd=tmp_path, headless=headless)
    registry = _registry(tmp_path, adapter, tier)
    await registry.execute(ToolCall("state", "browser_state", {}))
    if navigation_kind == "javascript":
        adapter.queue_navigation("https://other.test/javascript")

    seen: list[BrowserRiskEvidence] = []

    async def classify(evidence: BrowserRiskEvidence) -> SafetyOutcome:
        seen.append(evidence)
        return SafetyOutcome(
            "deny" if headless else "ask",
            "layer0",
            reason="external_origin",
        )

    async def choose(
        *_args: object, **_kwargs: object
    ) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",))

    async def page_gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(True, None, None)

    monkeypatch.setattr(tier, "evaluate_browser_action", classify)
    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr(
        "zeta.tools.browser.evaluate_page_state_with_provider", page_gate
    )

    result = await registry.execute(
        ToolCall(
            "navigate by page action",
            "browser_click",
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "role": "button",
                "affordance": "click",
            },
        )
    )

    assert result["isError"] is True
    structured = _structured(result)
    assert structured["error"]["kind"] == "safety_denied"
    assert "external_origin" in structured["error"]["message"]
    assert seen and seen[0].target_url == (
        "https://other.test/javascript"
        if navigation_kind == "javascript"
        else "https://other.test/redirected"
    )
    assert (await adapter.observe(SnapshotLimits())).url == first.url


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [False, True], ids=["interactive", "headless"])
async def test_fake_redirect_uses_pre_hop_url_for_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
) -> None:
    first = _observation()
    destination = "https://other.test/redirected"
    second = PageObservation(2, 2, destination, "Other", "Other page", (), True, True)
    adapter = FakeBrowserAdapter([first, second])
    tier = SafetyTier(cwd=tmp_path, headless=headless)
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = None if headless else ApprovalPolicy(store=store)
    registry = _registry(
        tmp_path,
        adapter,
        tier,
        approval_policy=policy,
        approval_store=store if policy is not None else None,
    )
    await registry.execute(ToolCall("state", "browser_state", {}))
    seen: list[BrowserRiskEvidence] = []

    async def classify(evidence: BrowserRiskEvidence) -> SafetyOutcome:
        seen.append(evidence)
        return SafetyOutcome("deny" if headless else "ask", "layer0", reason="redirect")

    monkeypatch.setattr(tier, "evaluate_browser_action", classify)
    task = asyncio.create_task(
        registry.execute(ToolCall("redirect", "browser_navigate", {"url": first.url}))
    )
    if headless:
        result = await task
    else:
        request = await _wait_for_approval(policy, "redirect:nav:")
        assert request.tool_call.arguments["source_origin"] == "https://example.test"
        assert request.tool_call.arguments["destination"] == destination
        assert policy.resolve(request.request_id, ApprovalDecision.DENY)
        result = await task

    assert result["isError"] is True
    assert seen and seen[0].current_origin == "https://example.test"
    assert seen[0].target_url == destination
    assert adapter.navigation_classifications[-1] == (destination, first.url)


async def _wait_for_approval(
    policy: ApprovalPolicy, prefix: str
) -> object:
    for _ in range(100):
        request = next(
            (item for item in policy.pending_requests() if item.request_id.startswith(prefix)),
            None,
        )
        if request is not None:
            return request
        await asyncio.sleep(0.01)
    pytest.fail(f"approval request {prefix!r} was not persisted")


def _approval_interceptor(
    registry: object,
    tool_call: ToolCall,
    operation_token: int,
) -> object:
    return _navigation_interceptor(
        registry,  # type: ignore[arg-type]
        {},
        abort_signal=AbortSignal(),
        execution_context=ToolExecutionContext(tool_call, None, None),
        fallback_url="https://source.test/",
        operation_token=operation_token,
    )


@pytest.mark.asyncio
async def test_navigation_approval_identity_does_not_reuse_prior_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store)
    tier = SafetyTier(cwd=tmp_path)
    registry = _registry(
        tmp_path, adapter, tier, approval_policy=policy, approval_store=store
    )
    seen: list[BrowserRiskEvidence] = []

    async def ask(evidence: BrowserRiskEvidence) -> SafetyOutcome:
        seen.append(evidence)
        return SafetyOutcome("ask", "layer0", reason="external_origin")

    monkeypatch.setattr(tier, "evaluate_browser_action", ask)
    call = ToolCall("navigate", "browser_navigate", {})
    interceptor = _approval_interceptor(registry, call, 1)
    first = asyncio.create_task(
        interceptor("https://first.test/", "https://source.test/")
    )
    request = await _wait_for_approval(policy, "navigate:nav:")
    assert policy.resolve(request.request_id, ApprovalDecision.ALLOW)
    await first

    second = asyncio.create_task(
        interceptor("https://second.test/", "https://source.test/")
    )
    new_request = await _wait_for_approval(policy, "navigate:nav:")
    assert new_request.request_id != request.request_id
    assert new_request.tool_call.arguments["destination"] == "https://second.test/"
    assert policy.resolve(new_request.request_id, ApprovalDecision.DENY)
    with pytest.raises(NavigationBlockedError):
        await second
    assert [item.target_url for item in seen] == [
        "https://first.test/",
        "https://second.test/",
    ]


@pytest.mark.asyncio
async def test_navigation_approval_reuses_same_triple_within_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store)
    tier = SafetyTier(cwd=tmp_path)
    registry = _registry(
        tmp_path, adapter, tier, approval_policy=policy, approval_store=store
    )

    async def ask(_evidence: BrowserRiskEvidence) -> SafetyOutcome:
        return SafetyOutcome("ask", "layer0", reason="external_origin")

    monkeypatch.setattr(tier, "evaluate_browser_action", ask)
    call = ToolCall("navigate", "browser_navigate", {})
    interceptor = _approval_interceptor(registry, call, 1)
    first = asyncio.create_task(
        interceptor("https://first.test/", "https://source.test/")
    )
    request = await _wait_for_approval(policy, "navigate:nav:")
    assert policy.resolve(request.request_id, ApprovalDecision.ALLOW)
    await first

    await interceptor("https://first.test/", "https://source.test/")
    assert policy.pending_requests() == []


@pytest.mark.asyncio
async def test_navigation_approval_asks_again_for_new_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store)
    tier = SafetyTier(cwd=tmp_path)
    registry = _registry(
        tmp_path, adapter, tier, approval_policy=policy, approval_store=store
    )

    async def ask(_evidence: BrowserRiskEvidence) -> SafetyOutcome:
        return SafetyOutcome("ask", "layer0", reason="external_origin")

    monkeypatch.setattr(tier, "evaluate_browser_action", ask)
    call = ToolCall("navigate", "browser_navigate", {})
    first_interceptor = _approval_interceptor(registry, call, 1)
    first = asyncio.create_task(
        first_interceptor("https://first.test/", "https://source.test/")
    )
    request = await _wait_for_approval(policy, "navigate:nav:")
    assert policy.resolve(request.request_id, ApprovalDecision.ALLOW)
    await first

    second_interceptor = _approval_interceptor(registry, call, 2)
    second = asyncio.create_task(
        second_interceptor("https://first.test/", "https://source.test/")
    )
    new_request = await _wait_for_approval(policy, "navigate:nav:")
    assert new_request.request_id != request.request_id
    assert policy.resolve(new_request.request_id, ApprovalDecision.DENY)
    with pytest.raises(NavigationBlockedError):
        await second


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["state", "extract"])
async def test_read_only_operations_deny_page_navigation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    registry = _registry(tmp_path, adapter)
    if operation == "extract":
        await registry.execute(ToolCall("state", "browser_state", {}))
        original = adapter.extract

        async def extract(*args: object, **kwargs: object) -> object:
            await adapter.trigger_navigation("https://example.test/changed")
            await adapter.observe(SnapshotLimits())
            return await original(*args, **kwargs)

        monkeypatch.setattr(adapter, "extract", extract)
        arguments = {"attributes": ["text"]}
    else:
        original = adapter.observe

        async def observe(limits: SnapshotLimits) -> PageObservation:
            await adapter.trigger_navigation("https://example.test/changed")
            return await original(limits)

        monkeypatch.setattr(adapter, "observe", observe)
        arguments = {}

    result = await registry.execute(ToolCall(operation, f"browser_{operation}", arguments))

    assert result["isError"] is True
    assert _structured(result)["error"]["kind"] == "safety_denied"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "affordance", "arguments"),
    [
        ("browser_click", "click", {}),
        ("browser_type", "type", {"text": "hello", "replace": True}),
        ("browser_select", "select", {"value": "one"}),
        ("browser_submit", "submit", {}),
    ],
)
async def test_same_origin_action_navigation_runs_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    affordance: str,
    arguments: dict[str, object],
) -> None:
    element = _observation().elements[0]
    element = element.__class__(
        element.snapshot_id,
        element.element_id,
        "textbox" if affordance == "type" else "combobox" if affordance == "select" else element.role,
        affordance,
        element.text,
        element.name,
        "one" if affordance == "select" else element.value_hint,
        element.landmark,
        element.disabled,
        element.visible,
        generation=element.generation,
    )
    first = _element_observation(element)
    second = PageObservation(2, 2, "https://example.test/next", "Next", "Next", (element,), True, True)
    adapter = FakeBrowserAdapter([first, second])
    tier = SafetyTier(cwd=tmp_path)
    registry = _registry(tmp_path, adapter, tier)
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(*_args: object, **_kwargs: object) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",), affordance)

    async def page_gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(True, None, None)

    async def allow(_evidence: BrowserRiskEvidence) -> SafetyOutcome:
        return SafetyOutcome("allow", "layer0")

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr("zeta.tools.browser.evaluate_page_state_with_provider", page_gate)
    monkeypatch.setattr(tier, "evaluate_browser_action", allow)
    result = await registry.execute(
        ToolCall(
            tool_name,
            tool_name,
            {
                "snapshot_id": 1,
                "element_id": "e1",
                "role": element.role,
                "affordance": affordance,
                **arguments,
            },
        )
    )

    assert result["isError"] is False
    assert adapter.navigation_classifications == [
        ("https://example.test/next", "https://example.test/")
    ]


@pytest.mark.asyncio
async def test_same_origin_browser_navigate_runs_classification(
    tmp_path: Path,
) -> None:
    first = _observation()
    destination = "https://example.test/next"
    adapter = FakeBrowserAdapter(
        [first, PageObservation(2, 2, destination, "Next", "Next", (), True, True)]
    )
    registry = _registry(tmp_path, adapter)
    await registry.execute(ToolCall("state", "browser_state", {}))

    result = await registry.execute(
        ToolCall("navigate", "browser_navigate", {"url": destination})
    )

    assert result["isError"] is False
    assert adapter.navigation_classifications == [(destination, first.url)]


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [False, True], ids=["interactive", "headless"])
async def test_delayed_navigation_after_action_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
) -> None:
    adapter = FakeBrowserAdapter([_observation()])
    tier = SafetyTier(cwd=tmp_path, headless=headless)
    registry = _registry(tmp_path, adapter, tier)
    await registry.execute(ToolCall("state", "browser_state", {}))

    async def choose(
        *_args: object, **_kwargs: object
    ) -> jev.BrowserElementChoiceResult:
        return _choice("e1", 0.9, ("e1",))

    async def page_gate(**_kwargs: object) -> PageStateDecision:
        return PageStateDecision(True, None, None)

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr(
        "zeta.tools.browser.evaluate_page_state_with_provider", page_gate
    )

    result = await registry.execute(
        ToolCall(
            "click",
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

    delayed = asyncio.create_task(
        adapter.trigger_navigation("https://other.test/timer")
    )
    await asyncio.sleep(0)
    await delayed

    observation = await registry.execute(ToolCall("state", "browser_state", {}))
    assert observation["isError"] is True
    assert _structured(observation)["error"]["kind"] == "safety_denied"
    assert adapter.navigations == []
