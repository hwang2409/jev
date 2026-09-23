from __future__ import annotations


import json


from itertools import pairwise


from pathlib import Path


import pytest


import zeta.runtime.loop as loop_module


import zeta.tools.memory as memory_tools


import zeta.tools.route as route_module


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.store import ConversationStore


from zeta.runtime.loop import AgentLoop


from zeta.providers.anthropic_payload import build_messages_payload


from zeta.providers.codex_payload import build_responses_payload


from zeta.providers.jev import AutoRouteResult, MemoryRelevanceResult


from zeta.skills import SkillCatalog


from zeta.tools.registry import ToolRegistry


from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


async def collect(events):
    return [event async for event in events]


def _serialized_message_prefix(
    payload: dict[str, object], marker: tuple[int, int]
) -> bytes:
    message_index, block_index = marker
    messages = payload["messages"]
    assert isinstance(messages, list)
    prefix = [dict(message) for message in messages[: message_index + 1]]
    content = prefix[-1]["content"]
    assert isinstance(content, list)
    prefix[-1]["content"] = [
        dict(block) for block in content[: block_index + 1]
    ]
    prefix[-1]["content"][-1].pop("cache_control", None)
    return json.dumps(prefix, sort_keys=True).encode()


def result(
    tool: str,
    *,
    confidence: float = 0.9,
    needs_tool: float = 1.0,
    probabilities: dict[str, float] | None = None,
) -> AutoRouteResult:
    return AutoRouteResult(
        tool,
        probabilities or {tool: confidence},
        confidence,
        needs_tool,
        {},
    )


def async_result(value: AutoRouteResult):
    async def route(*_args, **_kwargs):
        return value

    return route


def memory_result(path: str, heading: list[str], excerpt: str) -> dict[str, object]:
    return {"path": path, "heading": heading, "excerpt": excerpt, "score": 1.0}


def build_loop(
    tmp_path: Path,
    turns: list[ScriptedTurn],
    *,
    names: tuple[str, ...] = ("read", "write", "bash"),
) -> AgentLoop:
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    route_module.register(registry)
    for name in names:
        registry.register(
            name,
            lambda _arguments, *, name=name: name,
            description=f"{name} description",
            parameters={"type": "object"},
            requires_approval=False,
        )
    return AgentLoop(
        FakeBackend(turns),
        store,
        registry=registry,
        approval_policy=ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW),
        router_style="auto",
        skill_catalog=SkillCatalog.empty(),
    )


def memory_config(tmp_path: Path, corpus: Path) -> Path:
    config = tmp_path / "pausanias.toml"
    config.write_text(
        f'database = "{tmp_path / "index.sqlite3"}"\n\n'
        "[[roots]]\n"
        'id = "fixture"\n'
        f'path = "{corpus}"\n'
        'project = "fixture"\n',
        encoding="utf-8",
    )
    return config


@pytest.mark.asyncio
async def test_memory_injection_off_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module, "auto_route", async_result(result("read")))
    turns = [ScriptedTurn(content=[TextContent("done")])]
    default_loop = build_loop(tmp_path / "default", turns)
    explicit_off_loop = build_loop(
        tmp_path / "explicit-off", [ScriptedTurn(content=[TextContent("done")])]
    )
    explicit_off_loop.memory_injection = False

    await collect(default_loop.run_turn("read the file"))
    await collect(explicit_off_loop.run_turn("read the file"))

    def payload_bytes(agent_loop: AgentLoop) -> list[bytes]:
        return [
            json.dumps(
                build_messages_payload(
                    messages,
                    tools,
                    model="test",
                    max_tokens=16_384,
                    thinking_budget=8_192,
                ),
                sort_keys=True,
            ).encode()
            for messages, tools in agent_loop.backend.calls
        ]

    assert payload_bytes(default_loop) == payload_bytes(explicit_off_loop)


@pytest.mark.asyncio
async def test_memory_injection_is_bounded_and_dedupes_tool_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        loop_module,
        "auto_route",
        async_result(
            AutoRouteResult(
                "read",
                {"read": 1.0},
                1.0,
                1.0,
                {},
                memory_relevance={"candidate-0": 0.9, "candidate-1": 0.9},
            )
        ),
    )
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("done")])])
    loop.memory_injection = True
    loop.tool_registry.memory_config = "fixture.toml"

    async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "content": [],
            "isError": False,
            "structuredContent": {
                "items": [
                    memory_result("one.md", ["One"], "a" * 700),
                    memory_result("two.md", ["Two"], "b" * 700),
                    memory_result("three.md", ["Three"], "c" * 100),
                ]
            },
        }

    monkeypatch.setattr(loop_module, "_memory_search", search)
    await collect(loop.run_turn("remember this"))

    injected = [
        block.text
        for message in loop.store.messages()
        for block in message.content
        if isinstance(block, TextContent)
        and block.text.startswith("Recalled reference material")
    ]
    assert len(injected) == 2
    assert any(text.endswith("a" * 600) for text in injected)
    assert any(text.endswith("b" * 600) for text in injected)
    assert all(not text.endswith("c" * 100) for text in injected)
    assert sum(len(text) for text in injected) <= 1500
    assert loop.store.messages()[0].metadata["compaction_droppable"] is True


@pytest.mark.asyncio
async def test_memory_injection_caps_framed_blocks_not_only_excerpts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = build_loop(tmp_path, [])
    loop.tool_registry.memory_config = "fixture.toml"
    loop.store.append_message(Message(MessageRole.USER, [TextContent("remember")]))

    async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "content": [],
            "isError": False,
            "structuredContent": {
                "items": [
                    memory_result("a" * 600, ["Same"], "a" * 600),
                    memory_result("b" * 600, ["Same"], "b" * 600),
                ]
            },
        }

    monkeypatch.setattr(loop_module, "_memory_search", search)
    candidates, retrieval_reason, capped = await loop._retrieve_memory("remember")
    decision = await loop._inject_memory(
        candidates,
        {candidate["id"]: 0.9 for candidate in candidates},
        retrieval_reason=retrieval_reason,
        retrieval_capped=capped,
    )

    injected = [
        block.text
        for message in loop.store.messages()
        for block in message.content
        if isinstance(block, TextContent)
        and block.text.startswith("Recalled reference material")
    ]
    assert len(injected) == 1
    assert sum(len(text) for text in injected) <= 1500
    assert decision["reason"] == "capped"


@pytest.mark.asyncio
async def test_memory_injection_dedupes_identical_content_at_different_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = build_loop(tmp_path, [])
    loop.tool_registry.memory_config = "fixture.toml"
    loop.store.append_message(Message(MessageRole.USER, [TextContent("remember")]))

    async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "content": [],
            "isError": False,
            "structuredContent": {
                "items": [
                    memory_result("one.md", ["Same"], "same  content"),
                    memory_result("two.md", ["Same"], "same\ncontent"),
                ]
            },
        }

    monkeypatch.setattr(loop_module, "_memory_search", search)
    candidates, retrieval_reason, capped = await loop._retrieve_memory("remember")
    decision = await loop._inject_memory(
        candidates,
        {candidate["id"]: 0.9 for candidate in candidates},
        retrieval_reason=retrieval_reason,
        retrieval_capped=capped,
    )

    injected = [
        block.text
        for message in loop.store.messages()
        for block in message.content
        if isinstance(block, TextContent)
        and block.text.startswith("Recalled reference material")
    ]
    assert len(injected) == 1
    assert decision["reason"] is None


@pytest.mark.asyncio
async def test_memory_injection_uses_newest_dated_section(
    tmp_path: Path,
) -> None:
    loop = build_loop(tmp_path, [])
    loop.store.append_message(Message(MessageRole.USER, [TextContent("remember")]))
    candidates = [
        {
            "id": "candidate-0",
            "path": "topic.md",
            "heading": ["Topic", "2026-09-20T10:00:00+00:00"],
            "excerpt": "old version",
            "content_hash": loop_module._memory_content_hash("old version"),
        },
        {
            "id": "candidate-1",
            "path": "topic.md",
            "heading": ["Topic", "2026-09-21T10:00:00+00:00"],
            "excerpt": "new version",
            "content_hash": loop_module._memory_content_hash("new version"),
        },
    ]

    decision = await loop._inject_memory(
        candidates,
        {"candidate-0": 0.9, "candidate-1": 0.9},
    )

    injected = [
        block.text
        for message in loop.store.messages()
        for block in message.content
        if isinstance(block, TextContent)
        and block.text.startswith("Recalled reference material")
    ]
    assert len(injected) == 1
    assert injected[0].endswith("new version")
    assert decision["skipped_candidates"] == [
        {"id": "candidate-0", "reason": "superseded"}
    ]


@pytest.mark.asyncio
async def test_memory_injection_keeps_single_undated_section_eligible(
    tmp_path: Path,
) -> None:
    loop = build_loop(tmp_path, [])
    loop.store.append_message(Message(MessageRole.USER, [TextContent("remember")]))
    candidates = [
        {
            "id": "candidate-0",
            "path": "topic.md",
            "heading": ["Topic"],
            "excerpt": "first fact",
            "content_hash": loop_module._memory_content_hash("first fact"),
        },
        {
            "id": "candidate-1",
            "path": "topic.md",
            "heading": ["Topic"],
            "excerpt": "second fact",
            "content_hash": loop_module._memory_content_hash("second fact"),
        },
    ]

    decision = await loop._inject_memory(
        candidates,
        {"candidate-0": 0.9, "candidate-1": 0.9},
    )

    injected = [
        block.text
        for message in loop.store.messages()
        for block in message.content
        if isinstance(block, TextContent)
        and block.text.startswith("Recalled reference material")
    ]
    assert len(injected) == 2
    assert decision["skipped_candidates"] == []


@pytest.mark.asyncio
async def test_memory_store_suppresses_real_pausanias_topic_for_session(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    config = memory_config(tmp_path, corpus)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    memory_tools.register(registry)
    (corpus / "deployment.md").write_text(
        "# Deployment\n\nUse the seeded deployment policy.\n",
        encoding="utf-8",
    )
    registry.memory_config = str(config)
    exit_code, _stdout, stderr = await memory_tools._run_pausanias(
        registry, ["index"]
    )
    assert exit_code == 0, stderr
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path),
        registry=registry,
        router_mode=False,
        memory_injection=True,
        skill_catalog=SkillCatalog.empty(),
    )
    loop.tool_registry.memory_config = str(config)

    stored = await loop.tool_registry.execute(
        ToolCall(
            "store-1",
            "memory_store",
            {"topic": "Deployment", "content": "Use the new deployment policy."},
        )
    )
    assert stored["isError"] is False
    assert (corpus / "deployment.md").exists()

    assert "Use the new deployment policy." in (
        corpus / "deployment.md"
    ).read_text(encoding="utf-8")
    candidates, retrieval_reason, retrieval_capped = await loop._retrieve_memory(
        "deployment policy"
    )
    decision = await loop._inject_memory(
        candidates,
        None,
        retrieval_reason=retrieval_reason,
        retrieval_capped=retrieval_capped,
    )

    assert candidates == []
    assert decision["reason"] == "actively_modified"
    assert decision["skipped_candidates"]
    assert all(
        item["reason"] == "actively_modified"
        for item in decision["skipped_candidates"]
    )


@pytest.mark.asyncio
async def test_memory_injection_reinjects_changed_prior_memory_search_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("done")])])
    loop.memory_injection = True
    loop.tool_registry.memory_config = "fixture.toml"
    monkeypatch.setattr(
        loop_module,
        "auto_route",
        async_result(
            AutoRouteResult(
                "read", {"read": 1.0}, 1.0, 1.0, {}, memory_relevance={"candidate-0": 0.9}
            )
        ),
    )

    async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "content": [],
            "isError": False,
            "structuredContent": {
                "items": [memory_result("same.md", ["Same"], "new")]
            },
        }

    loop.store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("memory search returned")],
            tool_result=ToolResult(
                "memory-call",
                "memory search returned",
                structured_content={
                    "items": [memory_result("same.md", ["Same"], "old")]
                },
            ),
        )
    )
    monkeypatch.setattr(loop_module, "_memory_search", search)

    candidates, retrieval_reason, capped = await loop._retrieve_memory("remember this")
    decision = await loop._inject_memory(
        candidates,
        {candidate["id"]: 0.9 for candidate in candidates},
        retrieval_reason=retrieval_reason,
        retrieval_capped=capped,
    )

    injected = [
        block.text
        for message in loop.store.messages()
        for block in message.content
        if isinstance(block, TextContent)
        and block.text.startswith("Recalled reference material")
    ]
    assert len(injected) == 1
    assert injected[0].endswith("new")
    assert decision["reason"] is None


@pytest.mark.asyncio
async def test_memory_injection_dedupes_same_content_at_same_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = build_loop(tmp_path, [])
    loop.tool_registry.memory_config = "fixture.toml"
    existing = memory_result("same.md", ["Same"], "stored")
    loop.store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("stored")],
            tool_result=ToolResult(
                "memory-call", "stored", structured_content={"items": [existing]}
            ),
        )
    )

    async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "content": [],
            "isError": False,
            "structuredContent": {"items": [existing]},
        }

    monkeypatch.setattr(loop_module, "_memory_search", search)
    candidates, retrieval_reason, capped = await loop._retrieve_memory("remember this")
    decision = await loop._inject_memory(
        candidates,
        {candidate["id"]: 0.9 for candidate in candidates},
        retrieval_reason=retrieval_reason,
        retrieval_capped=capped,
    )

    assert not any(
        isinstance(block, TextContent)
        and block.text.startswith("Recalled reference material")
        for message in loop.store.messages()
        for block in message.content
    )
    assert decision["reason"] == "deduped"


@pytest.mark.asyncio
async def test_auto_injection_does_not_make_a_second_jev_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def route(*_args: object, **_kwargs: object) -> AutoRouteResult:
        nonlocal calls
        calls += 1
        return AutoRouteResult(
            "read", {"read": 1.0}, 1.0, 1.0, {}, memory_relevance={}
        )

    monkeypatch.setattr(loop_module, "auto_route", route)
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("done")])])
    loop.memory_injection = True
    loop.tool_registry.memory_config = "fixture.toml"

    await collect(loop.run_turn("answer this"))

    assert calls == 1


@pytest.mark.asyncio
async def test_auto_retrieval_passes_candidates_to_existing_jev_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, object]] = []

    async def route(*_args: object, **kwargs: object) -> AutoRouteResult:
        seen.append(kwargs)
        return result("read")

    monkeypatch.setattr(loop_module, "auto_route", route)
    loop = build_loop(tmp_path, [])
    loop.memory_injection = True

    await loop._prepare_auto_route("answer this")
    loop.tool_registry.memory_config = "fixture.toml"
    async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "isError": False,
            "structuredContent": {
                "items": [memory_result("one.md", ["Fact"], "stored")]
            },
        }

    monkeypatch.setattr(loop_module, "_memory_search", search)
    await loop._prepare_auto_route("answer this")

    assert len(seen) == 2
    assert seen[0] == {}
    assert seen[1] == {
        "memory_candidates": [{
            "id": "candidate-0",
            "path": "one.md",
            "heading": ["Fact"],
            "excerpt": "stored",
            "content_hash": loop_module._memory_content_hash("stored"),
        }]
    }


@pytest.mark.asyncio
async def test_empty_retrieval_skips_candidate_jev_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"isError": False, "structuredContent": {"items": []}}

    async def relevance(*_args: object, **_kwargs: object) -> MemoryRelevanceResult:
        raise AssertionError("candidate judging should not run")

    monkeypatch.setattr(loop_module, "_memory_search", search)
    monkeypatch.setattr(loop_module, "memory_relevance", relevance)
    loop = build_loop(tmp_path, [])
    loop.tool_registry.memory_config = "fixture.toml"

    decision, usage = await loop._prepare_user_memory("remember this")

    assert decision["reason"] == "no_candidates"
    assert usage == {}


@pytest.mark.asyncio
async def test_stock_memory_relevance_runs_once_per_user_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relevance_calls = 0
    search_calls = 0

    async def relevance(_query: str, _candidates: list[dict[str, object]]) -> MemoryRelevanceResult:
        nonlocal relevance_calls
        relevance_calls += 1
        return MemoryRelevanceResult({"candidate-0": 0.9}, {"input_tokens": 1})

    async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal search_calls
        search_calls += 1
        return {
            "content": [],
            "isError": False,
            "structuredContent": {
                "items": [memory_result("stock.md", ["Fact"], "stored")]
            },
        }

    monkeypatch.setattr(loop_module, "memory_relevance", relevance)
    monkeypatch.setattr(loop_module, "_memory_search", search)
    loop = build_loop(
        tmp_path,
        [ScriptedTurn(content=[TextContent("first")]), ScriptedTurn(content=[TextContent("second")])],
    )
    loop.router_mode = False
    loop.memory_injection = True
    loop.tool_registry.memory_config = "fixture.toml"

    await collect(loop.run_turn("answer this"))

    assert relevance_calls == 1
    assert search_calls == 1


@pytest.mark.asyncio
async def test_memory_injection_relevance_failure_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def relevance(_query: str, _candidates: list[dict[str, object]]) -> MemoryRelevanceResult:
        raise RuntimeError("jev unavailable")

    monkeypatch.setattr(loop_module, "memory_relevance", relevance)
    async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "isError": False,
            "structuredContent": {
                "items": [memory_result("one.md", ["Fact"], "stored")]
            },
        }

    monkeypatch.setattr(loop_module, "_memory_search", search)
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("done")])])
    loop.router_mode = False
    loop.memory_injection = True
    loop.tool_registry.memory_config = "fixture.toml"

    events = await collect(loop.run_turn("answer this"))

    assert not any(event.type.value == "error" for event in events)
    assert not any(
        isinstance(block, TextContent)
        and block.text.startswith("Recalled reference material")
        for message in loop.store.messages()
        for block in message.content
    )
    usage = next(event.data for event in events if event.type.value == "usage")
    assert usage["memory_injection"]["reason"] == "jev_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("unconfigured", "memory_unconfigured"),
        ("below_threshold", "below_relevance"),
        ("memory_error", "memory_error"),
        ("deduped", "deduped"),
        ("capped", "capped"),
    ],
)
async def test_memory_injection_skip_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected: str,
) -> None:
    loop = build_loop(tmp_path / case, [])
    if case != "unconfigured":
        loop.tool_registry.memory_config = "fixture.toml"
    loop.store.append_message(Message(MessageRole.USER, [TextContent("remember")]))

    candidates = [{"id": "candidate-0", "path": "same.md", "heading": ["Same"], "excerpt": "stored", "content_hash": loop_module._memory_content_hash("stored")}]
    if case == "below_threshold":
        decision = await loop._inject_memory(candidates, {"candidate-0": 0.2})
    else:
        if case == "memory_error":
            search_result: dict[str, object] = {"isError": True}
        elif case == "deduped":
            existing = memory_result("same.md", ["Same"], "stored")
            loop.store.append_message(
                Message(
                    MessageRole.TOOL_RESULT,
                    [TextContent("stored")],
                    tool_result=ToolResult(
                        "memory-call", "stored", structured_content={"items": [existing]}
                    ),
                )
            )
            search_result = {
                "isError": False,
                "structuredContent": {"items": [existing]},
            }
        else:
            items = [
                memory_result(f"{index}.md", ["Fact"], character * 600)
                for index, character in enumerate(("x", "y", "z"))
            ]
            search_result = {
                "isError": False,
                "structuredContent": {"items": items},
            }

        async def search(*_args: object, **_kwargs: object) -> dict[str, object]:
            return search_result

        monkeypatch.setattr(loop_module, "_memory_search", search)
        candidates, retrieval_reason, retrieval_capped = await loop._retrieve_memory(
            "remember"
        )
        decision = await loop._inject_memory(
            candidates,
            {candidate["id"]: 0.9 for candidate in candidates},
            retrieval_reason=retrieval_reason,
            retrieval_capped=retrieval_capped,
        )

    if case == "unconfigured":
        candidates, retrieval_reason, retrieval_capped = await loop._retrieve_memory(
            "remember"
        )
        decision = await loop._inject_memory(
            candidates,
            None,
            retrieval_reason=retrieval_reason,
            retrieval_capped=retrieval_capped,
        )

    assert decision["reason"] == expected
