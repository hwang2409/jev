from __future__ import annotations


import asyncio


import gzip


import zlib


from pathlib import Path


import httpx


import pytest


from zeta.core.approval import ApprovalPolicy


from zeta.core.store import ConversationStore


from zeta.skills import SkillCatalog


from zeta.tools import ToolRegistry, websearch


from zeta.tools import fetch as fetch_tool


from zeta.protocol.types import ToolCall, flatten_tool_content


_ASYNC_CLIENT = httpx.AsyncClient


class _ChunkedByteStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes, *, first_chunk_size: int | None = None) -> None:
        self.content = content
        self.first_chunk_size = first_chunk_size

    async def __aiter__(self):
        split_at = self.first_chunk_size or max(1, len(self.content) // 2)
        yield self.content[:split_at]
        yield self.content[split_at:]

    async def aclose(self) -> None:
        return None


def _raw_deflate(value: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(value) + compressor.flush()


def _mock_client(
    monkeypatch: pytest.MonkeyPatch,
    handler,
    *,
    client_kwargs: dict[str, object] | None = None,
):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (fetch_tool.socket.AF_INET, fetch_tool.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )
    def make_client(**kwargs):
        if client_kwargs is not None:
            client_kwargs.update(kwargs)
        return _ASYNC_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(fetch_tool.httpx, "AsyncClient", make_client)


async def _execute_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    *,
    arguments: dict[str, object] | None = None,
    max_output_chars: int = 50_000,
):
    def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    _mock_client(monkeypatch, handler)
    registry = ToolRegistry(tmp_path, max_output_chars=max_output_chars, skill_catalog=SkillCatalog.empty())
    return await registry.execute(
        ToolCall("fetch-1", "fetch", arguments or {"url": "example.com"})
    )


@pytest.mark.asyncio
async def test_real_network_connections_are_blocked() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(AssertionError, match="real network"):
            await client.get("https://example.com")


@pytest.mark.asyncio
@pytest.mark.parametrize("with_stream", [False, True])
async def test_parallel_fetches_cancel_on_registry_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_stream: bool,
) -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()
    started_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal started_count
        del request
        started_count += 1
        if started_count == 2:
            started.set()
        await blocked.wait()
        raise AssertionError("blocked transport was not canceled")

    _mock_client(monkeypatch, handler)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    calls = [
        ToolCall("fetch-a", "fetch", {"url": "example.com/a"}),
        ToolCall("fetch-b", "fetch", {"url": "example.com/b"}),
    ]
    if with_stream:
        async def run_streaming() -> list[object]:
            return await asyncio.gather(
                *(
                    registry.execute(call, _stream_sink=lambda event: None)
                    for call in calls
                )
            )

        task = asyncio.create_task(run_streaming())
    else:
        task = asyncio.create_task(registry.execute_many(calls))

    await asyncio.wait_for(started.wait(), timeout=1)
    registry.abort()
    results = await asyncio.wait_for(task, timeout=1)

    assert [result["content"][0]["text"] for result in results] == [
        "tool execution canceled",
        "tool execution canceled",
    ]


@pytest.mark.asyncio
async def test_discovery_and_approval_gate_network_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def forbidden(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network handler should not run")

    _mock_client(monkeypatch, forbidden)
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(always_deny={"fetch", "websearch"}, store=store)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
skill_catalog=SkillCatalog.empty(),
    )

    assert {"fetch", "websearch"} <= registry.definitions_by_name.keys()
    fetch_result = await registry.execute(
        ToolCall("fetch-1", "fetch", {"url": "example.com"})
    )
    search_result = await registry.execute(
        ToolCall("search-1", "websearch", {"query": "zeta"})
    )

    assert fetch_result["content"][0]["text"] == "tool execution denied"
    assert search_result["content"][0]["text"] == "tool execution denied"
    assert calls == 0
