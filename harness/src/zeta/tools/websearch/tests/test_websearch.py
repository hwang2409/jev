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


from zeta.types import ToolCall, flatten_tool_content


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
async def test_websearch_parses_saved_duckduckgo_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "duckduckgo.html"
    body = fixture.read_text(encoding="utf-8")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/html/"
        assert request.content == b"q=zeta"
        assert request.headers["accept-language"] == "en-US,en;q=0.9"
        assert request.headers["user-agent"].startswith("Mozilla/5.0")
        return httpx.Response(200, headers={"content-type": "text/html"}, text=body)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("search-1", "websearch", {"query": "zeta", "max_results": 1})
    )

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "results": [
            {
                "title": "First result",
                "url": "https://example.com/one",
                "snippet": "A short description.",
            }
        ]
    }


@pytest.mark.asyncio
async def test_websearch_falls_back_to_lite_for_provider_challenge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge = (
        Path(__file__).parent / "fixtures" / "duckduckgo_challenge.html"
    ).read_text(encoding="utf-8")
    lite = (Path(__file__).parent / "fixtures" / "duckduckgo_lite.html").read_text(
        encoding="utf-8"
    )
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            assert request.headers["host"] == "html.duckduckgo.com"
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text=challenge
            )
        assert request.headers["host"] == "lite.duckduckgo.com"
        assert request.method == "POST"
        assert request.content == b"q=zeta"
        return httpx.Response(200, headers={"content-type": "text/html"}, text=lite)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("search-1", "websearch", {"query": "zeta", "max_results": 1})
    )

    assert result["isError"] is False
    assert result["structuredContent"]["results"][0]["title"] == (
        "Login / Sign up - zeta"
    )
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_websearch_falls_back_to_lite_for_parser_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lite = (Path(__file__).parent / "fixtures" / "duckduckgo_lite.html").read_text(
        encoding="utf-8"
    )
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body>unexpected response</body></html>",
            )
        return httpx.Response(200, headers={"content-type": "text/html"}, text=lite)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("search-1", "websearch", {"query": "zeta", "max_results": 1})
    )

    assert result["isError"] is False
    assert calls == 2


def test_websearch_detects_captured_provider_challenge() -> None:
    body = (Path(__file__).parent / "fixtures" / "duckduckgo_challenge.html").read_text(
        encoding="utf-8"
    )

    with pytest.raises(
        websearch.SearchProviderChallengeError,
        match="search provider served a no-results/challenge page",
    ):
        websearch.parse_search_results(body, max_results=8)


def test_websearch_parses_lite_fixture() -> None:
    body = (Path(__file__).parent / "fixtures" / "duckduckgo_lite.html").read_text(
        encoding="utf-8"
    )

    assert websearch.parse_lite_search_results(body, max_results=1) == [
        {
            "title": "Login / Sign up - zeta",
            "url": "https://zeta-ai.io/en/login",
            "snippet": (
                "The No.1 AI chat! Over 13 hours of weekly use — and it's free. "
                "Not using zeta yet? Everyone else is!"
            ),
        }
    ]


def test_websearch_decodes_wrapped_lite_result_url() -> None:
    body = (
        '<a class="result-link" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fresult">'
        "Wrapped result"
        "</a>"
    )

    assert websearch.parse_lite_search_results(body, max_results=1) == [
        {
            "title": "Wrapped result",
            "url": "https://example.com/result",
            "snippet": "",
        }
    ]


@pytest.mark.asyncio
async def test_websearch_output_keeps_registry_truncation_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        '<a class="result__a" href="https://example.com/one">First result</a>'
        '<div class="result__snippet">A short description.</div>'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=body)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, max_output_chars=64, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("search-1", "websearch", {"query": "zeta"})
    )

    block = result["content"][0]
    assert block["truncated"] is True
    assert block["text"].endswith("\n...[output truncated]")


@pytest.mark.asyncio
async def test_websearch_empty_results_and_parse_failure() -> None:
    empty = (Path(__file__).parent / "fixtures" / "duckduckgo_empty.html").read_text(
        encoding="utf-8"
    )
    assert websearch.parse_search_results(empty, max_results=8) == []
    with pytest.raises(ValueError, match="search backend failed"):
        websearch.parse_search_results(
            "<html><body>No results found</body></html>", max_results=8
        )
    with pytest.raises(ValueError, match="search backend failed"):
        websearch.parse_search_results(
            '<div class="no-results__container result__title"><span class="no-results">'
            '<div class="no-results__message"><h1>temporarily blocked</h1></div></span></div>',
            max_results=8,
        )


@pytest.mark.asyncio
async def test_websearch_parser_failure_error_carries_status_and_body_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "<html><body>the backend was temporarily blocked, please try later</body></html>"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=body,
            request=request,
        )

    _mock_client(monkeypatch, handler)
    with pytest.raises(websearch.SearchParserError) as excinfo:
        await websearch._ddg_search("zeta", max_results=1)

    message = str(excinfo.value)
    assert "search backend failed" in message
    assert "HTTP 200" in message
    assert "temporarily blocked" in message


@pytest.mark.asyncio
async def test_websearch_tool_surface_includes_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "<html><body>captcha challenge please solve</body></html>"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=body,
            request=request,
        )

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("search-1", "websearch", {"query": "zeta"})
    )

    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "search backend failed" in text
    assert "HTTP 200" in text
    assert "captcha challenge" in text
