from __future__ import annotations

from pathlib import Path

import pytest

from zeta.cli.main import build_parser
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.session import SessionManager
from zeta.protocol.types import TextContent
from zeta.providers.jev import AutoRouteResult
from zeta.runtime.headless import run_headless
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog


def test_fake_print_mode_skips_default_auto_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(
        ["--provider", "fake", "--no-session", "-p", "hello"]
    )

    import zeta.runtime.loop as loop_module

    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def capture_auto_route(
        *route_args: object, **route_kwargs: object
    ) -> AutoRouteResult:
        captured.append((route_args, route_kwargs))
        return AutoRouteResult("read", {"read": 1.0}, 1.0, 1.0, {})

    monkeypatch.setattr(loop_module, "auto_route", capture_auto_route)

    assert run_headless(args, args.prompt) == 0
    assert captured == []


@pytest.mark.parametrize(
    ("provider", "expected_auto_route"),
    [("fake", False), ("claude", True)],
)
def test_resumed_headless_router_default_uses_session_provider(
    tmp_path: Path,
    monkeypatch,
    provider: str,
    expected_auto_route: bool,
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    session = SessionManager(tmp_path / "home").create(
        provider=provider,
        model="test-model",
        cwd=tmp_path,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
    )

    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    monkeypatch.setattr(
        "zeta.runtime.bootstrap.build_backend",
        lambda *_args, **_kwargs: (backend, "test-model"),
    )

    import zeta.runtime.loop as loop_module

    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def capture_auto_route(
        *route_args: object, **route_kwargs: object
    ) -> AutoRouteResult:
        captured.append((route_args, route_kwargs))
        return AutoRouteResult("read", {"read": 1.0}, 1.0, 1.0, {})

    monkeypatch.setattr(loop_module, "auto_route", capture_auto_route)
    args = build_parser().parse_args(
        ["--resume", session.metadata.session_id, "-p", "hello"]
    )

    assert run_headless(args, args.prompt) == 0
    assert bool(captured) is expected_auto_route
