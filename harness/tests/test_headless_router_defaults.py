from __future__ import annotations

from pathlib import Path

from zeta.cli.main import build_parser
from zeta.runtime.headless import run_headless


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

    async def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fake print mode must not call auto_route")

    monkeypatch.setattr(loop_module, "auto_route", fail_if_called)

    assert run_headless(args, args.prompt) == 0
