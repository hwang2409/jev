from __future__ import annotations

import argparse

from tools import browser_live_smoke


def test_browser_smoke_enablement_does_not_preflight_gateway_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JEV_BROWSER_SMOKE", "1")
    monkeypatch.setenv("JEV_BROWSER_SMOKE_URL", "https://example.test")

    assert browser_live_smoke._enabled(argparse.Namespace(live=True)) is True
