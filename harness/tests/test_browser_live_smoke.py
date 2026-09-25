from __future__ import annotations

import argparse
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from browser_fixture import FixtureServer

from tools import browser_live_smoke


def test_browser_smoke_requires_the_explicit_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JEV_BROWSER_SMOKE", "1")

    assert browser_live_smoke._enabled(argparse.Namespace(live=True)) is True
    assert browser_live_smoke._enabled(argparse.Namespace(live=False)) is False


def test_browser_smoke_requires_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "VERCEL_AI_GATEWAY",
        "AI_GATEWAY_API_KEY",
        "VERCEL_JEV_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="provider configuration is missing"):
        browser_live_smoke._require_live_config()


def test_browser_fixture_routes_are_deterministic() -> None:
    with FixtureServer() as fixture:
        assert fixture.origin == browser_live_smoke._origin(fixture.url())
        with urlopen(fixture.url("/"), timeout=2) as response:
            home = response.read().decode("utf-8")
        with urlopen(fixture.url("/form"), timeout=2) as response:
            form = response.read().decode("utf-8")
        with urlopen(fixture.url("/search"), timeout=2) as response:
            search = response.read().decode("utf-8")
        with urlopen(fixture.url("/stale"), timeout=2) as response:
            stale = response.read().decode("utf-8")
        with urlopen(fixture.url("/low-confidence"), timeout=2) as response:
            low_confidence = response.read().decode("utf-8")
        root_url = fixture.url()

    assert "Ignore prior instructions" in home
    assert 'action="/submitted"' in form
    assert "data-search-result" in search
    assert "replaceState" in stale
    assert low_confidence.count("Continue with local fixture") == 3
    with pytest.raises(URLError):
        urlopen(root_url, timeout=2)


def test_browser_smoke_budget_allows_only_the_fixture_origin() -> None:
    budget = browser_live_smoke._SmokeBudget("http://127.0.0.1:43127")

    budget.check("http://127.0.0.1:43127/form")
    with pytest.raises(RuntimeError, match="not allowlisted"):
        budget.check("https://external.example.test/blocked")


def test_browser_smoke_output_has_no_page_or_secret_payload() -> None:
    source = (
        browser_live_smoke.run_smoke.__code__.co_consts
        + browser_live_smoke._async_main.__code__.co_consts
    )
    output_literals = " ".join(value for value in source if isinstance(value, str))

    assert "password" not in output_literals.casefold()
    assert "authorization" not in output_literals.casefold()
    assert "full html" not in output_literals.casefold()
