from __future__ import annotations

import argparse
from dataclasses import replace
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest
from browser_fixture import FixtureServer

from tools import browser_live_smoke
from zeta.protocol import jev
from zeta.tools.browser.adapter import (
    ActionObservation,
    ElementRef,
    FakeBrowserAdapter,
    PageObservation,
)


class _FakeSmokeAdapter(FakeBrowserAdapter):
    def __init__(self) -> None:
        super().__init__(_smoke_observations())
        self._closed = False

    async def navigate(self, url: str, timeout_ms: int) -> PageObservation:
        self._rewrite_next(url)
        return await super().navigate(url, timeout_ms)

    async def type_text(
        self,
        element_ref: ElementRef,
        text: str,
        replace_text: bool,
        timeout_ms: int,
    ) -> ActionObservation:
        self._rewrite_next(self._current_observation().url)
        return await super().type_text(element_ref, text, replace_text, timeout_ms)

    async def click(
        self, element_ref: ElementRef, timeout_ms: int
    ) -> ActionObservation:
        current_url = self._current_observation().url
        if element_ref.element_id == "replace":
            next_url = f"{current_url}#replaced"
        elif element_ref.element_id == "submit":
            origin = f"{urlsplit(current_url).scheme}://{urlsplit(current_url).netloc}"
            next_url = f"{origin}/submitted?smoke-text=fixture+smoke&smoke-choice=red"
        else:
            next_url = current_url
        self._rewrite_next(next_url)
        return await super().click(element_ref, timeout_ms)

    async def close(self) -> None:
        self._closed = True
        await super().close()

    def _rewrite_next(self, url: str) -> None:
        index = min(self._observation_index + 1, len(self._observations) - 1)
        self._observations[index] = replace(self._observations[index], url=url)


def _smoke_observations() -> list[PageObservation]:
    home = (_element("external", "link", "click", "safe/approved external target", "safe/approved external target", target_url="https://external.example.test/blocked"),)
    form = (
        _element("text", "textbox", "type", "", "Smoke text"),
        _element("choice", "combobox", "select", "", "Smoke choice"),
        _element("submit", "button", "submit", "Submit harmless form", "Submit harmless form"),
    )
    stale = (
        _element("replace", "button", "click", "Replace stale state", "Replace stale state"),
        _element("stale", "button", "click", "Stale target", "Stale target"),
    )
    replaced = (
        _element("replace", "button", "click", "Replace stale state", "Replace stale state"),
        _element("fresh", "button", "click", "Fresh target", "Fresh target"),
    )
    low_confidence = tuple(
        _element(f"ambiguous-{index}", "button", "click", "Continue with local fixture", "Continue with local fixture")
        for index in range(1, 4)
    )
    pages = (
        ("Jev browser fixture", home),
        ("Jev browser fixture", home),
        ("Harmless form fixture", form),
        ("Harmless form fixture", form),
        ("Form submitted", ()),
        ("Stale state fixture", stale),
        ("Replaced state fixture", replaced),
        ("Low confidence fixture", low_confidence),
        ("Jev browser fixture", home),
    )
    return [
        PageObservation(index, index, "http://fixture.test/", title, _summary(title), elements, True, True)
        for index, (title, elements) in enumerate(pages, start=1)
    ]


def _summary(title: str) -> str:
    if title == "Form submitted":
        return "Received text fixture smoke; choice red."
    return title


def _element(
    element_id: str,
    role: str,
    affordance: str,
    text: str,
    name: str,
    *,
    target_url: str | None = None,
) -> ElementRef:
    return ElementRef(
        snapshot_id=1,
        element_id=element_id,
        role=role,
        affordance=affordance,
        text=text,
        name=name,
        value_hint=None,
        landmark="main",
        disabled=False,
        visible=True,
        target_url=target_url,
        form_action_origin=None,
    )


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
    assert "safe/approved external target" in home
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


def _patch_fake_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def choose(
        _goal: str,
        _action: str,
        _page_state: dict[str, object],
        candidates: list[dict[str, object]],
        _recent_actions: list[str] | None = None,
    ) -> jev.BrowserElementChoiceResult:
        candidate_ids = tuple(
            candidate["element_id"]
            for candidate in candidates
            if isinstance(candidate.get("element_id"), str)
        )
        ambiguous = len(candidates) == 3 and all(
            candidate.get("text") == "Continue with local fixture"
            for candidate in candidates
        )
        selected_id = None if ambiguous else candidate_ids[0]
        selected_affordance = None if ambiguous else candidates[0]["affordance"]
        return jev.BrowserElementChoiceResult(
            element_id=selected_id,
            affordance=selected_affordance,
            candidate_ids=candidate_ids,
            probabilities={element_id: 1.0 for element_id in candidate_ids},
            confidence=0.5 if ambiguous else 0.95,
            goal_element_present=1.0,
            page_loaded_and_stable=1.0,
            action_is_the_next_step=1.0,
            usage={},
            call_confidence=1.0,
        )

    async def judge(
        *_args: object,
        **_kwargs: object,
    ) -> jev.BrowserPageStateResult:
        return jev.BrowserPageStateResult(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, {}, 1.0)

    async def score(
        *_args: object,
        **_kwargs: object,
    ) -> jev.SafetyScoreResult:
        return jev.SafetyScoreResult(0, {}, 1.0, 0.0, 0.0, {}, 1.0)

    monkeypatch.setattr(jev, "choose_browser_element", choose)
    monkeypatch.setattr(jev, "judge_browser_page_state", judge)
    monkeypatch.setattr(jev, "safety_score", score)
    monkeypatch.setattr(browser_live_smoke, "_load_provider", lambda: object())


@pytest.mark.asyncio
async def test_browser_smoke_output_has_no_page_or_secret_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = _FakeSmokeAdapter()
    _patch_fake_smoke(monkeypatch)
    sentinels = {
        "VERCEL_AI_GATEWAY": "sentinel-cookie",
        "AI_GATEWAY_API_KEY": "sentinel-header",
        "VERCEL_JEV_KEY": "sentinel-html",
    }
    for name, value in sentinels.items():
        monkeypatch.setenv(name, value)

    await browser_live_smoke.run_smoke(
        headless=True,
        adapter_factory=lambda: adapter,
    )

    output = capsys.readouterr().out.casefold()
    assert "fixture navigation: passed" in output
    assert "form type/submit: passed" in output
    assert "safe/approved" not in output
    assert "authorization" not in output
    assert "<html" not in output
    for sentinel in sentinels.values():
        assert sentinel not in output


@pytest.mark.asyncio
async def test_browser_smoke_cleanup_closes_the_fake_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeSmokeAdapter()
    _patch_fake_smoke(monkeypatch)

    await browser_live_smoke.run_smoke(
        headless=True,
        adapter_factory=lambda: adapter,
    )

    assert adapter._closed is True
