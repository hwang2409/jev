"""Run the explicitly enabled browser smoke against the local fixture."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlsplit

from zeta.core.safety import SafetyTier, normalize_origin
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.browser import register
from zeta.tools.browser.adapter import (
    BrowserAdapter,
    PlaywrightBrowserAdapter,
    SnapshotLimits,
)

try:
    from browser_fixture.fixture_server import FixtureServer
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from browser_fixture.fixture_server import FixtureServer

__test__ = False

SMOKE_PAGE_JEV_CALL_BUDGET = 8
SMOKE_PAGE_JEV_TOKEN_BUDGET = 12_000
SMOKE_TASK_ACTION_BUDGET = 20
SMOKE_TASK_WALL_CLOCK_SECONDS = 120.0


def _enabled(args: argparse.Namespace) -> bool:
    """Return true only after the explicit smoke gate is enabled."""

    return bool(args.live and os.environ.get("JEV_BROWSER_SMOKE") == "1")


def _provider_configured() -> bool:
    """Require a configured gateway key before live provider calls."""

    provider_env = (
        "VERCEL_" + "AI_GATEWAY",
        "AI_" + "GATEWAY_API_KEY",
        "VERCEL_" + "JEV_KEY",
    )
    return any(os.environ.get(name) for name in provider_env)


def _require_live_config() -> None:
    if not _provider_configured():
        raise RuntimeError(
            "live smoke provider configuration is missing; "
            "set the gateway key"
        )


def _origin(url: str) -> str:
    origin = normalize_origin(url)
    parsed = urlsplit(url)
    if origin is None or parsed.username or parsed.password:
        raise RuntimeError("smoke URL must have a valid http or https origin")
    return origin


class _SmokeBudget:
    """Bound smoke calls before the session's named budgets run."""

    def __init__(self, origin: str) -> None:
        self._origin = origin
        self._actions = 0

    def check(self, url: str) -> None:
        requested_origin = _origin(url)
        if requested_origin != self._origin:
            raise RuntimeError(f"smoke origin is not allowlisted: {requested_origin}")
        if self._actions >= SMOKE_TASK_ACTION_BUDGET:
            raise RuntimeError("smoke action cap reached")
        self._actions += 1


def _element_payload(
    state: Mapping[str, object], affordance: str, *, label: str
) -> dict[str, object]:
    entries = state.get("entries")
    if not isinstance(entries, list):
        raise TypeError("browser handler returned no element catalog")
    wanted = label.casefold()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = " ".join(
            value.casefold()
            for value in (entry.get("text"), entry.get("name"))
            if isinstance(value, str)
        )
        if entry.get("affordance") == affordance and wanted in text:
            return {
                "element_id": entry["element_id"],
                "role": entry["role"],
                "affordance": entry["affordance"],
                "snapshot_id": state["snapshot_id"],
            }
    raise RuntimeError(f"smoke element is not in the current catalog: {label}")


def _structured(
    result: Mapping[str, object], *, allow_error: bool = False
) -> Mapping[str, object]:
    if result.get("isError") and not allow_error:
        raise RuntimeError(str(result.get("structuredContent", result)))
    structured = result.get("structuredContent")
    if not isinstance(structured, dict):
        raise TypeError("browser handler returned no structured result")
    return structured


async def _call(
    registry: ToolRegistry,
    budget: _SmokeBudget,
    call_name: str,
    tool_name: str,
    arguments: dict[str, object],
    current_url: str,
    *,
    allow_error: bool = False,
) -> Mapping[str, object]:
    budget.check(current_url)
    result = await registry.execute(ToolCall(call_name, tool_name, arguments))
    return _structured(result, allow_error=allow_error)


def _error_kind(result: Mapping[str, object]) -> str | None:
    error = result.get("error")
    return error.get("kind") if isinstance(error, dict) else None


def _load_provider() -> object:
    from zeta.providers import jev

    return jev


async def run_smoke(
    *, headless: bool, adapter_factory: Callable[[], BrowserAdapter] | None = None
) -> None:
    _require_live_config()
    _load_provider()
    with FixtureServer() as fixture:
        origin = fixture.origin
        budget = _SmokeBudget(origin)
        registry = ToolRegistry(
            Path.cwd(),
            register_builtin=False,
            browser_enabled=True,
            browser_page_jev_call_budget=SMOKE_PAGE_JEV_CALL_BUDGET,
            browser_page_jev_token_budget=SMOKE_PAGE_JEV_TOKEN_BUDGET,
            browser_task_action_budget=SMOKE_TASK_ACTION_BUDGET,
            browser_task_wall_clock_seconds=SMOKE_TASK_WALL_CLOCK_SECONDS,
            browser_allowed_origins=(origin,),
            browser_safety_tier=SafetyTier(cwd=Path.cwd(), headless=headless),
            skill_catalog=SkillCatalog.empty(),
        )
        adapter = (
            adapter_factory()
            if adapter_factory is not None
            else PlaywrightBrowserAdapter(headless=headless, limits=SnapshotLimits())
        )
        registry.browser_adapter_factory = lambda: adapter
        register(registry)
        try:
            state = await _call(
                registry,
                budget,
                "navigate-home",
                "browser_navigate",
                {"url": fixture.url()},
                fixture.url(),
            )
            print("fixture navigation: passed")

            state = await _call(
                registry,
                budget,
                "navigate-form",
                "browser_navigate",
                {"url": fixture.url("/form")},
                str(state["url"]),
            )
            state = await _call(
                registry,
                budget,
                "form-state",
                "browser_state",
                {},
                str(state["url"]),
            )
            type_ref = _element_payload(state, "type", label="Smoke text")
            registry.browser_goal = "type smoke text in the harmless form"
            state = await _call(
                registry,
                budget,
                "type",
                "browser_type",
                type_ref | {"text": "fixture smoke", "replace": True},
                str(state["url"]),
            )
            submit_ref = _element_payload(state, "submit", label="Submit harmless form")
            registry.browser_goal = "submit the harmless form"
            submitted = await _call(
                registry,
                budget,
                "submit",
                "browser_submit",
                submit_ref,
                str(state["url"]),
            )
            if "/submitted" not in str(submitted["url"]):
                raise RuntimeError("smoke form did not reach the submitted page")
            if "Received text fixture smoke; choice red." not in str(
                submitted["summary"]
            ):
                raise RuntimeError("smoke form did not submit the entered value")
            print("form type/submit: passed")

            state = await _call(
                registry,
                budget,
                "navigate-stale",
                "browser_navigate",
                {"url": fixture.url("/stale")},
                str(submitted["url"]),
            )
            stale_target = _element_payload(state, "click", label="Stale target")
            replace_ref = _element_payload(state, "click", label="Replace stale state")
            registry.browser_goal = "replace the stale fixture state"
            replaced = await _call(
                registry,
                budget,
                "replace-state",
                "browser_click",
                replace_ref,
                str(state["url"]),
            )
            stale = await _call(
                registry,
                budget,
                "stale-target",
                "browser_click",
                stale_target,
                str(replaced["url"]),
                allow_error=True,
            )
            if _error_kind(stale) != "stale_snapshot":
                raise RuntimeError("smoke stale-id recovery returned the wrong result")
            await _call(
                registry,
                budget,
                "recover-state",
                "browser_state",
                {},
                str(replaced["url"]),
            )
            print("stale-id recovery: passed")

            state = await _call(
                registry,
                budget,
                "navigate-low-confidence",
                "browser_navigate",
                {"url": fixture.url("/low-confidence")},
                str(replaced["url"]),
            )
            ambiguous_ref = _element_payload(
                state, "click", label="Continue with local fixture"
            )
            registry.browser_goal = "choose the ambiguous local fixture control"
            low_confidence = await _call(
                registry,
                budget,
                "low-confidence",
                "browser_click",
                ambiguous_ref,
                str(state["url"]),
            )
            candidates = low_confidence.get("candidates")
            candidate_ids = low_confidence.get("candidate_ids")
            if not isinstance(candidates, list) or not isinstance(candidate_ids, list):
                raise TypeError("smoke low-confidence response omitted candidates")
            candidate_labels = [
                candidate.get("text")
                for candidate in candidates
                if isinstance(candidate, dict)
            ]
            candidate_element_ids = [
                candidate.get("element_id")
                for candidate in candidates
                if isinstance(candidate, dict)
            ]
            if (
                not low_confidence.get("requires_choice")
                or candidate_labels != ["Continue with local fixture"] * 3
                or candidate_ids != candidate_element_ids
                or len(candidate_ids) != 3
                or len(set(candidate_ids)) != 3
            ):
                raise RuntimeError(
                    "smoke low-confidence response did not expose the three "
                    "ambiguous controls"
                )
            print("low-confidence top-three: passed")

            state = await _call(
                registry,
                budget,
                "navigate-external-link",
                "browser_navigate",
                {"url": fixture.url()},
                str(state["url"]),
            )
            external_ref = _element_payload(
                state, "click", label="safe/approved external target"
            )
            registry.browser_goal = "open the external target"
            external = await _call(
                registry,
                budget,
                "external-denial",
                "browser_click",
                external_ref,
                str(state["url"]),
                allow_error=True,
            )
            if _error_kind(external) != "safety_denied":
                raise RuntimeError("smoke external navigation was not denied")
            print("external navigation denial: passed")
        finally:
            await registry.close()
            if not getattr(adapter, "_closed", False):
                raise RuntimeError("browser adapter cleanup did not close the adapter")
            print("browser context and registry cleanup: passed")
    print("fixture server cleanup: passed")


async def _async_main(args: argparse.Namespace) -> int:
    if not _enabled(args):
        print(
            "browser live smoke skipped; pass --live and set JEV_BROWSER_SMOKE=1"
        )
        return 0
    await run_smoke(headless=not args.headed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="enable the local fixture smoke"
    )
    parser.add_argument(
        "--headed", action="store_true", help="show the browser for manual debugging"
    )
    return asyncio.run(_async_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
