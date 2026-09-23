"""Run the explicitly enabled browser smoke through the browser tool handlers."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from zeta.core.safety import SafetyTier
from zeta.protocol.types import ToolCall
from zeta.providers import jev
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.browser import register
from zeta.tools.browser.adapter import PlaywrightBrowserAdapter, SnapshotLimits

__test__ = False

# These values stay unset until Henry approves the live-browser policy.
origin_allowlist: tuple[str, ...] | None = None
per_page_budgets: dict[str, int] | None = None
action_cap: int | None = None


def _enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.live
        and os.environ.get("JEV_BROWSER_SMOKE") == "1"
        and jev._resolve_gateway_key()
        and os.environ.get("JEV_BROWSER_SMOKE_URL")
    )


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("smoke URL must have an http or https origin")
    return f"{parsed.scheme}://{parsed.netloc}".casefold()


def _require_live_config() -> None:
    missing = [
        name
        for name, value in (
            ("origin_allowlist", origin_allowlist),
            ("per_page_budgets", per_page_budgets),
            ("action_cap", action_cap),
        )
        if value is None
    ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"live smoke policy is unset ({names}); awaiting Henry")


class _SmokeBudget:
    def __init__(self) -> None:
        _require_live_config()
        assert origin_allowlist is not None
        assert per_page_budgets is not None
        assert action_cap is not None
        self._origins = set(origin_allowlist)
        self._remaining = dict(per_page_budgets)
        self._actions = 0
        self._action_cap = action_cap

    def check(self, url: str) -> None:
        page_origin = _origin(url)
        if page_origin not in self._origins:
            raise RuntimeError(f"smoke origin is not allowlisted: {page_origin}")
        if self._actions >= self._action_cap:
            raise RuntimeError("smoke action cap reached")
        remaining = self._remaining.get(page_origin, 0)
        if remaining < 1:
            raise RuntimeError(f"smoke page budget exhausted: {page_origin}")
        self._remaining[page_origin] = remaining - 1
        self._actions += 1


def _element_payload(
    state: Mapping[str, object], env_name: str, affordance: str
) -> dict[str, object]:
    entries = state.get("entries")
    if not isinstance(entries, list):
        raise TypeError("browser handler returned no element catalog")
    requested = os.environ.get(env_name)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if requested and entry.get("element_id") != requested:
            continue
        if requested or entry.get("affordance") == affordance:
            return {
                "element_id": entry["element_id"],
                "role": entry["role"],
                "affordance": entry["affordance"],
                "snapshot_id": state["snapshot_id"],
            }
    raise RuntimeError(f"smoke element is not in the current catalog: {env_name}")


def _structured(result: Mapping[str, object]) -> Mapping[str, object]:
    if result.get("isError"):
        raise RuntimeError(str(result))
    structured = result.get("structuredContent")
    if not isinstance(structured, dict):
        raise TypeError("browser handler returned no structured state")
    return structured


async def _call(
    registry: ToolRegistry,
    budget: _SmokeBudget,
    call_name: str,
    tool_name: str,
    arguments: dict[str, object],
    current_url: str,
) -> Mapping[str, object]:
    budget.check(current_url)
    result = await registry.execute(ToolCall(call_name, tool_name, arguments))
    return _structured(result)


async def run_smoke(*, headless: bool) -> None:
    _require_live_config()
    url = os.environ["JEV_BROWSER_SMOKE_URL"]
    budget = _SmokeBudget()
    registry = ToolRegistry(
        Path.cwd(),
        register_builtin=False,
        safety_tier=SafetyTier(cwd=Path.cwd(), headless=True),
        skill_catalog=SkillCatalog.empty(),
    )
    registry.browser_adapter_factory = lambda: PlaywrightBrowserAdapter(
        headless=headless,
        limits=SnapshotLimits(),
    )
    register(registry)
    try:
        state = await _call(
            registry,
            budget,
            "navigate",
            "browser_navigate",
            {"url": url},
            url,
        )
        print(f"navigate/state: {state['url']}")

        type_ref = _element_payload(state, "JEV_BROWSER_SMOKE_TYPE_ID", "type")
        registry.browser_goal = "type the configured smoke text"
        state = await _call(
            registry,
            budget,
            "type",
            "browser_type",
            type_ref
            | {
                "text": os.environ.get("JEV_BROWSER_SMOKE_TYPE_TEXT", "smoke"),
                "replace": True,
            },
            str(state["url"]),
        )

        select_ref = _element_payload(state, "JEV_BROWSER_SMOKE_SELECT_ID", "select")
        registry.browser_goal = "select the configured smoke value"
        state = await _call(
            registry,
            budget,
            "select",
            "browser_select",
            select_ref
            | {"value": os.environ.get("JEV_BROWSER_SMOKE_SELECT_VALUE", "")},
            str(state["url"]),
        )

        submit_ref = _element_payload(state, "JEV_BROWSER_SMOKE_SUBMIT_ID", "submit")
        registry.browser_goal = "submit the smoke form"
        await _call(
            registry,
            budget,
            "submit",
            "browser_submit",
            submit_ref,
            str(state["url"]),
        )
        print("handlers/routing/page-state/safety: passed")
    finally:
        await registry.close()


async def _async_main(args: argparse.Namespace) -> int:
    if not _enabled(args):
        print(
            "browser live smoke skipped; pass --live, set JEV_BROWSER_SMOKE=1, "
            "a Vercel AI Gateway key, and JEV_BROWSER_SMOKE_URL"
        )
        return 0
    await run_smoke(headless=not args.headed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="enable the external-site smoke"
    )
    parser.add_argument(
        "--headed", action="store_true", help="show the browser for debugging"
    )
    return asyncio.run(_async_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
