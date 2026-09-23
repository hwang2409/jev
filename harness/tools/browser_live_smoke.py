"""Run the explicitly enabled browser smoke against a disposable site."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence

from zeta.tools.browser.adapter import (
    ElementRef,
    ElementUnavailableError,
    PlaywrightBrowserAdapter,
    SnapshotLimits,
)


def _enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.live
        and os.environ.get("JEV_BROWSER_SMOKE") == "1"
        and os.environ.get("JEV_API_KEY")
        and os.environ.get("JEV_BROWSER_SMOKE_URL")
    )


def _element(elements: Sequence[ElementRef], env_name: str, affordance: str) -> ElementRef:
    requested = os.environ.get(env_name)
    if requested:
        return next(element for element in elements if element.element_id == requested)
    return next(element for element in elements if element.affordance == affordance)


async def run_smoke(*, headless: bool) -> None:
    url = os.environ["JEV_BROWSER_SMOKE_URL"]
    adapter = PlaywrightBrowserAdapter(headless=headless, limits=SnapshotLimits())
    await adapter.launch()
    try:
        initial = await adapter.navigate(url, 30_000)
        if not initial.loaded or not initial.stable:
            raise RuntimeError("smoke page did not become loaded and stable")
        print(f"navigate/state: {initial.url}")

        stale_ref = next((element for element in initial.elements if element.affordance == "click"), None)
        await adapter.observe(SnapshotLimits())
        if stale_ref is not None:
            try:
                await adapter.click(stale_ref, 10_000)
            except ElementUnavailableError:
                print("stale-id recovery: passed")
            else:
                raise RuntimeError("stale-id recovery did not reject the old snapshot")

        current = await adapter.observe(SnapshotLimits())
        type_ref = _element(current.elements, "JEV_BROWSER_SMOKE_TYPE_ID", "type")
        await adapter.type_text(
            type_ref,
            os.environ.get("JEV_BROWSER_SMOKE_TYPE_TEXT", "smoke"),
            True,
            10_000,
        )
        current = await adapter.observe(SnapshotLimits())
        select_ref = next(
            (element for element in current.elements if element.affordance == "select"),
            None,
        )
        if select_ref is not None:
            await adapter.select(
                select_ref,
                os.environ.get("JEV_BROWSER_SMOKE_SELECT_VALUE", ""),
                10_000,
            )
        current = await adapter.observe(SnapshotLimits())
        submit_ref = _element(current.elements, "JEV_BROWSER_SMOKE_SUBMIT_ID", "submit")
        await adapter.click(submit_ref, 10_000)
        print("type/select/submit: passed")
        print("low-confidence routing and approval: run through the harness smoke owner")
    finally:
        await adapter.close()


async def _async_main(args: argparse.Namespace) -> int:
    if not _enabled(args):
        print(
            "browser live smoke skipped; pass --live, set JEV_BROWSER_SMOKE=1, "
            "JEV_API_KEY, and JEV_BROWSER_SMOKE_URL"
        )
        return 0
    await run_smoke(headless=not args.headed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="enable the external-site smoke")
    parser.add_argument("--headed", action="store_true", help="show the browser for debugging")
    return asyncio.run(_async_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
