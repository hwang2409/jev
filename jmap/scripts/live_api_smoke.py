"""Run a small, manual live-API smoke for the jmap CLI paths."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TextIO

from jmap.api import TypeSafeClient
from jmap.cache import CacheStore
from jmap.cli import main

DIFF_INPUT = "\n".join(
    [
        "diff --git a/src/app.py b/src/app.py",
        "--- a/src/app.py",
        "+++ b/src/app.py",
        "@@ -1,1 +1,1 @@",
        "-return 0",
        "+return 1",
    ]
)


def _run(
    name: str,
    argv: list[str],
    input_text: str,
    client: TypeSafeClient,
    cache_store: CacheStore,
) -> tuple[str, int, int]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        argv,
        judge_fn=client,
        stdin=io.StringIO(input_text),
        stdout=stdout,
        stderr=stderr,
        cache_store=cache_store,
    )
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    coverage = records[-1] if records else {}
    return name, exit_code, coverage.get("coverage_counts", {}).get("judged", 0)


def run_smoke(cache_dir: Path, *, output: TextIO) -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_store = CacheStore(cache_dir)
    client = TypeSafeClient()
    cases = [
        (
            "jgrep",
            ["jgrep", "--query", "launch decision", "--by", "para"],
            "launch decision: go",
        ),
        (
            "jfilter",
            ["jfilter", "failed payment", "--by", "record", "--state-ref", "id"],
            '{"id":"event-1","status":"payment failed"}',
        ),
        (
            "diff-risk-heat",
            ["run", "--preset", "diff-risk-heat", "--by", "hunk"],
            DIFF_INPUT,
        ),
        (
            "gate",
            [
                "gate",
                "--preset",
                "diff-risk-heat",
                "--by",
                "hunk",
                "--policy",
                "any(change_scope.score >= 2)",
            ],
            DIFF_INPUT,
        ),
    ]
    output.write("case\tstatus\texit\tjudged\n")
    try:
        statuses = []
        for name, argv, input_text in cases:
            case_name, exit_code, judged = _run(
                name, argv, input_text, client, cache_store
            )
            status = "pass" if exit_code == 0 else "fail"
            statuses.append(status)
            output.write(f"{case_name}\t{status}\t{exit_code}\t{judged}\n")
    finally:
        client.close()
    return 0 if all(status == "pass" for status in statuses) else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="explicit directory for temporary smoke answers",
    )
    return parser


def main_cli(argv: list[str] | None = None, *, output: TextIO | None = None) -> int:
    output = output or sys.stdout
    if not os.environ.get("JEV_API_KEY"):
        output.write("skipped: JEV_API_KEY is not set; no API request made\n")
        return 0

    args = _parser().parse_args(argv)
    if args.cache_dir is not None:
        return run_smoke(args.cache_dir, output=output)

    with tempfile.TemporaryDirectory(prefix="jmap-live-smoke-") as temporary_dir:
        return run_smoke(Path(temporary_dir), output=output)


if __name__ == "__main__":
    raise SystemExit(main_cli())
