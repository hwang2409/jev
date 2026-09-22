from __future__ import annotations

import argparse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jmap",
        description="Apply typed Jev questions to finite input states.",
    )
    parser.add_argument("command", nargs="?", help="a v1 command")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is not None:
        parser.error(f"command {args.command!r} is not implemented")
    parser.error("a command is required")
