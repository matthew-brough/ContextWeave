import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import version


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level `cw` argument parser."""
    parser = argparse.ArgumentParser(prog="cw", description="ContextWeave code-context server")
    parser.add_argument("--version", action="version", version=version("contextweave"))
    parser.add_subparsers(dest="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `cw` / `contextweave`.

    argv: Sequence[str] | None — arguments excluding program name; defaults to sys.argv.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
