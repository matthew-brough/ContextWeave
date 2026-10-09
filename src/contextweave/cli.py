import argparse
import os
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from contextweave.core.config import load_config
from contextweave.core.paths import user_state_dir
from contextweave.eval.cli_entry import add_eval_parser, options_from_args, run_eval
from contextweave.index.projects import ProjectRegistry
from contextweave.server.context import ToolContext, inproc_opener
from contextweave.server.mcp import run_stdio


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level `cw` argument parser."""
    parser = argparse.ArgumentParser(prog="cw", description="ContextWeave code-context server")
    parser.add_argument("--version", action="version", version=version("contextweave"))
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="Run the MCP server over stdio")
    serve.add_argument("--inproc", action="store_true", help="Serve from this process, no daemon")
    serve.add_argument("--readonly", action="store_true", help="Never write to the index")
    add_eval_parser(commands)
    return parser


def _serve(inproc: bool, readonly: bool) -> int:
    if not inproc:
        print("cw serve: daemon not implemented yet, use --inproc", file=sys.stderr)
        return 2
    config = load_config(None, strict=False).config
    registry = ProjectRegistry(user_state_dir(os.environ))
    run_stdio(
        ToolContext(config, registry, inproc_opener(config, readonly=readonly), cwd=Path.cwd())
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `cw` / `contextweave`.

    argv: Sequence[str] | None — arguments excluding program name; defaults to sys.argv.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "serve":
        return _serve(args.inproc, args.readonly)
    if args.command == "eval":
        return run_eval(options_from_args(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
