import argparse
import json

from contextweave.cli_commands.common import (
    EXIT_FAILED,
    EXIT_OK,
    CommandContext,
    SubparserRegistry,
    complain,
    say,
)
from contextweave.core.config import ConfigError, flatten_config, load_config
from contextweave.index.root import discover_root


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("config", help="Inspect configuration")
    actions = parser.add_subparsers(dest="config_command", required=True)
    show = actions.add_parser("show", help="Print the effective configuration")
    show.add_argument("--origin", action="store_true", help="Show which layer set each key")


def format_value(value: object) -> str:
    """JSON rendering of one config value."""
    return json.dumps(value)


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    origin: bool = args.origin
    try:
        root = discover_root(ctx.cwd)
    except OSError, RuntimeError:
        root = None
    try:
        loaded = load_config(root, strict=False, env=ctx.env)
    except ConfigError as exc:
        complain("config", str(exc))
        return EXIT_FAILED
    for issue in loaded.issues:
        complain("config", f"{issue.message}: key '{issue.key}' in {issue.file}")
    for key, value in sorted(flatten_config(loaded.config).items()):
        suffix = f"  # {loaded.origins[key]}" if origin and key in loaded.origins else ""
        say(f"{key} = {format_value(value)}{suffix}")
    return EXIT_OK
