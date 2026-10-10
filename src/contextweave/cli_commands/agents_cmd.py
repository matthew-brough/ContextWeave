import argparse
import sys
from pathlib import Path

from contextweave.agents.catalog import DEFAULT_AGENT, NO_AGENT
from contextweave.agents.cli_entry import CommandResult, run_hook, run_init, run_uninstall
from contextweave.cli_commands.common import CommandContext, SubparserRegistry, complain, say
from contextweave.cli_commands.indexing import full_index
from contextweave.core.config import Config
from contextweave.index.indexer import IndexReport
from contextweave.index.root import discover_root


def add_parsers(commands: SubparserRegistry) -> None:
    init = commands.add_parser("init", help="Index this project and install agent integration")
    init.add_argument(
        "--agent", default=DEFAULT_AGENT, help=f"Agent to install for; '{NO_AGENT}' = index only"
    )
    init.add_argument("--plugin", action="store_true", help="Write a plugin dir instead")
    init.add_argument("--force", action="store_true", help="Install despite parent conflicts")
    init.add_argument("--dry-run", action="store_true", dest="dry_run")
    uninstall = commands.add_parser("uninstall", help="Remove agent integration (keeps the index)")
    uninstall.add_argument("--agent", default=None)
    uninstall.add_argument("--dry-run", action="store_true", dest="dry_run")
    hook = commands.add_parser("hook", help="Agent hook entrypoint, payload JSON on stdin")
    hook.add_argument("event")


def _home(ctx: CommandContext) -> Path | None:
    home = ctx.env.get("HOME")
    return Path(home) if home else None


def _print(result: CommandResult, dry_run: bool) -> int:
    for message in result.messages:
        if result.exit_code == 0:
            say(message)
        else:
            complain("init", message)
    for change in result.changes:
        verb = "delete" if change.after is None else "create" if change.before is None else "update"
        say(f"{'would ' if dry_run else ''}{verb} {change.path}")
    if result.registered:
        say("project registered")
    return result.exit_code


def run_init_command(args: argparse.Namespace, ctx: CommandContext) -> int:
    agent: str = args.agent
    plugin: bool = args.plugin
    force: bool = args.force
    dry_run: bool = args.dry_run

    def indexer(root: Path, config: Config) -> IndexReport:
        return full_index(root, config, ctx.env)

    result = run_init(
        discover_root(ctx.cwd),
        agent,
        plugin,
        force,
        dry_run,
        home=_home(ctx),
        env=ctx.env,
        indexer=indexer,
    )
    return _print(result, dry_run)


def run_uninstall_command(args: argparse.Namespace, ctx: CommandContext) -> int:
    agent: str | None = args.agent
    dry_run: bool = args.dry_run
    return _print(
        run_uninstall(discover_root(ctx.cwd), _home(ctx), dry_run, agent, ctx.env), dry_run
    )


def run_hook_command(args: argparse.Namespace, ctx: CommandContext) -> int:
    event: str = args.event
    output, code = run_hook(event, sys.stdin.buffer.read(), ctx.env)
    if output:
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
    return code
