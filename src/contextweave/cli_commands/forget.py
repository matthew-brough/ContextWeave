import argparse
from contextlib import suppress

from contextweave.cli_commands.common import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    CommandContext,
    SubparserRegistry,
    complain,
    current_project,
    say,
)
from contextweave.daemon.client import DaemonUnavailable, RemoteError
from contextweave.daemon.lifecycle import client_for
from contextweave.index.projects import ForgetResult


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("forget", help="Remove a project's index and registry entry")
    parser.add_argument("target", nargs="?", help="Project root or key; default is cwd's project")
    parser.add_argument("--missing", action="store_true", help="Forget projects whose root is gone")
    parser.add_argument("--keep-memory", action="store_true", dest="keep_memory")


def _close_in_daemon(ctx: CommandContext, key: str) -> None:
    with suppress(DaemonUnavailable, RemoteError):
        client_for(ctx.env, 1.0).call("project.close", {"project": key})


def _report(result: ForgetResult) -> str:
    memory = "memory kept" if result.kept_memory else "memory removed"
    return f"forgot {result.key} {result.root} ({memory})"


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    target: str | None = args.target
    missing: bool = args.missing
    keep_memory: bool = args.keep_memory
    registry = ctx.registry()
    if missing and target is not None:
        complain("forget", "give a target or --missing, not both")
        return EXIT_USAGE
    if missing:
        stale = [entry.key for entry in registry.list() if entry.missing]
        for key in stale:
            _close_in_daemon(ctx, key)
        results = registry.forget_missing(keep_memory=keep_memory)
        for result in results:
            say(_report(result))
        if not results:
            say("no missing projects")
        return EXIT_OK
    if target is None:
        project = current_project("forget", ctx)
        if project is None:
            return EXIT_FAILED
        target = project.key
    entry = registry.get(target)
    if entry is None:
        complain("forget", f"no registered project {target}")
        return EXIT_FAILED
    _close_in_daemon(ctx, entry.key)
    result = registry.forget(entry.key, keep_memory=keep_memory)
    if result is None:
        complain("forget", f"no registered project {target}")
        return EXIT_FAILED
    say(_report(result))
    return EXIT_OK
