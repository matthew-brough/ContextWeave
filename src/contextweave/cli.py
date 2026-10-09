import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from contextweave.core.config import load_config
from contextweave.core.paths import user_state_dir
from contextweave.core.types import JSON
from contextweave.daemon.client import DaemonUnavailable, RemoteError
from contextweave.daemon.lifecycle import (
    DaemonPaths,
    client_for,
    daemon_status,
    ensure_running,
    is_running,
    stop_daemon,
)
from contextweave.daemon.models import ModelRuntime
from contextweave.daemon.server import run_daemon
from contextweave.daemon.shim import run_shim, shim_backend
from contextweave.eval.cli_entry import add_eval_parser, options_from_args, run_eval
from contextweave.index.projects import ProjectRegistry
from contextweave.server.context import ToolContext, inproc_opener
from contextweave.server.mcp import run_stdio

_PARAMS: TypeAdapter[dict[str, JSON]] = TypeAdapter(dict[str, JSON])

EXIT_NOT_RUNNING = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level `cw` argument parser."""
    parser = argparse.ArgumentParser(prog="cw", description="ContextWeave code-context server")
    parser.add_argument("--version", action="version", version=version("contextweave"))
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="Run the MCP server over stdio")
    serve.add_argument("--inproc", action="store_true", help="Serve from this process, no daemon")
    serve.add_argument("--readonly", action="store_true", help="Never write to the index")
    daemon = commands.add_parser("daemon", help="Control the per-user daemon")
    actions = daemon.add_subparsers(dest="daemon_command", required=True)
    actions.add_parser("start", help="Start the daemon in the background if it is not running")
    actions.add_parser("stop", help="Stop the daemon")
    actions.add_parser("status", help="Show daemon status")
    actions.add_parser("run", help="Run the daemon in the foreground")
    call = actions.add_parser("call", help="Send one IPC request and print the result")
    call.add_argument("method", help="IPC method, e.g. ping or status")
    call.add_argument("params", nargs="?", default="{}", help="JSON object of parameters")
    add_eval_parser(commands)
    return parser


def _serve(inproc: bool, readonly: bool, env: Mapping[str, str]) -> int:
    config = load_config(None, strict=False, env=env).config
    registry = ProjectRegistry(user_state_dir(env))
    if inproc:
        attached = readonly or is_running(env)
        context = ToolContext(
            config, registry, inproc_opener(config, readonly=attached), cwd=Path.cwd()
        )
        run_stdio(context)
        return 0
    local = ToolContext(config, registry, inproc_opener(config, readonly=True), cwd=Path.cwd())
    client = client_for(env)
    run_shim(shim_backend(local, client, lambda: ensure_running(env)))
    return 0


def _daemon_run(env: Mapping[str, str]) -> int:
    config = load_config(None, strict=False, env=env).config
    try:
        models = ModelRuntime.create(config, env)
    except Exception as exc:
        print(f"cw daemon: models unavailable: {exc}", file=sys.stderr)
        models = None
    return run_daemon(env, config, models=models)


def _daemon_call(env: Mapping[str, str], method: str, raw_params: str) -> int:
    try:
        params = _PARAMS.validate_json(raw_params)
    except ValidationError:
        print("cw daemon call: params must be a JSON object", file=sys.stderr)
        return EXIT_USAGE
    try:
        result = client_for(env).call(method, params)
    except DaemonUnavailable as exc:
        print(f"cw daemon call: daemon not reachable ({exc})", file=sys.stderr)
        return EXIT_NOT_RUNNING
    except RemoteError as exc:
        print(f"cw daemon call: {exc}", file=sys.stderr)
        return EXIT_NOT_RUNNING
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _daemon(action: str, args: argparse.Namespace, env: Mapping[str, str]) -> int:
    match action:
        case "start":
            if ensure_running(env):
                print(f"daemon running, socket {DaemonPaths.from_env(env).socket}")
                return 0
            print("cw daemon: failed to start, see daemon.log", file=sys.stderr)
            return EXIT_NOT_RUNNING
        case "stop":
            if stop_daemon(env):
                print("daemon stopped")
                return 0
            print("cw daemon: still running", file=sys.stderr)
            return EXIT_NOT_RUNNING
        case "status":
            status = daemon_status(env)
            if status is None:
                print("daemon not running")
                return EXIT_NOT_RUNNING
            print(json.dumps(status, indent=2, sort_keys=True))
            return 0
        case "run":
            return _daemon_run(env)
        case _:
            return _daemon_call(env, args.method, args.params)


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
        return _serve(args.inproc, args.readonly, os.environ)
    if args.command == "daemon":
        return _daemon(args.daemon_command, args, os.environ)
    if args.command == "eval":
        return run_eval(options_from_args(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
