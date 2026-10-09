import argparse
import json

from contextweave.cli_commands.common import EXIT_OK, CommandContext, SubparserRegistry, say


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("projects", help="List registered projects")
    parser.add_argument("--json", action="store_true", dest="as_json")


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    as_json: bool = args.as_json
    entries = ctx.registry().list()
    if as_json:
        rows = [
            {
                "key": entry.key,
                "root": str(entry.root),
                "last_used": entry.last_used.isoformat(),
                "missing": entry.missing,
            }
            for entry in entries
        ]
        say(json.dumps(rows, indent=2, sort_keys=True))
        return EXIT_OK
    if not entries:
        say("no registered projects")
    for entry in entries:
        flag = "  [missing]" if entry.missing else ""
        say(f"{entry.key}  {entry.root}  last used {entry.last_used:%Y-%m-%d %H:%M}{flag}")
    return EXIT_OK
