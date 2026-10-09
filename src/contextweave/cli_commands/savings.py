import argparse
import json

from contextweave.cli_commands.common import (
    EXIT_FAILED,
    EXIT_OK,
    CommandContext,
    SubparserRegistry,
    current_project,
    say,
)
from contextweave.memory.savings import savings_summary
from contextweave.memory.store import MemoryStore, memory_db_path


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("savings", help="Token savings ledger")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--reset", action="store_true", help="Clear this project's ledger")


def reset_savings(store: MemoryStore) -> int:
    """Delete the project's ledger rows; returns how many were removed."""
    with store.conn:
        cursor = store.conn.execute(
            "DELETE FROM savings_log WHERE project_key = ?", (store.project_key,)
        )
    return cursor.rowcount


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    as_json: bool = args.as_json
    reset: bool = args.reset
    project = current_project("savings", ctx)
    if project is None:
        return EXIT_FAILED
    store = MemoryStore.open(memory_db_path(project.key, ctx.env), project.key)
    try:
        if reset:
            say(f"cleared {reset_savings(store)} ledger rows")
            return EXIT_OK
        totals = savings_summary(store)
    finally:
        store.close()
    if as_json:
        say(json.dumps(totals, indent=2, sort_keys=True))
        return EXIT_OK
    for bucket, row in totals.items():
        percent = f"{row['saved'] / row['baseline']:.0%}" if row["baseline"] else "-"
        say(
            f"{bucket:<12} calls {row['calls']:>5}  baseline {row['baseline']:>8}"
            f"  served {row['served']:>8}  saved {row['saved']:>8} ({percent})"
        )
    return EXIT_OK
