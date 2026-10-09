import argparse
import re
from collections.abc import Iterator
from datetime import timedelta

from contextweave.cli_commands.common import (
    EXIT_OK,
    EXIT_USAGE,
    CommandContext,
    SubparserRegistry,
    complain,
    say,
)
from contextweave.core.config import load_config
from contextweave.core.paths import derived_db_path
from contextweave.index.derived import DerivedCache
from contextweave.index.embed import annot_inputs
from contextweave.index.projects import ProjectEntry
from contextweave.index.store import Store

DURATION = re.compile(r"(\d+)([hdw]?)")
UNIT_HOURS = {"h": 1, "": 24, "d": 24, "w": 168}


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("cache", help="Derived cache maintenance")
    actions = parser.add_subparsers(dest="cache_command", required=True)
    gc = actions.add_parser("gc", help="Prune derived rows no project references")
    gc.add_argument("--older-than", dest="older_than", metavar="D", help="e.g. 30d, 12h, 2w")


def parse_duration(text: str) -> timedelta | None:
    """Duration like 30d, 12h or 2w; a bare number is days."""
    match = DURATION.fullmatch(text.strip())
    if match is None:
        return None
    return timedelta(hours=int(match.group(1)) * UNIT_HOURS[match.group(2)])


def _project_hashes(entry: ProjectEntry, ctx: CommandContext) -> Iterator[str]:
    if not (entry.root / ".contextweave" / "index.db").is_file():
        return
    config = load_config(entry.root, strict=False, env=ctx.env).config
    with Store.open(entry.root, readonly=True, chunk_config=config.chunk) as store:
        for row in store.query("SELECT DISTINCT content_hash FROM chunks"):
            yield str(row[0])
        for item in annot_inputs(store, config):
            yield item.key


def live_hashes(ctx: CommandContext) -> Iterator[str]:
    """Content hashes referenced by every registered project's index."""
    for entry in ctx.registry().list():
        if not entry.missing:
            yield from _project_hashes(entry, ctx)


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    older_than_text: str | None = args.older_than
    older_than = None
    if older_than_text is not None:
        older_than = parse_duration(older_than_text)
        if older_than is None:
            complain("cache gc", f"bad duration {older_than_text!r}, use e.g. 30d, 12h, 2w")
            return EXIT_USAGE
    cache = DerivedCache(derived_db_path(ctx.env))
    try:
        result = cache.gc(lambda: live_hashes(ctx), older_than)
    finally:
        cache.close()
    say(f"removed {result.annotations} annotations, {result.vectors} vectors")
    return EXIT_OK
