import argparse

from contextweave.cli_commands.common import (
    EXIT_FAILED,
    EXIT_OK,
    CommandContext,
    SubparserRegistry,
    complain,
    current_project,
    say,
)
from contextweave.memory.recall import recall
from contextweave.memory.store import MemoryStore, memory_db_path
from contextweave.memory.wire import RecallHit

DEFAULT_LIMIT = 10


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("recall", help="Query project memory")
    parser.add_argument("topic")
    parser.add_argument("--scope", choices=("project", "global"), default="project")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)


def hit_line(hit: RecallHit) -> str:
    """One line per memory hit."""
    text = " ".join(hit["text"].split())
    reason = hit.get("reason")
    suffix = f" (because {' '.join(reason.split())})" if reason else ""
    return f"[{hit['kind']} {hit['project']}#{hit['id']} {hit['score']:.2f}] {text}{suffix}"


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    topic: str = args.topic
    scope: str = args.scope
    limit: int = args.limit
    project = current_project("recall", ctx)
    if project is None:
        return EXIT_FAILED
    store = MemoryStore.open(memory_db_path(project.key, ctx.env), project.key)
    try:
        result = recall(store, topic, scope=scope, env=ctx.env, limit=limit)
    finally:
        store.close()
    for error in result["errors"]:
        complain("recall", error.get("message", error["code"]))
    for hit in result["hits"]:
        say(hit_line(hit))
    if not result["hits"] and not result["errors"]:
        say("no matches")
    return EXIT_FAILED if result["errors"] else EXIT_OK
