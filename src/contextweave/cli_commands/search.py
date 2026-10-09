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
from contextweave.core.config import load_config
from contextweave.index.store import Store
from contextweave.retrieve.pack import ContextPack
from contextweave.retrieve.search import SearchOptions, SearchResult, search


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("search", help="Run the retrieval pipeline and print the pack")
    parser.add_argument("query")
    parser.add_argument("--explain", action="store_true", help="Print plan, coverage and scores")
    parser.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")


def explain_lines(pack: ContextPack) -> list[str]:
    """Plan, coverage and per-item scores of a pack."""
    plan = pack["plan"]
    coverage = pack["coverage"]
    lines = [
        f"plan: intent {plan['intent']} planner {plan['planner']} fast_path {plan['fast_path']}"
        f" compaction {plan['compaction']} rels {','.join(plan['rels']) or '-'}",
        f"rrf weights: {plan['rrf_weights']}",
        f"devices: {plan['devices']}",
        f"coverage: score {coverage['score']} sufficient {coverage['sufficient']}"
        f" iterations {coverage['iterations']}",
    ]
    for item in pack["items"]:
        scores = item["scores"]
        lines.append(
            f"item {item['ref']} {item['path']}:{item['start_line']} why={item['why']}"
            f" final={scores['final']:.3f} gate={scores['gate']:.3f}"
            f" lexical={scores['lexical_match']:.3f}"
        )
    return lines


def summary_lines(result: SearchResult) -> list[str]:
    """Budget, degraded entries, errors and per-stage timings."""
    pack = result.pack
    budget = pack["budget"]
    lines = [
        f"served {budget['served_tokens']}/{budget['ceiling']} tokens, "
        f"{len(pack['items'])} items, {len(result.omitted)} omitted",
    ]
    lines.extend(f"degraded {entry['stage']}: {entry['reason']}" for entry in pack["degraded"])
    lines.extend(f"error {entry['code']}: {entry.get('message', '')}" for entry in pack["errors"])
    lines.append("stages:")
    lines.extend(
        f"  {timing.stage:<12} {timing.elapsed_ms:8.2f} ms{' FAILED' if timing.failed else ''}"
        for timing in result.timings
    )
    return lines


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    query: str = args.query
    explain: bool = args.explain
    max_tokens: int | None = args.max_tokens
    project = current_project("search", ctx)
    if project is None:
        return EXIT_FAILED
    config = load_config(project.root, strict=False, env=ctx.env).config
    if not (project.root / ".contextweave" / "index.db").is_file():
        complain("search", "index not built (run cw init)")
        return EXIT_FAILED
    with Store.open(project.root, readonly=True, chunk_config=config.chunk) as store:
        result = search(store, config, query, SearchOptions(max_tokens=max_tokens))
    say(result.text)
    if explain:
        for line in explain_lines(result.pack):
            say(line)
    for line in summary_lines(result):
        say(line)
    return EXIT_OK
