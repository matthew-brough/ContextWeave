import argparse
from collections.abc import Mapping
from pathlib import Path

from contextweave.cli_commands.common import (
    EXIT_FAILED,
    EXIT_OK,
    CommandContext,
    SubparserRegistry,
    complain,
    current_project,
    open_models,
    say,
)
from contextweave.cli_commands.indexing import full_index, progress_printer
from contextweave.core.config import Config, load_config
from contextweave.daemon.client import DaemonUnavailable, RemoteError
from contextweave.daemon.lifecycle import client_for, is_running
from contextweave.daemon.models import DeriveReport, ModelRuntime
from contextweave.daemon.rebuild import build_shadow, swap_shadow
from contextweave.index.annotate import AnnotateReport, annotate_lines, annotate_project
from contextweave.index.embed import EmbedReport, embed_project
from contextweave.index.indexer import IndexReport, index_project
from contextweave.index.store import Store

STAGES = ("structure", "annotate", "embed")
REBUILD_WAIT_S = 3600.0


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("index", help="Build or refresh the index")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from scratch, then swap")
    parser.add_argument("--stage", choices=STAGES, default=None, help="Run one stage only")


def _embed_lines(report: EmbedReport) -> list[str]:
    lines = [f"embedded {report.embedded} chunks, {report.cached} cached, {report.pending} pending"]
    lines.extend(f"degraded {entry['stage']}: {entry['reason']}" for entry in report.degraded)
    return lines


def _report_lines(report: IndexReport) -> list[str]:
    lines = [
        f"{len(report.added)} added, {len(report.updated)} updated, "
        f"{len(report.removed)} removed, {report.unchanged} unchanged, "
        f"generation {report.generation}"
    ]
    if report.annotate is not None:
        lines.extend(annotate_lines(report.annotate))
    if report.embed is not None:
        lines.extend(_embed_lines(report.embed))
    return lines


def _embed_only(root: Path, config: Config, models: ModelRuntime) -> EmbedReport:
    with Store.open(root, chunk_config=config.chunk, tokenizer=models.tokenizer) as store:
        return embed_project(store, models.slots, models.cache, config, tokenizer=models.tokenizer)


def _annotate_only(root: Path, config: Config, models: ModelRuntime) -> AnnotateReport:
    with Store.open(root, chunk_config=config.chunk, tokenizer=models.tokenizer) as store:
        return annotate_project(
            store, models.slots, models.cache, config, on_progress=progress_printer()
        )


def _rebuild_via_daemon(env: Mapping[str, str], key: str) -> bool:
    try:
        client_for(env, REBUILD_WAIT_S).call(
            "index.rebuild", {"project": key, "wait": True, "bulk": True}
        )
    except DaemonUnavailable, RemoteError:
        return False
    return True


def _derive_lines(report: DeriveReport) -> list[str]:
    return [*annotate_lines(report.annotate), *_embed_lines(report.embed)]


def _rebuild_local(
    root: Path, config: Config, env: Mapping[str, str], stage: str | None
) -> DeriveReport | None:
    """Build the shadow with each model alone on the GPU in turn, then swap; the derive report."""
    models = open_models(config, env) if stage != "structure" else None
    derived: list[DeriveReport] = []
    try:

        def finish(store: Store) -> None:
            if models is not None:
                derived.append(models.bulk_derive(store, serve_after=False))

        build_shadow(
            root,
            config,
            tokenizer=models.tokenizer if models is not None else None,
            finish=finish if models is not None else None,
        )
        swap_shadow(root)
    finally:
        if models is not None:
            models.close()
    return derived[0] if derived else None


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    rebuild: bool = args.rebuild
    stage: str | None = args.stage
    project = current_project("index", ctx)
    if project is None:
        return EXIT_FAILED
    config = load_config(project.root, strict=False, env=ctx.env).config
    if rebuild:
        if is_running(ctx.env) and _rebuild_via_daemon(ctx.env, project.key):
            say("rebuilt via daemon")
            return EXIT_OK
        derived = _rebuild_local(project.root, config, ctx.env, stage)
        say("rebuilt")
        for line in _derive_lines(derived) if derived is not None else []:
            say(line)
        return EXIT_OK
    if stage == "structure":
        for line in _report_lines(index_project(project.root, config, env=ctx.env)):
            say(line)
        return EXIT_OK
    if stage == "annotate":
        models = open_models(config, ctx.env)
        if models is None:
            complain("index", "annotate stage needs usable models")
            return EXIT_FAILED
        try:
            annotate_report = _annotate_only(project.root, config, models)
        finally:
            models.close()
        for line in annotate_lines(annotate_report):
            say(line)
        return EXIT_OK
    if stage == "embed":
        models = open_models(config, ctx.env)
        if models is None:
            complain("index", "embed stage needs usable models")
            return EXIT_FAILED
        try:
            report = _embed_only(project.root, config, models)
        finally:
            models.close()
        for line in _embed_lines(report):
            say(line)
        return EXIT_OK
    for line in _report_lines(full_index(project.root, config, ctx.env)):
        say(line)
    return EXIT_OK
