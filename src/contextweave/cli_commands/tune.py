import argparse
import dataclasses
import tomllib

from contextweave.cli_commands.common import (
    EXIT_FAILED,
    EXIT_OK,
    CommandContext,
    SubparserRegistry,
    complain,
    current_project,
    say,
)
from contextweave.core.config import ScoreConfig, load_config
from contextweave.feedback.dataset import load_dataset
from contextweave.feedback.fit import MIN_SAMPLES, InsufficientDataError, fit_weights
from contextweave.feedback.localtoml import local_path, merge_score
from contextweave.index.store import Store

PRECISION = 6


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("tune", help="Fit [score] weights from recorded feedback")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES, dest="min_samples")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")


def _changes(old: ScoreConfig, new: ScoreConfig) -> dict[str, float]:
    before = dataclasses.asdict(old)
    return {
        name: round(value, PRECISION)
        for name, value in dataclasses.asdict(new).items()
        if round(value, PRECISION) != round(before[name], PRECISION)
    }


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    min_samples: int = args.min_samples
    dry_run: bool = args.dry_run
    project = current_project("tune", ctx)
    if project is None:
        return EXIT_FAILED
    current = load_config(project.root, strict=False, env=ctx.env).config.score
    try:
        with Store.open(project.root, readonly=True) as store:
            dataset = load_dataset(store)
    except Exception as exc:
        complain("tune", f"cannot read feedback: {exc}")
        return EXIT_FAILED
    try:
        fit = fit_weights(dataset.samples, current, min_samples)
    except InsufficientDataError as exc:
        complain("tune", f"not enough feedback to tune: {exc}")
        return EXIT_FAILED
    changes = _changes(current, fit.weights)
    say(
        f"fit {fit.samples} items ({fit.positives} used), auc {fit.auc_before:.3f} -> {fit.auc_after:.3f}"
    )
    for name, value in changes.items():
        say(f"  score.{name} = {value}")
    if dry_run or not changes:
        return EXIT_OK
    path = local_path(project.root)
    try:
        merge_score(path, changes)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        complain("tune", f"cannot update {path}: {exc}")
        return EXIT_FAILED
    say(f"wrote {path}")
    return EXIT_OK
