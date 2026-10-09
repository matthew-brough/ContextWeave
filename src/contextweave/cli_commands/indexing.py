import time
from collections.abc import Callable, Mapping
from pathlib import Path

from contextweave.cli_commands.common import open_models, say
from contextweave.core.config import Config
from contextweave.index.annotate import AnnotateProgress, format_eta
from contextweave.index.indexer import IndexReport, index_project

PROGRESS_INTERVAL_S = 5.0


def progress_printer(
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[AnnotateProgress], None]:
    """Print annotate progress with ETA at most every few seconds, and at completion."""
    last = clock()

    def show(progress: AnnotateProgress) -> None:
        nonlocal last
        now = clock()
        if progress.done < progress.total and now - last < PROGRESS_INTERVAL_S:
            return
        last = now
        eta = f", eta {format_eta(progress.eta_s)}" if progress.eta_s is not None else ""
        say(f"annotating {progress.done}/{progress.total}{eta}")

    return show


def full_index(root: Path, config: Config, env: Mapping[str, str]) -> IndexReport:
    """Reconcile the tree, then annotate and embed what changed; without models only structure."""
    models = open_models(config, env)
    if models is None:
        return index_project(root, config, env=env)
    try:
        return index_project(
            root,
            config,
            slots=models.slots,
            cache=models.cache,
            env=env,
            on_annotate_progress=progress_printer(),
        )
    finally:
        models.close()
