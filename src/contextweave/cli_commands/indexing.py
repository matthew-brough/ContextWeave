import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

from contextweave.cli_commands.common import close_models, open_models
from contextweave.core.config import Config
from contextweave.index.annotate import format_eta
from contextweave.index.indexer import IndexReport, index_project
from contextweave.index.progress import Progress

PROGRESS_INTERVAL_S = 5.0
REDRAW_INTERVAL_S = 0.2
LABELS = {
    "scan": "scanning files",
    "edges": "resolving edges",
    "annotate": "annotating",
    "embed code": "embedding code",
    "embed annot": "embedding annotations",
}


def progress_text(progress: Progress) -> str:
    """One status line for a stage update."""
    label = LABELS.get(progress.stage, progress.stage)
    if progress.total <= 1:
        return f"{label} done" if progress.done >= progress.total else f"{label}..."
    percent = progress.done * 100 // progress.total
    eta = f", eta {format_eta(progress.eta_s)}" if progress.eta_s is not None else ""
    return f"{label} {progress.done}/{progress.total} ({percent}%){eta}"


class ProgressPrinter:
    """Stage progress on stderr: redrawn in place on a terminal, else a line every few seconds."""

    def __init__(
        self, stream: TextIO | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock
        self.redraw = self.stream.isatty()
        self.stage: str | None = None
        self.finished = False
        self.last = 0.0
        self.open_line = False

    def __call__(self, progress: Progress) -> None:
        if progress.total == 0:
            return
        now = self.clock()
        fresh = progress.stage != self.stage
        finished = progress.done >= progress.total
        interval = REDRAW_INTERVAL_S if self.redraw else PROGRESS_INTERVAL_S
        if not fresh and (self.finished or (not finished and now - self.last < interval)):
            return
        if fresh:
            self.close()
        self.stage, self.finished, self.last = progress.stage, finished, now
        self._write(progress_text(progress), finished)

    def _write(self, text: str, finished: bool) -> None:
        if not self.redraw:
            self.stream.write(f"{text}\n")
        else:
            self.stream.write(f"\r\x1b[K{text}{'\n' if finished else ''}")
            self.open_line = not finished
        self.stream.flush()

    def close(self) -> None:
        """End a line left open by an unfinished stage."""
        if self.open_line:
            self.stream.write("\n")
            self.stream.flush()
            self.open_line = False


def full_index(root: Path, config: Config, env: Mapping[str, str]) -> IndexReport:
    """Reconcile the tree, then annotate and embed what changed; without models only structure."""
    printer = ProgressPrinter()
    models = open_models(config, env)
    try:
        if models is None:
            return index_project(root, config, env=env, on_progress=printer)
        return index_project(
            root,
            config,
            slots=models.slots,
            cache=models.cache,
            env=env,
            on_progress=printer,
            bulk=True,
        )
    finally:
        printer.close()
        if models is not None:
            close_models(models)
