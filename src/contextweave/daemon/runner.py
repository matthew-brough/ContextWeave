import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from contextweave.core.config import Config
from contextweave.daemon.guard import ProjectGuard
from contextweave.daemon.models import ModelRuntime
from contextweave.daemon.rebuild import RebuildReport, build_shadow, swap_shadow
from contextweave.daemon.reconcile import ReconcilePlan, plan_reconcile
from contextweave.daemon.scheduler import Job, JobKind, Priority, Scheduler
from contextweave.index.edges import resolve_edges
from contextweave.index.store import SchemaMismatchError, Store

type RootLookup = Callable[[str], Path | None]


@dataclass(slots=True)
class Activity:
    """Per-project counters surfaced by `status`."""

    indexed_paths: int = 0
    reconciles: int = 0
    rebuilds: int = 0
    rebuilding: bool = False
    last_error: str | None = None
    last_indexed_at: float | None = None


@dataclass(slots=True)
class IndexRunner:
    """Executes scheduler jobs against project dbs; the daemon is their only writer."""

    config: Config
    scheduler: Scheduler
    roots: RootLookup
    models: ModelRuntime | None = None
    activity: dict[str, Activity] = field(default_factory=dict[str, Activity])
    guards: dict[str, ProjectGuard] = field(default_factory=dict[str, ProjectGuard])
    _threads: dict[str, threading.Thread] = field(default_factory=dict[str, threading.Thread])
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def guard(self, key: str) -> ProjectGuard:
        with self._lock:
            return self.guards.setdefault(key, ProjectGuard())

    def stats(self, key: str) -> Activity:
        with self._lock:
            return self.activity.setdefault(key, Activity())

    def run(self, job: Job) -> None:
        root = self.roots(job.key)
        if root is None:
            return
        match job.kind:
            case JobKind.FILES:
                self.index_paths(job.key, root, job.paths)
            case JobKind.RECONCILE:
                self._reconcile(job.key, root)
            case JobKind.REBUILD:
                self.start_rebuild(job.key, root, bulk=False)

    def note_error(self, key: str, error: Exception) -> None:
        self.stats(key).last_error = f"{type(error).__name__}: {error}"

    def index_paths(self, key: str, root: Path, paths: tuple[str, ...]) -> None:
        """Refresh the given paths (reparse, upsert or drop) and embed the changed chunks."""
        stats = self.stats(key)
        tokenizer = self.models.tokenizer if self.models is not None else None
        with (
            self.guard(key).shared(),
            Store.open(root, chunk_config=self.config.chunk, tokenizer=tokenizer) as store,
            store.writer_lock(),
        ):
            for path in paths:
                store.refresh(path)
            self._resolve_edges(key, store)
            stats.indexed_paths += len(paths)
            stats.last_indexed_at = time.time()
            if self.models is not None:
                try:
                    self.models.derive_changed(store)
                except Exception as exc:
                    self.note_error(key, exc)

    def _resolve_edges(self, key: str, store: Store) -> None:
        try:
            resolve_edges(store, self.config.index.test_globs, events=self.config.edges.events)
        except Exception as exc:
            self.note_error(key, exc)

    def reconcile_plan(self, root: Path) -> ReconcilePlan:
        return plan_reconcile(root, self.config.index.ignore, self.config.index.include)

    def _reconcile(self, key: str, root: Path) -> None:
        try:
            plan = self.reconcile_plan(root)
        except SchemaMismatchError:
            self.start_rebuild(key, root, bulk=False)
            return
        self.stats(key).reconciles += 1
        for path in plan.paths:
            self.scheduler.enqueue_path(key, path, Priority.RECONCILE)

    def start_rebuild(self, key: str, root: Path, *, bulk: bool) -> threading.Thread:
        """Rebuild into the shadow db on its own thread, swap on completion, then reconcile."""
        with self._lock:
            running = self._threads.get(key)
            if running is not None and running.is_alive():
                return running
            thread = threading.Thread(
                target=self._rebuild, args=(key, root, bulk), name=f"cw-rebuild-{key}", daemon=True
            )
            self._threads[key] = thread
        self.stats(key).rebuilding = True
        thread.start()
        return thread

    def _rebuild(self, key: str, root: Path, bulk: bool) -> None:
        stats = self.stats(key)
        try:
            self.rebuild_now(key, root, bulk=bulk)
        except Exception as exc:
            self.note_error(key, exc)
        finally:
            stats.rebuilding = False

    def rebuild_now(self, key: str, root: Path, *, bulk: bool) -> RebuildReport:
        """Build the shadow db, swap it in under the project guard, queue a reconcile."""
        models = self.models

        finish = None
        if models is not None:
            finish = models.bulk_derive if bulk else models.derive_changed
        report = build_shadow(
            root,
            self.config,
            tokenizer=models.tokenizer if models is not None else None,
            finish=finish,
        )
        with self.guard(key).exclusive():
            swap_shadow(root)
        self.stats(key).rebuilds += 1
        self.scheduler.enqueue_project(key, JobKind.RECONCILE)
        return report

    def join_rebuild(self, key: str, timeout_s: float) -> bool:
        with self._lock:
            thread = self._threads.get(key)
        if thread is None:
            return True
        thread.join(timeout_s)
        return not thread.is_alive()

    def snapshot(self) -> Mapping[str, Activity]:
        with self._lock:
            return dict(self.activity)
