import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from contextweave.daemon.scheduler import JobKind, Priority, Scheduler
from contextweave.daemon.watcher import FileEvent, PollWatcher

type WatcherFactory = Callable[[Path, str, Callable[[Sequence[FileEvent]], None]], PollWatcher]


@dataclass(slots=True)
class OpenProject:
    """A project the daemon is serving: root, live watcher and last request time."""

    key: str
    root: Path
    watcher: PollWatcher
    opened_at: float
    last_request: float


class ProjectManager:
    """Opens projects on first request, watches them, and closes idle ones."""

    def __init__(
        self,
        scheduler: Scheduler,
        watcher_factory: WatcherFactory,
        idle_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._scheduler = scheduler
        self._watcher_factory = watcher_factory
        self._idle_s = idle_s
        self._clock = clock
        self._open: dict[str, OpenProject] = {}
        self._lock = threading.Lock()

    def open(self, key: str, root: Path) -> OpenProject:
        """Idempotent: start the watcher and queue a reconcile on first open, else touch."""
        with self._lock:
            existing = self._open.get(key)
            now = self._clock()
            if existing is not None:
                existing.last_request = now
                return existing

            def on_events(events: Sequence[FileEvent]) -> None:
                for event in events:
                    self._scheduler.enqueue_path(key, event.path, Priority.WATCH)

            watcher = self._watcher_factory(root, key, on_events)
            watcher.start()
            project = OpenProject(key, root, watcher, now, now)
            self._open[key] = project
        self._scheduler.enqueue_project(key, JobKind.RECONCILE)
        return project

    def close(self, key: str) -> bool:
        """Stop the watcher and drop queued work; False when the project was not open."""
        with self._lock:
            project = self._open.pop(key, None)
        if project is None:
            return False
        project.watcher.stop()
        self._scheduler.discard_project(key)
        return True

    def reap_idle(self) -> list[str]:
        """Close projects without a request for project_idle_minutes."""
        now = self._clock()
        with self._lock:
            stale = [
                key
                for key, project in self._open.items()
                if now - project.last_request >= self._idle_s
            ]
        return [key for key in stale if self.close(key)]

    def close_all(self) -> None:
        for key in self.keys():
            self.close(key)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._open)

    def get(self, key: str) -> OpenProject | None:
        with self._lock:
            return self._open.get(key)

    def snapshot(self) -> list[OpenProject]:
        with self._lock:
            return [self._open[key] for key in sorted(self._open)]
