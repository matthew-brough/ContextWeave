import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from contextweave.index.walk import walk_project

type Stamp = tuple[int, int]


class EventKind(StrEnum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class FileEvent:
    kind: EventKind
    path: str


class PollWatcher:
    """Stat-based watcher: each scan walks the tree with the index ignores and diffs stamps."""

    def __init__(
        self,
        root: Path,
        ignore: Sequence[str],
        interval_s: float,
        on_events: Callable[[Sequence[FileEvent]], None],
    ) -> None:
        self._root = root
        self._ignore = tuple(ignore)
        self._interval_s = interval_s
        self._on_events = on_events
        self._known: dict[str, Stamp] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _snapshot(self) -> dict[str, Stamp]:
        return {
            item.path: (item.mtime_ns, item.size) for item in walk_project(self._root, self._ignore)
        }

    def prime(self) -> None:
        """Record the current tree as the baseline without reporting it."""
        self._known = self._snapshot()

    def scan(self) -> list[FileEvent]:
        """Diff the tree against the last scan and advance the baseline."""
        current = self._snapshot()
        events = [
            FileEvent(EventKind.CREATED if path not in self._known else EventKind.MODIFIED, path)
            for path, stamp in current.items()
            if self._known.get(path) != stamp
        ]
        events.extend(
            FileEvent(EventKind.DELETED, path) for path in self._known if path not in current
        )
        self._known = current
        return events

    def start(self) -> None:
        """Prime the baseline, then poll on a daemon thread."""
        self.prime()
        self._thread = threading.Thread(target=self._loop, name="cw-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(self._interval_s + 5.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                events = self.scan()
                if events:
                    self._on_events(events)
            except Exception:
                continue
