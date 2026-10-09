import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum, StrEnum

type Clock = Callable[[], float]

BATCH_LIMIT = 256
IDLE_POLL_S = 0.5


class Priority(IntEnum):
    """Lower runs first: interactive reindex, then watcher events, then reconcile sweeps."""

    QUERY = 0
    WATCH = 1
    RECONCILE = 2


class JobKind(StrEnum):
    FILES = "files"
    RECONCILE = "reconcile"
    REBUILD = "rebuild"


@dataclass(frozen=True, slots=True)
class Job:
    """Work for one project; paths is filled for FILES jobs only."""

    key: str
    kind: JobKind
    paths: tuple[str, ...] = ()


@dataclass(slots=True)
class _Entry:
    priority: Priority
    first_seen: float
    ready_at: float
    touched: float
    seq: int


type _Slot = tuple[str, JobKind, str]


class Scheduler:
    """Priority queue of per-file and per-project work with debounce and batching."""

    def __init__(
        self, debounce_s: float = 0.3, max_delay_s: float = 3.0, clock: Clock = time.monotonic
    ) -> None:
        self._debounce_s = debounce_s
        self._max_delay_s = max_delay_s
        self._clock = clock
        self._entries: dict[_Slot, _Entry] = {}
        self._seq = 0
        self._condition = threading.Condition()
        self._closed = False

    def enqueue_path(self, key: str, path: str, priority: Priority = Priority.WATCH) -> None:
        """Queue a file; repeat events inside the debounce window push its start back."""
        self._put((key, JobKind.FILES, path), priority)

    def enqueue_project(
        self, key: str, kind: JobKind, priority: Priority = Priority.RECONCILE
    ) -> None:
        """Queue a project-level job (reconcile or rebuild); duplicates coalesce."""
        self._put((key, kind, ""), priority)

    def _put(self, slot: _Slot, priority: Priority) -> None:
        now = self._clock()
        delay = self._debounce_s if priority is Priority.WATCH else 0.0
        with self._condition:
            self._seq += 1
            entry = self._entries.get(slot)
            if entry is None:
                self._entries[slot] = _Entry(priority, now, now + delay, now, self._seq)
            else:
                entry.priority = min(entry.priority, priority)
                entry.touched = now
                entry.seq = self._seq
                entry.ready_at = min(now + delay, entry.first_seen + self._max_delay_s)
                if priority is not Priority.WATCH:
                    entry.ready_at = min(entry.ready_at, now)
            self._condition.notify_all()

    def pop_ready(self, now: float | None = None) -> Job | None:
        """Take the best ready job: priority, then most recently touched; files batch per project."""
        moment = self._clock() if now is None else now
        with self._condition:
            ready = [
                (slot, entry) for slot, entry in self._entries.items() if entry.ready_at <= moment
            ]
            if not ready:
                return None
            ordered = sorted(ready, key=lambda item: _order(item[1]))
            (key, kind, _), best = ordered[0]
            if kind is not JobKind.FILES:
                del self._entries[(key, kind, "")]
                return Job(key, kind)
            batch = [
                slot
                for slot, entry in ordered
                if slot[0] == key and slot[1] is JobKind.FILES and entry.priority >= best.priority
            ][:BATCH_LIMIT]
            for slot in batch:
                del self._entries[slot]
            return Job(key, JobKind.FILES, tuple(slot[2] for slot in batch))

    def next_ready_at(self) -> float | None:
        with self._condition:
            return min((entry.ready_at for entry in self._entries.values()), default=None)

    def pending(self, key: str | None = None) -> int:
        with self._condition:
            return sum(1 for slot in self._entries if key is None or slot[0] == key)

    def discard_project(self, key: str) -> None:
        with self._condition:
            for slot in [slot for slot in self._entries if slot[0] == key]:
                del self._entries[slot]

    def wait(self, timeout_s: float) -> None:
        """Block until something is queued, the queue closes or the timeout passes."""
        with self._condition:
            if not self._closed:
                self._condition.wait(timeout_s)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


def _order(entry: _Entry) -> tuple[int, float, int]:
    return (int(entry.priority), -entry.touched, entry.seq)


class ActivityGate:
    """Counts in-flight queries so background work can yield to them."""

    def __init__(self) -> None:
        self._active = 0
        self._condition = threading.Condition()

    @contextmanager
    def query(self) -> Generator[None]:
        with self._condition:
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def wait_idle(self, timeout_s: float) -> bool:
        """Wait until no query is running; False when the timeout passed first."""
        with self._condition:
            return self._condition.wait_for(lambda: self._active == 0, timeout_s)


class Worker:
    """Background thread running scheduler jobs one at a time, pausing while queries run."""

    def __init__(
        self,
        scheduler: Scheduler,
        run: Callable[[Job], None],
        on_error: Callable[[Job, Exception], None],
        gate: ActivityGate | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._run = run
        self._on_error = on_error
        self._gate = gate or ActivityGate()
        self._thread = threading.Thread(target=self._loop, name="cw-scheduler", daemon=True)
        self._idle = threading.Event()
        self._idle.set()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._scheduler.close()
        if self._thread.is_alive():
            self._thread.join(timeout_s)

    def wait_drained(self, timeout_s: float) -> bool:
        """Block until the queue is empty and no job is running."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._scheduler.pending() == 0 and self._idle.is_set():
                return True
            time.sleep(0.01)
        return False

    def _loop(self) -> None:
        scheduler = self._scheduler
        while not scheduler.closed:
            self._gate.wait_idle(IDLE_POLL_S)
            self._idle.clear()
            job = scheduler.pop_ready()
            if job is None:
                self._idle.set()
                due = scheduler.next_ready_at()
                now = time.monotonic()
                scheduler.wait(
                    IDLE_POLL_S if due is None else max(min(due - now, IDLE_POLL_S), 0.0)
                )
                continue
            try:
                self._run(job)
            except Exception as exc:
                self._on_error(job, exc)
            finally:
                self._idle.set()
