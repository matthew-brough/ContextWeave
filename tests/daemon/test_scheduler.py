import threading
import time

from contextweave.daemon.scheduler import (
    ActivityGate,
    Job,
    JobKind,
    Priority,
    Scheduler,
    Worker,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make(debounce: float = 1.0, max_delay: float = 5.0) -> tuple[Scheduler, FakeClock]:
    clock = FakeClock()
    return Scheduler(debounce_s=debounce, max_delay_s=max_delay, clock=clock), clock


def test_watch_events_wait_for_debounce() -> None:
    scheduler, clock = make()
    scheduler.enqueue_path("k", "a.py")
    assert scheduler.pop_ready() is None
    clock.now += 1.0
    assert scheduler.pop_ready() == Job("k", JobKind.FILES, ("a.py",))
    assert scheduler.pending() == 0


def test_repeat_events_push_deadline_back_until_max_delay() -> None:
    scheduler, clock = make(debounce=1.0, max_delay=2.5)
    scheduler.enqueue_path("k", "a.py")
    for _ in range(2):
        clock.now += 0.9
        scheduler.enqueue_path("k", "a.py")
        assert scheduler.pop_ready() is None
    clock.now += 0.9
    scheduler.enqueue_path("k", "a.py")
    assert scheduler.pop_ready() is not None


def test_priority_then_most_recently_touched_first() -> None:
    scheduler, clock = make(debounce=0.0)
    scheduler.enqueue_path("a", "old.py", Priority.WATCH)
    clock.now += 1
    scheduler.enqueue_path("b", "new.py", Priority.WATCH)
    clock.now += 1
    scheduler.enqueue_project("c", JobKind.RECONCILE)
    clock.now += 1
    scheduler.enqueue_path("d", "urgent.py", Priority.QUERY)
    order = [scheduler.pop_ready() for _ in range(4)]
    assert [job.key for job in order if job] == ["d", "b", "a", "c"]
    assert scheduler.pop_ready() is None


def test_files_batch_per_project_and_coalesce() -> None:
    scheduler, clock = make(debounce=0.0)
    for name in ("a.py", "b.py", "a.py"):
        scheduler.enqueue_path("k", name)
        clock.now += 1
    scheduler.enqueue_path("other", "z.py")
    job = scheduler.pop_ready()
    assert job is not None and job.key == "other"
    job = scheduler.pop_ready()
    assert job is not None and sorted(job.paths) == ["a.py", "b.py"]


def test_project_jobs_coalesce_and_discard() -> None:
    scheduler, _ = make(debounce=0.0)
    scheduler.enqueue_project("k", JobKind.RECONCILE)
    scheduler.enqueue_project("k", JobKind.RECONCILE)
    scheduler.enqueue_path("k", "a.py")
    assert scheduler.pending("k") == 2
    scheduler.discard_project("k")
    assert scheduler.pending() == 0


def test_worker_runs_jobs_and_reports_errors() -> None:
    scheduler = Scheduler(debounce_s=0.0)
    ran: list[Job] = []
    failed: list[str] = []

    def run(job: Job) -> None:
        if job.paths == ("bad.py",):
            raise RuntimeError("boom")
        ran.append(job)

    worker = Worker(scheduler, run, lambda job, error: failed.append(str(error)))
    worker.start()
    try:
        scheduler.enqueue_path("k", "bad.py")
        assert worker.wait_drained(5)
        scheduler.enqueue_path("k", "good.py")
        assert worker.wait_drained(5)
    finally:
        worker.stop()
    assert failed == ["boom"]
    assert [job.paths for job in ran] == [("good.py",)]


def test_worker_yields_to_running_queries() -> None:
    scheduler = Scheduler(debounce_s=0.0)
    gate = ActivityGate()
    started = threading.Event()
    worker = Worker(scheduler, lambda job: started.set(), lambda job, error: None, gate)
    worker.start()
    try:
        with gate.query():
            scheduler.enqueue_path("k", "a.py")
            time.sleep(0.2)
            assert not started.is_set()
        assert started.wait(5)
    finally:
        worker.stop()
