import threading
from concurrent.futures import Future

import pytest

from contextweave.index.workers import worker_pool


def test_interrupt_drops_queued_calls_without_waiting() -> None:
    release = threading.Event()
    started = threading.Event()

    def block() -> None:
        started.set()
        release.wait(5.0)

    futures: list[Future[None]] = []
    with pytest.raises(KeyboardInterrupt), worker_pool(1, "test") as pool:
        futures.extend(pool.submit(block) for _ in range(4))
        started.wait(5.0)
        raise KeyboardInterrupt
    running, *queued = futures
    assert not running.done()
    assert all(future.cancelled() for future in queued)
    release.set()
    running.result(5.0)


def test_other_errors_still_wait_for_running_calls() -> None:
    finished: list[bool] = []

    def work() -> None:
        finished.append(True)

    with pytest.raises(ValueError), worker_pool(2, "test") as pool:
        pool.submit(work)
        raise ValueError
    assert finished == [True]
