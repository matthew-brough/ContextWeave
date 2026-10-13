from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def worker_pool(workers: int, prefix: str) -> Generator[ThreadPoolExecutor]:
    """A thread pool that, on Ctrl+C, drops queued calls and stops waiting on running ones."""
    pool = ThreadPoolExecutor(workers, thread_name_prefix=prefix)
    interrupted = False
    try:
        yield pool
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
