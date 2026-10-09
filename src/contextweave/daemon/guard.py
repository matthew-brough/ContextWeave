import threading
from collections.abc import Generator
from contextlib import contextmanager


class ProjectGuard:
    """Shared holders (calls, incremental jobs) block a db swap, which needs exclusive access."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._shared = 0
        self._exclusive = False

    @contextmanager
    def shared(self) -> Generator[None]:
        with self._condition:
            self._condition.wait_for(lambda: not self._exclusive)
            self._shared += 1
        try:
            yield
        finally:
            with self._condition:
                self._shared -= 1
                self._condition.notify_all()

    @contextmanager
    def exclusive(self) -> Generator[None]:
        with self._condition:
            self._condition.wait_for(lambda: not self._exclusive)
            self._exclusive = True
            self._condition.wait_for(lambda: self._shared == 0)
        try:
            yield
        finally:
            with self._condition:
                self._exclusive = False
                self._condition.notify_all()
