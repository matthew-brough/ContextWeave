import os
import sys
from pathlib import Path
from types import TracebackType
from typing import Self

if sys.platform == "win32":

    def _try_lock(descriptor: int) -> bool:
        return True

    def _unlock(descriptor: int) -> None:
        return None

else:
    import fcntl

    def _try_lock(descriptor: int) -> bool:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


class PidLock:
    """Single-instance guard: an exclusive flock on the pidfile, which also records the pid."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        if not _try_lock(descriptor):
            os.close(descriptor)
            return False
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())
        self._descriptor = descriptor
        return True

    def release(self) -> None:
        if self._descriptor is None:
            return
        self._path.unlink(missing_ok=True)
        _unlock(self._descriptor)
        os.close(self._descriptor)
        self._descriptor = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def lock_is_held(path: Path) -> bool:
    """True when another process holds the pidfile lock."""
    try:
        descriptor = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        if _try_lock(descriptor):
            _unlock(descriptor)
            return False
        return True
    finally:
        os.close(descriptor)
