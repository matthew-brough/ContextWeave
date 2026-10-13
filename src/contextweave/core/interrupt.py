import signal
import threading
from collections.abc import Generator
from contextlib import contextmanager
from types import FrameType

EXIT_INTERRUPTED = 130


@contextmanager
def deferred_interrupt() -> Generator[None]:
    """Hold Ctrl+C until the block finishes, then deliver it; a no-op off the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    received: list[FrameType | None] = []

    def hold(_signum: int, frame: FrameType | None) -> None:
        received.append(frame)

    previous = signal.signal(signal.SIGINT, hold)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    if not received:
        return
    if callable(previous):
        previous(signal.SIGINT, received[0])
    elif previous != signal.SIG_IGN:
        raise KeyboardInterrupt
