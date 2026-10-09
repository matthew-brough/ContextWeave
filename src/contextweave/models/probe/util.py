from collections.abc import Callable
from dataclasses import dataclass

from contextweave.models.probe.host import HostInspector


@dataclass(frozen=True, slots=True)
class Failure:
    reason: str


def attempt[T](call: Callable[[], T]) -> T | Failure:
    """Run a probe call; any failure becomes a Failure so the probe can degrade."""
    try:
        return call()
    except Exception as exc:
        return Failure(str(exc) or type(exc).__name__)


def gpu_used_mib(host: HostInspector) -> float | None:
    reading = host.gpu()
    return None if reading is None else reading.used_mib
