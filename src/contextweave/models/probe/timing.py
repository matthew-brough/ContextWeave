import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

type Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class Latency:
    runs: int
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def summarize(samples_ms: Sequence[float]) -> Latency:
    """Median, nearest-rank p95, min and max of a non-empty sample list."""
    if not samples_ms:
        raise ValueError("no samples to summarize")
    ordered = sorted(samples_ms)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return Latency(
        runs=len(ordered),
        p50_ms=statistics.median(ordered),
        p95_ms=ordered[rank],
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


def time_ms(call: Callable[[], object], clock: Clock) -> float:
    """Wall time of one call in milliseconds."""
    started = clock()
    call()
    return (clock() - started) * 1000


def sample_ms(call: Callable[[], object], clock: Clock, *, runs: int, warmup: int) -> list[float]:
    """Run warmup calls untimed, then time `runs` calls."""
    for _ in range(warmup):
        call()
    return [time_ms(call, clock) for _ in range(runs)]


def rate_per_s(count: int, total_ms: float) -> float:
    """Items per second over a total wall time."""
    return count / (total_ms / 1000) if total_ms > 0 else 0.0
