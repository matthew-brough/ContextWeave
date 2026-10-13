from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock

from contextweave.models.providers import Device, Slot

WINDOW = 200

type Clock = Callable[[], float]
type Warm = Callable[[Device], None]
type Submit = Callable[[Callable[[], None]], None]


@dataclass(frozen=True, slots=True)
class HealthStatus:
    slot: Slot
    active: Device | None
    down: tuple[Device, ...]
    skips_total: int
    calls_in_window: int
    skip_rate: float
    warn: bool
    slow_rate: float = 0.0
    slow_warn: bool = False
    reason: str | None = None


class SlotHealth:
    """Per-slot device chain state: active device, down marks, background warm-up, skip stats."""

    def __init__(
        self,
        slot: Slot,
        chain: Sequence[Device],
        *,
        warm: Warm,
        submit: Submit,
        clock: Clock,
        retry_after_s: float,
        skip_warn: float,
        slow_warn: float = 1.0,
        window: int = WINDOW,
        loaded: bool = True,
    ) -> None:
        self.slot = slot
        self.chain = tuple(chain)
        self._warm = warm
        self._submit = submit
        self._clock = clock
        self._retry_after_s = retry_after_s
        self._skip_warn = skip_warn
        self._slow_warn = slow_warn
        self._lock = RLock()
        self._ready: Device | None = self.chain[0] if self.chain and loaded else None
        self._down: dict[Device, float] = {}
        self._last_error: str | None = None
        self._warming = False
        self._suspended = False
        self._recent: deque[bool] = deque(maxlen=window)
        self._slow: deque[bool] = deque(maxlen=window)
        self._skips = 0

    @property
    def active(self) -> Device | None:
        """The device calls currently go to, or None when none is ready."""
        with self._lock:
            return None if self._suspended else self._ready

    @property
    def suspended(self) -> bool:
        with self._lock:
            return self._suspended

    def acquire(self) -> Device | None:
        """Return the device for one call, kicking background warm-up when a better one is due."""
        self._maybe_warm()
        return self.active

    def unavailable_reason(self) -> str:
        """Why acquire returned None."""
        with self._lock:
            if self._suspended:
                return "slot stopped by residency profile"
            if self._warming:
                return "device warming up"
            if self._last_error is not None:
                return f"no device ready, last start failed: {self._last_error}"
            return "no device ready"

    def record_success(self, *, slow: bool = False) -> None:
        """Count a call that reached a healthy device; slow means it passed its warn time."""
        with self._lock:
            self._recent.append(False)
            self._slow.append(slow)

    def record_unavailable(self) -> None:
        """Count a skip taken without attempting any device."""
        with self._lock:
            self._skips += 1
            self._recent.append(True)

    def record_failure(self, device: Device) -> None:
        """Mark the device down for retry_after_s, count the skip, warm the next device."""
        with self._lock:
            self._down[device] = self._clock() + self._retry_after_s
            if self._ready is device:
                self._ready = None
            self._skips += 1
            self._recent.append(True)
        self._maybe_warm()

    def activate(self, device: Device) -> bool:
        """Synchronously move the slot to a device; on failure mark it down and fall back."""
        with self._lock:
            if self._ready is device and not self._suspended:
                return True
            if self._warming:
                return False
            self._warming = True
            self._ready = None
        try:
            self._warm(device)
        except Exception as exc:
            with self._lock:
                self._down[device] = self._clock() + self._retry_after_s
                self._last_error = _describe(exc)
                self._warming = False
            self._maybe_warm()
            return False
        with self._lock:
            self._ready = device
            self._last_error = None
            self._warming = False
        return True

    def suspend(self) -> None:
        """Stop serving the slot (residency profile has it stopped)."""
        with self._lock:
            self._suspended = True
            self._ready = None

    def resume(self) -> None:
        """Allow the slot to serve again; a device must be activated or warmed."""
        with self._lock:
            self._suspended = False

    def skip_rate(self) -> float:
        """Fraction of skipped calls over the rolling window."""
        with self._lock:
            return sum(self._recent) / len(self._recent) if self._recent else 0.0

    def should_warn(self) -> bool:
        """True when the rolling skip rate exceeds skip_warn."""
        return self.skip_rate() > self._skip_warn

    def slow_rate(self) -> float:
        """Fraction of completed calls over the rolling window that passed their warn time."""
        with self._lock:
            return sum(self._slow) / len(self._slow) if self._slow else 0.0

    def status(self) -> HealthStatus:
        """Snapshot for status output."""
        with self._lock:
            down = tuple(device for device in self.chain if device in self._down)
            calls = len(self._recent)
        slow = self.slow_rate()
        active = self.active
        return HealthStatus(
            self.slot,
            active,
            down,
            self._skips,
            calls,
            self.skip_rate(),
            self.should_warn(),
            slow,
            slow > self._slow_warn,
            None if active is not None or not self.chain else self.unavailable_reason(),
        )

    def _candidates(self) -> list[Device]:
        now = self._clock()
        for device in [device for device, until in self._down.items() if now >= until]:
            del self._down[device]
        return [device for device in self.chain if device not in self._down]

    def _maybe_warm(self) -> None:
        with self._lock:
            if self._warming or self._suspended:
                return
            candidates = self._candidates()
            if not candidates:
                return
            target = candidates[0]
            if self._ready is not None and self.chain.index(self._ready) <= self.chain.index(
                target
            ):
                return
            self._warming = True
            self._ready = None
        self._submit(lambda: self._warm_job(target))

    def _warm_job(self, first: Device) -> None:
        target: Device | None = first
        while target is not None:
            try:
                self._warm(target)
            except Exception as exc:
                with self._lock:
                    self._down[target] = self._clock() + self._retry_after_s
                    self._last_error = _describe(exc)
                    candidates = self._candidates()
                    target = candidates[0] if candidates else None
                    if target is None:
                        self._warming = False
                continue
            with self._lock:
                self._ready = target
                self._last_error = None
                self._warming = False
            return


def _describe(exc: Exception) -> str:
    return str(exc) or type(exc).__name__
