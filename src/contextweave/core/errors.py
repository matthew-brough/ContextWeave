from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import NotRequired, TypedDict

from contextweave.core.refs import pack_omitted_uri
from contextweave.core.types import JSON


class ErrorEntry(TypedDict):
    code: str
    message: NotRequired[str]
    stage: NotRequired[str]
    detail: NotRequired[dict[str, JSON]]


class DegradedEntry(TypedDict):
    stage: str
    reason: str
    device_tried: NotRequired[str]


class OmittedEntry(TypedDict):
    ref: str
    qualified_name: str
    path: str
    reason: str


class OverflowEntry(TypedDict):
    reason: str
    count: int
    ref: str


class StageError(Exception):
    """Stage failure carrying a stable error code and optional device tried."""

    def __init__(self, code: str, message: str, *, device_tried: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.device_tried = device_tried


class Envelope:
    """Collects errors and degraded entries appended by pipeline stages."""

    def __init__(self) -> None:
        self.errors: list[ErrorEntry] = []
        self.degraded: list[DegradedEntry] = []

    def add_error(self, entry: ErrorEntry) -> None:
        self.errors.append(entry)

    def add_degraded(self, entry: DegradedEntry) -> None:
        self.degraded.append(entry)

    @contextmanager
    def guard(self, stage: str) -> Generator[None]:
        """Swallow any Exception from the block, recording it against the stage."""
        try:
            yield
        except Exception as exc:
            code = exc.code if isinstance(exc, StageError) else "stage_failed"
            reason = str(exc) or type(exc).__name__
            error = ErrorEntry(code=code, message=reason, stage=stage)
            degraded = DegradedEntry(stage=stage, reason=reason)
            if isinstance(exc, StageError) and exc.device_tried is not None:
                degraded["device_tried"] = exc.device_tried
            self.add_error(error)
            self.add_degraded(degraded)


def cap_omitted(
    entries: Sequence[OmittedEntry], *, omitted_max: int, key: str, pack_id: str
) -> list[OmittedEntry | OverflowEntry]:
    """Cap entries at omitted_max, replacing the excess with one overflow tail entry."""
    if len(entries) <= omitted_max:
        return list(entries)
    kept = max(omitted_max - 1, 0)
    tail = OverflowEntry(
        reason="overflow", count=len(entries) - kept, ref=pack_omitted_uri(key, pack_id)
    )
    return [*entries[:kept], tail]
