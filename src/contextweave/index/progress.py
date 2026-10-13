from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Progress:
    """How far one indexing stage has got: done of total items, with an ETA once measured."""

    stage: str
    done: int
    total: int
    eta_s: float | None = None


type ProgressSink = Callable[[Progress], None]


def discard(_: Progress) -> None:
    """A sink that drops every update."""
