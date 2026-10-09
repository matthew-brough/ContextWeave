from collections.abc import Callable
from dataclasses import dataclass

from contextweave.retrieve.pack import PackItem


@dataclass(frozen=True, slots=True)
class Rung:
    """One compaction step: where it applies and what it turns an item into (None = drop to ref)."""

    name: str
    applies: Callable[[PackItem, bool], bool]
    apply: Callable[[PackItem], PackItem | None]


def _can_sign(item: PackItem, allow_critical: bool) -> bool:
    return (
        item.rep == "source" and item.signature is not None and len(item.signature) < len(item.text)
    )


def _can_drop(item: PackItem, allow_critical: bool) -> bool:
    return allow_critical or not item.cand.critical


def _to_signature(item: PackItem) -> PackItem | None:
    return item.with_signature_rep()


def _to_ref(item: PackItem) -> PackItem | None:
    return None


LADDER: tuple[Rung, ...] = (
    Rung("signature", _can_sign, _to_signature),
    Rung("ref", _can_drop, _to_ref),
)


def next_rung(item: PackItem, allow_critical: bool) -> Rung | None:
    """The first rung that still applies to the item; critical items stop at signature."""
    for rung in LADDER:
        if rung.applies(item, allow_critical):
            return rung
    return None
