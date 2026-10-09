from contextweave.compact.rule import Rung
from contextweave.compact.rules import LADDER_RULES
from contextweave.retrieve.pack import PackItem

LADDER: tuple[Rung, ...] = LADDER_RULES.values()


def next_rung(item: PackItem, allow_critical: bool) -> Rung | None:
    """The first rung that still applies to the item; critical items stop at signature."""
    for rung in LADDER:
        if rung.applies(item, allow_critical):
            return rung
    return None


__all__ = ["LADDER", "Rung", "next_rung"]
