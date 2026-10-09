from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from contextweave.index.store import Store
from contextweave.index.walk import walk_project


@dataclass(frozen=True, slots=True)
class ReconcilePlan:
    """Paths whose stat identity differs from the index, and indexed paths gone from disk."""

    changed: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return (*self.changed, *self.removed)


def plan_reconcile(root: Path, ignore: Sequence[str]) -> ReconcilePlan:
    """Stat walk against the files table; reads the db without writing."""
    with Store.open(root, readonly=True) as store:
        known = {record.path: record.stamp for record in store.files()}
    walked = walk_project(root, ignore)
    changed = tuple(
        item.path
        for item in walked
        if (stamp := known.get(item.path)) is None
        or (stamp.mtime_ns, stamp.size) != (item.mtime_ns, item.size)
    )
    present = {item.path for item in walked}
    removed = tuple(path for path in sorted(known) if path not in present)
    return ReconcilePlan(changed, removed)
