import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from contextweave.core.config import Config
from contextweave.core.errors import OmittedEntry
from contextweave.index.projects import ProjectRegistry
from contextweave.index.store import Store

type StoreOpener = Callable[[Path], Store]


def inproc_opener(config: Config, *, readonly: bool) -> StoreOpener:
    """Opener giving every call its own connection to the project db."""

    def open_store(root: Path) -> Store:
        return Store.open(root, readonly=readonly, chunk_config=config.chunk)

    return open_store


class PackCache:
    """Bounded cache of the omitted entries that overflowed a response."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(capacity, 1)
        self._packs: OrderedDict[tuple[str, str], tuple[OmittedEntry, ...]] = OrderedDict()

    @staticmethod
    def pack_id(entries: Sequence[OmittedEntry]) -> str:
        """Content-derived id so identical overflow lists share one pack."""
        digest = hashlib.sha256(json.dumps(list(entries), sort_keys=True).encode())
        return digest.hexdigest()[:12]

    def put(self, key: str, pack_id: str, entries: Sequence[OmittedEntry]) -> None:
        self._packs[(key, pack_id)] = tuple(entries)
        self._packs.move_to_end((key, pack_id))
        while len(self._packs) > self._capacity:
            self._packs.popitem(last=False)

    def get(self, key: str, pack_id: str) -> tuple[OmittedEntry, ...] | None:
        return self._packs.get((key, pack_id))


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything a tool call needs besides its arguments."""

    config: Config
    registry: ProjectRegistry
    open_store: StoreOpener
    cwd: Path | None = None
    roots: tuple[Path, ...] = ()
    packs: PackCache = field(default_factory=lambda: PackCache(20))
