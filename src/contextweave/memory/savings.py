import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from contextweave.core.types import JSON
from contextweave.memory.store import MemoryStore, SavingsBucket
from contextweave.memory.wire import SavingsBucketTotals

BYTES_PER_TOKEN = 3.5


def full_file_tokens(root: Path, paths: Iterable[str]) -> int:
    """Estimate tokens of the distinct files a pack drew from, by on-disk size."""
    total = 0
    for rel in sorted(set(paths)):
        try:
            size = (root / rel).stat().st_size
        except OSError:
            continue
        total += math.ceil(size / BYTES_PER_TOKEN)
    return total


def record_savings(
    store: MemoryStore,
    *,
    tokens_served: int,
    full_file_tokens: int,
    bucket: SavingsBucket = "retrieval",
    meta: Mapping[str, JSON] | None = None,
) -> None:
    """Append a ledger row: baseline = full-file tokens, served = tokens returned."""
    meta_json = json.dumps(meta, sort_keys=True) if meta else None
    store.add_savings(bucket, max(full_file_tokens, 0), max(tokens_served, 0), meta_json)


def savings_summary(store: MemoryStore) -> dict[SavingsBucket, SavingsBucketTotals]:
    """Per-bucket baseline, served, saved and call totals."""
    return store.savings_totals()
