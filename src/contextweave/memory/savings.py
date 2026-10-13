import json
import math
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from contextweave.core.types import JSON
from contextweave.memory.store import MemoryStore, MemoryStoreError, SavingsBucket, memory_db_path
from contextweave.memory.wire import SavingsBucketTotals

BYTES_PER_TOKEN = 3.5

type SavingsSink = Callable[[str, SavingsBucket, int, int], None]


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


def ledger_sink(env: Mapping[str, str]) -> SavingsSink:
    """Sink that appends (key, bucket, served, baseline) to the project's memory.db; fail-open."""

    def sink(key: str, bucket: SavingsBucket, served: int, baseline: int) -> None:
        try:
            store = MemoryStore.open(memory_db_path(key, env), key)
        except MemoryStoreError, sqlite3.Error, OSError:
            return
        try:
            record_savings(store, tokens_served=served, full_file_tokens=baseline, bucket=bucket)
        except sqlite3.Error:
            return
        finally:
            store.close()

    return sink
