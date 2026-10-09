from dataclasses import dataclass

from contextweave.feedback.capture import MISS, USED
from contextweave.feedback.fit import Sample
from contextweave.feedback.served import FEATURE_NAMES, Features
from contextweave.index.store import Store

SAMPLE_CAP = 3000


@dataclass(frozen=True, slots=True)
class Dataset:
    """Labelled served items plus the raw signal counts they came from."""

    samples: list[Sample]
    used_signals: int
    miss_signals: int


def _count(store: Store, signal: str) -> int:
    rows = store.query("SELECT count(*) FROM feedback WHERE signal = ?", (signal,))
    count = rows[0][0]
    return count if isinstance(count, int) else 0


def _features(row: tuple[object, ...]) -> Features | None:
    values = [float(value) for value in row[: len(FEATURE_NAMES)] if isinstance(value, float | int)]
    if len(values) != len(FEATURE_NAMES):
        return None
    a, b, c, d, e, f, g = values
    return (a, b, c, d, e, f, g)


def load_dataset(store: Store) -> Dataset:
    """Items of packs that drew feedback; used when a used signal names their ref."""
    columns = ", ".join(f"s.{name}" for name in FEATURE_NAMES)
    rows = store.query(
        f"SELECT {columns}, EXISTS(SELECT 1 FROM feedback f WHERE f.pack_id = s.pack_id "
        "AND f.ref = s.ref AND f.signal = ?) "
        "FROM served_items s WHERE s.pack_id IN (SELECT pack_id FROM feedback) "
        "ORDER BY s.ts DESC, s.rowid DESC LIMIT ?",
        (USED, SAMPLE_CAP),
    )
    samples: list[Sample] = []
    for row in rows:
        features = _features(tuple(row))
        if features is not None:
            samples.append(Sample(features, row[len(FEATURE_NAMES)] == 1))
    return Dataset(samples, _count(store, USED), _count(store, MISS))
