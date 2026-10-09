import time

from contextweave.index.store import Store
from contextweave.retrieve.pack import ContextPack, ScoresWire

FEATURE_NAMES = (
    "answers_query",
    "relevance",
    "implementation",
    "lexical_match",
    "structural_value",
    "graph_proximity",
    "redundancy",
)
RETENTION_S = 90 * 86_400

type Features = tuple[float, float, float, float, float, float, float]


def features_of(scores: ScoresWire) -> Features:
    """The score components a weight fit can move, in FEATURE_NAMES order."""
    return (
        scores["answers_query"],
        scores["relevance"],
        scores["implementation"],
        scores["lexical_match"],
        scores["structural_value"],
        scores["graph_proximity"],
        scores["redundancy"],
    )


def record_served_pack(store: Store, pack: ContextPack, now: int | None = None) -> bool:
    """Persist the pack's per-item score components; fail-open, False when nothing was written."""
    if store.readonly or not pack["pack_id"] or not pack["items"]:
        return False
    stamp = int(time.time()) if now is None else now
    rows = [
        (pack["pack_id"], item["ref"], item["path"], *features_of(item["scores"]), stamp)
        for item in pack["items"]
    ]
    columns = ", ".join(("pack_id", "ref", "path", *FEATURE_NAMES, "ts"))
    marks = ", ".join("?" * (len(FEATURE_NAMES) + 4))
    cutoff = stamp - RETENTION_S
    try:
        with store.write_transaction():
            store.execute("DELETE FROM served_items WHERE pack_id = ?", (pack["pack_id"],))
            for row in rows:
                store.execute(f"INSERT INTO served_items({columns}) VALUES ({marks})", row)
            store.execute("DELETE FROM served_items WHERE ts < ?", (cutoff,))
            store.execute("DELETE FROM feedback WHERE ts < ?", (cutoff,))
    except Exception:
        return False
    return True
