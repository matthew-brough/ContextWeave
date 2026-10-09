from dataclasses import dataclass
from pathlib import Path

from contextweave.core.config import Config
from contextweave.index.records import Changed
from contextweave.index.store import Store
from contextweave.index.walk import walk_project


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What a reconcile pass changed."""

    added: tuple[str, ...]
    updated: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: int
    generation: int


def index_project(root: Path, config: Config) -> IndexReport:
    """Reconcile the db with the tree: parse new and changed files, drop vanished ones."""
    with Store.open(root, chunk_config=config.chunk) as store, store.writer_lock():
        known = {record.path: record for record in store.files()}
        walked = walk_project(root, config.index.ignore)
        added: list[str] = []
        updated: list[str] = []
        unchanged = 0
        for item in walked:
            previous = known.get(item.path)
            result = store.refresh(item.path)
            if previous is None and result.data is not None:
                added.append(item.path)
            elif (
                previous is not None
                and isinstance(result.check, Changed)
                and result.data is not None
                and result.data.file.content_hash != previous.content_hash
            ):
                updated.append(item.path)
            else:
                unchanged += 1
        present = {item.path for item in walked}
        removed = tuple(path for path in sorted(known) if path not in present)
        for path in removed:
            store.delete_file(path)
        return IndexReport(tuple(added), tuple(updated), removed, unchanged, store.generation)
