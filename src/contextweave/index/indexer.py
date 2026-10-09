from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from contextweave.core.config import Config
from contextweave.index.chunk_tokens import ChunkTokenizer, embed_tokenizer
from contextweave.index.derived import DerivedCache
from contextweave.index.edges import resolve_edges
from contextweave.index.embed import EmbedReport, embed_project
from contextweave.index.records import Changed
from contextweave.index.store import Store
from contextweave.index.walk import walk_project
from contextweave.models.slots import ModelSlots


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What a reconcile pass changed."""

    added: tuple[str, ...]
    updated: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: int
    generation: int
    embed: EmbedReport | None = None


def index_project(
    root: Path,
    config: Config,
    *,
    slots: ModelSlots | None = None,
    cache: DerivedCache | None = None,
    env: Mapping[str, str] | None = None,
) -> IndexReport:
    """Reconcile the db with the tree, then embed changed chunks when model slots are given.

    env enables sizing chunks with the embed model's local tokenizer.json when one is present.
    """
    tokenizer: ChunkTokenizer | None = (
        embed_tokenizer(config.models.embed.model, env) if env is not None else None
    )
    with (
        Store.open(root, chunk_config=config.chunk, tokenizer=tokenizer) as store,
        store.writer_lock(),
    ):
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
        resolve_edges(store, config.index.test_globs)
        embedded = (
            embed_project(store, slots, cache, config, tokenizer=tokenizer)
            if slots is not None and cache is not None
            else None
        )
        return IndexReport(
            tuple(added), tuple(updated), removed, unchanged, store.generation, embedded
        )
