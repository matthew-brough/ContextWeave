import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from contextweave.core.config import Config
from contextweave.index.chunk_tokens import ChunkTokenizer
from contextweave.index.root import state_dir
from contextweave.index.store import DB_NAME, Store
from contextweave.index.walk import walk_project

SHADOW_NAME = "index.next.db"
STAGING_DIR = "rebuild"
SIDECARS = ("-wal", "-shm")

type Finish = Callable[[Store], object]


@dataclass(frozen=True, slots=True)
class RebuildReport:
    files: int
    generation: int


def shadow_path(root: Path) -> Path:
    return state_dir(root) / SHADOW_NAME


def _staging_root(root: Path) -> Path:
    return state_dir(root) / STAGING_DIR


def build_shadow(
    root: Path,
    config: Config,
    *,
    tokenizer: ChunkTokenizer | None = None,
    finish: Finish | None = None,
) -> RebuildReport:
    """Build a complete index into index.next.db while the live db keeps serving."""
    staging = _staging_root(root)
    shutil.rmtree(staging, ignore_errors=True)
    shadow = shadow_path(root)
    for suffix in ("", *SIDECARS):
        Path(f"{shadow}{suffix}").unlink(missing_ok=True)
    staging.mkdir(parents=True)
    try:
        with Store.open(staging, chunk_config=config.chunk, tokenizer=tokenizer) as store:
            store.root = root
            walked = walk_project(root, config.index.ignore)
            for item in walked:
                store.refresh(item.path)
            if finish is not None:
                finish(store)
            generation = store.generation
            _checkpoint(store)
        staged = state_dir(staging) / DB_NAME
        staged.replace(shadow)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return RebuildReport(len(walked), generation)


def _checkpoint(store: Store) -> None:
    store.query("PRAGMA wal_checkpoint(TRUNCATE)")


def swap_shadow(root: Path) -> bool:
    """Atomically replace the live db with the finished shadow; False when none exists."""
    shadow = shadow_path(root)
    if not shadow.is_file():
        return False
    live = state_dir(root) / DB_NAME
    if live.is_file():
        connection = sqlite3.connect(live)
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            pass
        finally:
            connection.close()
    shadow.replace(live)
    for suffix in SIDECARS:
        Path(f"{live}{suffix}").unlink(missing_ok=True)
    return True
