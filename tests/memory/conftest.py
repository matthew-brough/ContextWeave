from collections.abc import Iterator
from pathlib import Path

import pytest
from memory_support import FakeEmbedder

from contextweave.memory.store import MemoryStore, memory_db_path

KEY_A = "aaaaaaaaaaaa"
KEY_B = "bbbbbbbbbbbb"


class Ticker:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        self.now += 1
        return self.now


@pytest.fixture
def state_env(tmp_path: Path) -> dict[str, str]:
    return {"XDG_STATE_HOME": str(tmp_path / "state")}


@pytest.fixture
def store_a(state_env: dict[str, str]) -> Iterator[MemoryStore]:
    store = MemoryStore.open(memory_db_path(KEY_A, state_env), KEY_A, clock=Ticker())
    yield store
    store.close()


@pytest.fixture
def store_b(state_env: dict[str, str]) -> Iterator[MemoryStore]:
    store = MemoryStore.open(memory_db_path(KEY_B, state_env), KEY_B, clock=Ticker())
    yield store
    store.close()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
