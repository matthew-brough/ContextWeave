from collections.abc import Iterator
from pathlib import Path

import pytest
from compact_support import SMALL

from contextweave.index.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path) as opened:
        yield opened


@pytest.fixture
def small_store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path, chunk_config=SMALL) as opened:
        yield opened
