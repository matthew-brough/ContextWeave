from collections.abc import Iterator
from pathlib import Path

import pytest

_write_text = Path.write_text


def _write_text_lf(
    self: Path,
    data: str,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = "\n",
) -> int:
    return _write_text(self, data, encoding, errors, newline)


@pytest.fixture(autouse=True, scope="session")
def lf_writes() -> Iterator[None]:
    """Write test files with LF on every platform; byte offsets and hashes assume it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "write_text", _write_text_lf)
        yield


@pytest.fixture(autouse=True, scope="session")
def isolated_state(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep project state symlink targets out of the real user state dir."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("xdg-state")))
        yield
