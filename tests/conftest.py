from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def isolated_state(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep project state symlink targets out of the real user state dir."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("xdg-state")))
        yield
