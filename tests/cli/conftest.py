from pathlib import Path

import pytest
from cli_support import cw, make_env, make_project


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return make_env(tmp_path)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return make_project(tmp_path)


@pytest.fixture
def initialised(root: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]) -> Path:
    result = cw(capsys, env, root, "init")
    assert result.code == 0, result.err
    return root
