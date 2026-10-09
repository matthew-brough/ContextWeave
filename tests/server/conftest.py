import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from contextweave.core.config import Config
from contextweave.core.refs import project_key
from contextweave.index.indexer import index_project
from contextweave.index.projects import ProjectRegistry
from contextweave.server.context import PackCache, ToolContext, inproc_opener

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lang"
STORAGE_SUFFIX = ".txt"
COPIED = ("core/sample.py.txt", "core/sample.go", "data/sample.json", "docs/sample.md")


def copy_fixtures(root: Path) -> None:
    for name in COPIED:
        source = FIXTURES / name
        target = root / Path(name).name.removesuffix(STORAGE_SUFFIX)
        shutil.copy(source, target)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".git").mkdir()
    copy_fixtures(root)
    index_project(root, Config())
    return root


@pytest.fixture
def key(project: Path) -> str:
    return project_key(project)


@pytest.fixture
def make_context(tmp_path: Path, project: Path) -> Callable[..., ToolContext]:
    registry = ProjectRegistry(tmp_path / "state")
    registry.register(project)

    def build(*, readonly: bool = False, cwd: Path | None = None) -> ToolContext:
        config = Config()
        return ToolContext(
            config,
            registry,
            inproc_opener(config, readonly=readonly),
            cwd=project if cwd is None else cwd,
            packs=PackCache(20),
        )

    return build


@pytest.fixture
def context(make_context: Callable[..., ToolContext]) -> ToolContext:
    return make_context()
