from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from contextweave.core.refs import project_key
from contextweave.index.projects import ProjectRegistry
from contextweave.index.root import ensure_self_ignore


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def state(tmp_path: Path) -> Path:
    return tmp_path / "state"


@pytest.fixture
def registry(state: Path, clock: Clock) -> ProjectRegistry:
    return ProjectRegistry(state, clock)


@pytest.fixture
def root(tmp_path: Path) -> Iterator[Path]:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    ensure_self_ignore(project)
    (project / ".contextweave" / "index.db").write_text("db", encoding="utf-8")
    yield project.resolve()


def snapshot(base: Path) -> set[str]:
    return {str(p.relative_to(base)) for p in base.rglob("*")}


def test_register_writes_meta(registry: ProjectRegistry, root: Path, state: Path) -> None:
    entry = registry.register(root)
    assert entry.key == project_key(root)
    assert entry.root == root
    assert not entry.missing
    assert (state / "projects" / entry.key / "meta.toml").is_file()


def test_register_idempotent_keeps_created(
    registry: ProjectRegistry, clock: Clock, root: Path
) -> None:
    first = registry.register(root)
    clock.now += timedelta(days=1)
    second = registry.register(root)
    assert second.created == first.created
    assert second.last_used == first.last_used + timedelta(days=1)
    assert len(registry.list()) == 1


def test_touch(registry: ProjectRegistry, clock: Clock, root: Path) -> None:
    entry = registry.register(root)
    clock.now += timedelta(hours=2)
    touched = registry.touch(entry.key)
    assert touched is not None
    assert touched.last_used == clock.now
    assert registry.touch("0" * 12) is None


def test_lookup_by_key_and_root(registry: ProjectRegistry, root: Path) -> None:
    entry = registry.register(root)
    assert registry.get(entry.key) == entry
    assert registry.get(root) == entry
    assert registry.get(str(root)) == entry
    assert registry.get("0" * 12) is None
    assert registry.known_roots() == {entry.key: root}


def test_list_marks_missing(registry: ProjectRegistry, root: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    registry.register(other)
    registry.register(root)
    other.rmdir()
    flags = {entry.root.name: entry.missing for entry in registry.list()}
    assert flags == {"proj": False, "other": True}


def test_list_empty(registry: ProjectRegistry) -> None:
    assert registry.list() == []


def test_forget_removes_everything(registry: ProjectRegistry, root: Path, state: Path) -> None:
    entry = registry.register(root)
    (state / "projects" / entry.key / "memory.db").write_text("m", encoding="utf-8")
    result = registry.forget(entry.key)
    assert result is not None and result.removed_root_state
    assert not (root / ".contextweave").exists()
    assert not (state / "projects" / entry.key).exists()
    assert (root / "a.py").read_text(encoding="utf-8") == "x = 1\n"
    assert registry.get(entry.key) is None


def test_forget_by_root_path(registry: ProjectRegistry, root: Path) -> None:
    registry.register(root)
    assert registry.forget(root) is not None
    assert registry.list() == []


def test_forget_keep_memory(registry: ProjectRegistry, root: Path, state: Path) -> None:
    entry = registry.register(root)
    directory = state / "projects" / entry.key
    (directory / "memory.db").write_text("m", encoding="utf-8")
    (directory / "memory.db-wal").write_text("w", encoding="utf-8")
    (directory / "other.bin").write_text("o", encoding="utf-8")
    registry.forget(entry.key, keep_memory=True)
    assert sorted(p.name for p in directory.iterdir()) == ["memory.db", "memory.db-wal"]
    assert not (root / ".contextweave").exists()
    assert registry.get(entry.key) is None


def test_forget_unknown(registry: ProjectRegistry) -> None:
    assert registry.forget("0" * 12) is None


def test_forget_missing_root(registry: ProjectRegistry, tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    registry.register(gone)
    gone.rmdir()
    results = registry.forget_missing()
    assert [r.root.name for r in results] == ["gone"]
    assert not results[0].removed_root_state
    assert registry.list() == []


def test_forget_missing_leaves_live(registry: ProjectRegistry, root: Path) -> None:
    registry.register(root)
    assert registry.forget_missing() == []
    assert len(registry.list()) == 1


def test_writes_confined(
    registry: ProjectRegistry, root: Path, tmp_path: Path, state: Path
) -> None:
    before = snapshot(tmp_path)
    entry = registry.register(root)
    registry.touch(entry.key)
    registry.forget(entry.key)
    changed = before ^ snapshot(tmp_path)
    assert all(c.startswith(("state", "proj/.contextweave")) for c in changed)
    assert (root / "a.py").read_text(encoding="utf-8") == "x = 1\n"
