import stat
import sys
from pathlib import Path

import pytest

from contextweave.core import atomic
from contextweave.core.atomic import write_atomic


def test_creates_parents_and_writes_bytes_verbatim(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.json"
    write_atomic(target, b"x\r\ny\n")
    assert target.read_bytes() == b"x\r\ny\n"
    assert [path.name for path in target.parent.iterdir()] == ["b.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes and symlinks")
def test_keeps_mode_and_writes_through_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("old")
    real.chmod(0o600)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    write_atomic(link, b"new")
    assert link.is_symlink()
    assert real.read_text() == "new"
    assert stat.S_IMODE(real.stat().st_mode) == 0o600


def test_interrupted_replace_leaves_target_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "settings.json"
    target.write_text("old")

    def interrupt(_source: Path, _target: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic, "_replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_atomic(target, b"new")
    assert target.read_text() == "old"
    assert [path.name for path in tmp_path.iterdir()] == ["settings.json"]


def test_retries_a_briefly_locked_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "settings.json"
    real_replace = Path.replace
    refusals = [PermissionError("held open")]

    def flaky(self: Path, other: Path) -> Path:
        if refusals:
            raise refusals.pop()
        return real_replace(self, other)

    monkeypatch.setattr(atomic, "REPLACE_ATTEMPTS", 3)
    monkeypatch.setattr(atomic, "REPLACE_BACKOFF_S", 0.0)
    monkeypatch.setattr(Path, "replace", flaky)
    write_atomic(target, b"new")
    assert target.read_bytes() == b"new"
