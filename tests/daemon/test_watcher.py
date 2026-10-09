import threading
from collections.abc import Sequence
from pathlib import Path

from contextweave.daemon.watcher import EventKind, FileEvent, PollWatcher


def make(root: Path) -> PollWatcher:
    return PollWatcher(root, (), 0.05, lambda events: None)


def test_scan_reports_create_modify_delete(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x = 1\n")
    (tmp_path / "gone.py").write_text("y = 1\n")
    watcher = make(tmp_path)
    watcher.prime()
    assert watcher.scan() == []
    (tmp_path / "new.py").write_text("z = 1\n")
    (tmp_path / "keep.py").write_text("x = 22\n")
    (tmp_path / "gone.py").unlink()
    events = set(watcher.scan())
    assert events == {
        FileEvent(EventKind.CREATED, "new.py"),
        FileEvent(EventKind.MODIFIED, "keep.py"),
        FileEvent(EventKind.DELETED, "gone.py"),
    }
    assert watcher.scan() == []


def test_ignored_and_state_files_are_invisible(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build/\n")
    watcher = make(tmp_path)
    watcher.prime()
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.py").write_text("a = 1\n")
    (tmp_path / ".contextweave").mkdir()
    (tmp_path / ".contextweave" / "index.db").write_bytes(b"x")
    assert watcher.scan() == []


def test_background_thread_delivers_events(tmp_path: Path) -> None:
    seen: list[FileEvent] = []
    arrived = threading.Event()

    def deliver(events: Sequence[FileEvent]) -> None:
        seen.extend(events)
        arrived.set()

    watcher = PollWatcher(tmp_path, (), 0.05, deliver)
    watcher.start()
    try:
        (tmp_path / "late.py").write_text("q = 1\n")
        assert arrived.wait(5)
    finally:
        watcher.stop()
    assert FileEvent(EventKind.CREATED, "late.py") in seen
