from pathlib import Path

from contextweave.core.config import Config, IndexConfig
from contextweave.index.indexer import index_project
from contextweave.index.progress import Progress
from contextweave.index.root import state_dir
from contextweave.index.store import Store


def test_reconcile_add_update_delete(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    pass\n", encoding="utf-8")
    first = index_project(tmp_path, Config())
    assert first.added == ("a.py", "b.py")
    assert first.updated == () and first.removed == () and first.unchanged == 0

    again = index_project(tmp_path, Config())
    assert again.added == () and again.updated == () and again.unchanged == 2
    assert again.generation == first.generation

    (tmp_path / "a.py").write_text("def a2():\n    pass\n\n# edit\n", encoding="utf-8")
    (tmp_path / "b.py").unlink()
    (tmp_path / "c.md").write_text("# Head\n", encoding="utf-8")
    third = index_project(tmp_path, Config())
    assert third.added == ("c.md",)
    assert third.updated == ("a.py",)
    assert third.removed == ("b.py",)
    assert third.generation > first.generation

    with Store.open(tmp_path, readonly=True) as store:
        assert [record.path for record in store.files()] == ["a.py", "c.md"]
        assert store.symbols_by_name("a") == ()
        assert [s.name for s in store.symbols_by_name("a2")] == ["a2"]


def test_newly_ignored_file_is_removed(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "skip.py").write_text("y = 1\n", encoding="utf-8")
    index_project(tmp_path, Config())
    config = Config(index=IndexConfig(ignore=("skip.py",)))
    report = index_project(tmp_path, config)
    assert report.removed == ("skip.py",)


def test_state_dir_not_indexed_and_self_ignored(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    index_project(tmp_path, Config())
    report = index_project(tmp_path, Config())
    assert report.added == ()
    assert (state_dir(tmp_path) / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_reports_scan_and_edge_progress(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    seen: list[Progress] = []
    index_project(tmp_path, Config(), on_progress=seen.append)
    assert seen == [
        Progress("scan", 0, 2),
        Progress("scan", 1, 2),
        Progress("scan", 2, 2),
        Progress("edges", 0, 1),
        Progress("edges", 1, 1),
    ]
