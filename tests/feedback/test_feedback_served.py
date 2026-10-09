from pathlib import Path

from contextweave.core.config import Config
from contextweave.feedback.served import record_served_pack
from contextweave.index.indexer import index_project
from contextweave.index.store import Store
from contextweave.retrieve.search import search

SOURCE = "def alpha():\n    return 1\n\n\ndef beta():\n    return alpha() + 1\n"


def test_served_pack_rows_match_pack_items_and_replace_on_repeat(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "app.py").write_text(SOURCE, encoding="utf-8")
    config = Config()
    index_project(tmp_path, config)
    with Store.open(tmp_path) as store:
        pack = search(store, config, "alpha").pack
        assert pack["items"]
        assert record_served_pack(store, pack, now=500)
        assert record_served_pack(store, pack, now=501)
        rows = store.query("SELECT ref, path, ts FROM served_items")
        assert sorted(str(row[0]) for row in rows) == sorted(item["ref"] for item in pack["items"])
        assert {row[2] for row in rows} == {501}


def test_readonly_store_records_nothing(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    with Store.open(tmp_path):
        pass
    with Store.open(tmp_path, readonly=True) as store:
        pack = search(store, Config(), "alpha").pack
        assert not record_served_pack(store, pack)
