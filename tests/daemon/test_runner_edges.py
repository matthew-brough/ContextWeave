from pathlib import Path

from contextweave.core.config import Config
from contextweave.daemon.runner import IndexRunner
from contextweave.daemon.scheduler import Scheduler
from contextweave.index.indexer import index_project
from contextweave.index.store import Store

KEY = "proj"

LABELS = """
SELECT COALESCE(fs.qualified_name, ff.path), e.rel, COALESCE(ts.qualified_name, tf.path), e.resolution
FROM symbol_edges e
JOIN files ff ON ff.id = e.from_file_id
LEFT JOIN symbols fs ON fs.id = e.from_id
JOIN files tf ON tf.id = e.to_file_id
LEFT JOIN symbols ts ON ts.id = e.to_id
"""

FILES = {
    "util.py": "def helper() -> int:\n    return 1\n",
    "app.py": "from util import helper\n\n\ndef run() -> int:\n    return helper()\n",
    "fxmanifest.lua": "client_script 'client.lua'\nserver_script 'server.lua'\n",
    "client.lua": 'function ask()\n    TriggerServerEvent("res:hello")\nend\n',
    "server.lua": 'RegisterNetEvent("res:hello", function()\n    print("hi")\nend)\n',
}


def write_project(root: Path, files: dict[str, str]) -> None:
    (root / ".git").mkdir()
    for relative, text in files.items():
        (root / relative).write_text(text, encoding="utf-8")


def runner_for(root: Path) -> IndexRunner:
    return IndexRunner(Config(), Scheduler(), lambda key: root if key == KEY else None)


def edge_labels(root: Path) -> set[tuple[str, str, str, str]]:
    with Store.open(root) as store:
        return {(str(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in store.query(LABELS)}


def test_edges_survive_a_reindexed_batch(tmp_path: Path) -> None:
    write_project(tmp_path, FILES)
    index_project(tmp_path, Config())
    before = edge_labels(tmp_path)
    assert ("run", "calls", "helper", "exact") in before
    assert ("ask", "emits", "res:hello", "event") in before

    (tmp_path / "util.py").write_text(
        "def helper() -> int:\n    return 2\n\n\ndef extra() -> int:\n    return 3\n",
        encoding="utf-8",
    )
    (tmp_path / "server.lua").write_text(
        'RegisterNetEvent("res:hello", function()\n    print("changed")\nend)\n', encoding="utf-8"
    )
    runner_for(tmp_path).index_paths(KEY, tmp_path, ("util.py", "server.lua"))

    after = edge_labels(tmp_path)
    assert ("run", "calls", "helper", "exact") in after
    assert ("ask", "emits", "res:hello", "event") in after
    assert ("app.py", "imports", "util.py", "exact") in after


def test_edges_follow_a_new_file_and_a_deletion(tmp_path: Path) -> None:
    write_project(tmp_path, FILES)
    index_project(tmp_path, Config())
    (tmp_path / "late.py").write_text(
        "from util import helper\n\n\ndef late() -> int:\n    return helper()\n", encoding="utf-8"
    )
    (tmp_path / "client.lua").unlink()
    runner_for(tmp_path).index_paths(KEY, tmp_path, ("late.py", "client.lua"))

    labels = edge_labels(tmp_path)
    assert ("late", "calls", "helper", "exact") in labels
    assert not {label for label in labels if label[1] == "emits"}


def test_unchanged_batch_keeps_edges(tmp_path: Path) -> None:
    write_project(tmp_path, FILES)
    index_project(tmp_path, Config())
    before = edge_labels(tmp_path)
    runner_for(tmp_path).index_paths(KEY, tmp_path, ("util.py",))
    assert edge_labels(tmp_path) == before
