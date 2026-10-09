from pathlib import Path

from contextweave.core.config import Config
from contextweave.index.indexer import index_project
from contextweave.index.store import Store

type EdgeRow = tuple[str, str, str, str]

LABELS = """
SELECT COALESCE(fs.qualified_name, ff.path), e.rel,
       COALESCE(ts.qualified_name, tn.key_path, tf.path, e.to_name), e.resolution
FROM symbol_edges e
LEFT JOIN symbols fs ON fs.id = e.from_id
LEFT JOIN files ff ON ff.id = e.from_file_id
LEFT JOIN symbols ts ON ts.id = e.to_id
LEFT JOIN files tf ON tf.id = e.to_file_id AND e.to_id IS NULL
LEFT JOIN schema_nodes tn ON tn.id = e.to_node_id
ORDER BY 1, 2, 3, 4
"""


def write_project(root: Path, files: dict[str, str]) -> None:
    (root / ".git").mkdir(exist_ok=True)
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def build_edges(root: Path, files: dict[str, str]) -> set[EdgeRow]:
    """Index a mini project and return every edge as (from, rel, to, resolution) labels."""
    write_project(root, files)
    index_project(root, Config())
    with Store.open(root) as store:
        return {(str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in store.query(LABELS)}
