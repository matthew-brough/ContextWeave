import sqlite3
import struct
import subprocess
import sys

import pytest
import sqlite_vec
from tree_sitter import Parser
from tree_sitter_language_pack import get_language

from contextweave.cli import main


def test_sqlite_vec_loads_and_queries() -> None:
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[3] distance_metric=cosine)")
    rows = [(1, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])]
    for rowid, vec in rows:
        conn.execute(
            "INSERT INTO v(rowid, embedding) VALUES (?, ?)",
            (rowid, struct.pack("3f", *vec)),
        )
    hit = conn.execute(
        "SELECT rowid FROM v WHERE embedding MATCH ? AND k = 1",
        (struct.pack("3f", 0.9, 0.1, 0.0),),
    ).fetchone()
    assert hit == (1,)


def test_fts5_bm25() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE f USING fts5(name, body)")
    conn.executemany(
        "INSERT INTO f VALUES (?, ?)",
        [("validate", "token expiry check"), ("render", "html output")],
    )
    hit = conn.execute("SELECT name FROM f WHERE f MATCH 'expiry' ORDER BY bm25(f)").fetchone()
    assert hit == ("validate",)


@pytest.mark.parametrize(
    ("lang", "source", "root_type"),
    [
        ("python", b"def f():\n    return 1\n", "module"),
        ("lua", b"local x = 1\n", "chunk"),
        ("csharp", b"class A {}\n", "compilation_unit"),
        ("typescript", b"const a: number = 1;\n", "program"),
    ],
)
def test_language_pack_parses(lang: str, source: bytes, root_type: str) -> None:
    parser = Parser(get_language(lang))
    tree = parser.parse(source)
    assert tree.root_node.type == root_type
    assert not tree.root_node.has_error


def test_cli_runs() -> None:
    assert main([]) == 0
    out = subprocess.run(
        [sys.executable, "-m", "contextweave.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip()
