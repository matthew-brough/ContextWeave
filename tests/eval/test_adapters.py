import sqlite3
from pathlib import Path

import pytest

from contextweave.core.budget import bytes35
from contextweave.eval.adapters import AdapterContext, ReadAdapter
from contextweave.eval.catalog import default_adapters
from contextweave.eval.jsonvalue import JSON
from contextweave.eval.mcp_adapters import (
    McpAdapter,
    ToolTarget,
    extract_covered,
    extract_items,
    extract_model_calls,
    prepare_checkout,
)
from contextweave.index.store import Store


def test_catalog_names() -> None:
    assert default_adapters().names() == ("read", "ast_mcp", "cw")


def test_read_adapter(tmp_path: Path) -> None:
    text = "class A:\n    def f(self) -> None: ...\n"
    (tmp_path / "m.py").write_text(text)
    (tmp_path / "n.py").write_text("def other() -> None: ...\n")
    context = AdapterContext(tmp_path, expected={"q": ["A.f", "ghost"]})
    served = ReadAdapter(context).serve_all(["q"])[0]
    assert served.items == ("A", "A.f")
    assert served.tokens == bytes35(text)
    assert served.error == "unresolved: ghost"


def test_extract_helpers() -> None:
    payload: JSON = {
        "items": [{"qualified_name": "A"}, {"qualified_name": "A"}, {"x": {"qualified_name": "B"}}],
        "omitted": [{"qualified_name": "Z"}],
        "coverage": {"sufficient": True},
        "model_calls": 2,
    }
    assert extract_items(payload) == ("A", "B")
    assert extract_covered(payload) is True
    assert extract_covered({"coverage": {"sufficient": False}}) is False
    assert extract_covered({}) is None
    assert extract_covered([]) is None
    assert extract_model_calls(payload) == 2
    assert extract_model_calls({"model_calls": True}) == 0


def test_missing_command_degrades(tmp_path: Path) -> None:
    target = ToolTarget("definitely-not-a-real-command-xyz", (), "context_search", {})
    served = McpAdapter(target, AdapterContext(tmp_path)).serve_all(["a", "b"])
    assert len(served) == 2
    assert all(item.error is not None and "not found" in item.error for item in served)
    assert all(item.items == () for item in served)


def test_crashing_server_degrades(tmp_path: Path) -> None:
    target = ToolTarget("false", (), "context_search", {})
    served = McpAdapter(target, AdapterContext(tmp_path)).serve_all(["a"])
    assert served[0].error is not None


@pytest.mark.parametrize("query", ["x"])
def test_ast_mcp_search_when_installed(tmp_path: Path, query: str) -> None:
    import shutil

    from contextweave.eval.mcp_adapters import ast_mcp_adapter

    if shutil.which("ast-mcp") is None:
        pytest.skip("ast-mcp not installed")
    (tmp_path / "m.py").write_text("def xylophone() -> None: ...\n")
    served = ast_mcp_adapter(AdapterContext(tmp_path)).serve_all(["xylophone"])
    assert served[0].error is None
    assert "xylophone" in served[0].items


def test_prepare_rebuilds_stale_schema(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("def xylophone() -> None: ...\n")
    prepare_checkout(root, {})
    stale = sqlite3.connect(root / ".contextweave" / "index.db")
    stale.execute("PRAGMA user_version = 2")
    stale.close()
    prepare_checkout(root, {})
    with Store.open(root) as store:
        assert [record.path for record in store.files()] == ["m.py"]
