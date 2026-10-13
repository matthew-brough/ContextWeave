from dataclasses import replace
from pathlib import Path

from contextweave.core.budget import bytes35
from contextweave.core.types import JSON
from contextweave.memory.savings import full_file_tokens, ledger_sink, savings_summary
from contextweave.memory.store import MemoryStore, SavingsBucket, memory_db_path
from contextweave.server.context import ToolContext
from contextweave.server.tools import TOOLS, served_paths

type Row = tuple[str, SavingsBucket, int, int]


def recording(context: ToolContext) -> tuple[ToolContext, list[Row]]:
    rows: list[Row] = []

    def sink(key: str, bucket: SavingsBucket, served: int, baseline: int) -> None:
        rows.append((key, bucket, served, baseline))

    return replace(context, ledger=sink), rows


def test_search_records_served_against_full_files(
    context: ToolContext, project: Path, key: str
) -> None:
    ctx, rows = recording(context)
    result = TOOLS.get("context_search").run(ctx, {"query": "render top level"})
    assert rows == [
        (
            key,
            "retrieval",
            bytes35(result.text),
            full_file_tokens(project, served_paths(result.structured)),
        )
    ]
    _, _, served, baseline = rows[0]
    assert served > 0 and baseline > 0


def test_failed_call_records_nothing(context: ToolContext) -> None:
    ctx, rows = recording(context)
    TOOLS.get("context_search").run(ctx, {"query": ""})
    TOOLS.get("get_symbol").run(ctx, {"name": "no_such_symbol_anywhere"})
    assert rows == []


def test_served_paths_skip_omitted_and_diagnostics() -> None:
    payload: dict[str, JSON] = {
        "path": "a.py",
        "items": [{"path": "b.py"}, {"path": "a.py"}],
        "omitted": [{"path": "c.py"}],
        "errors": [{"path": "d.py"}],
    }
    assert served_paths(payload) == {"a.py", "b.py"}


def test_ledger_sink_writes_project_memory(tmp_path: Path) -> None:
    env = {"XDG_STATE_HOME": str(tmp_path)}
    ledger_sink(env)("k1", "progressive_disclosure", 10, 40)
    store = MemoryStore.open(memory_db_path("k1", env), "k1")
    try:
        totals = savings_summary(store)
    finally:
        store.close()
    assert totals["progressive_disclosure"] == {
        "baseline": 40,
        "served": 10,
        "saved": 30,
        "calls": 1,
    }
    assert totals["retrieval"]["calls"] == 0
