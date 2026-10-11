import tempfile
from pathlib import Path

from compact_support import ingest, resolve_ref
from hypothesis import given, settings
from hypothesis import strategies as st

from contextweave.core.budget import bytes35
from contextweave.core.config import BudgetConfig, Config
from contextweave.core.refs import pack_omitted_uri, parse_ref, project_key
from contextweave.index.store import Store
from contextweave.retrieve.assemble import MIN_MAX_TOKENS
from contextweave.retrieve.search import SearchOptions, SearchResult, search

IMPORTS = "".join(f"import module_{n}\n" for n in range(10))


def corpus(root: Path, files: int, lines: int) -> Store:
    store = Store.open(root)
    for number in range(files):
        body = f"def handler_{number}():\n" + "    value = 0\n" * lines + "    return value\n"
        ingest(store, f"pkg/mod{number}.py", f"{IMPORTS}\n\n{body}")
    return store


def check_omitted(store: Store, result: SearchResult) -> None:
    omitted = result.pack["omitted"]
    limit = Config().budget.omitted_max
    assert len(omitted) <= limit
    key = project_key(store.root)
    for entry in omitted:
        if "count" in entry:
            assert set(entry) == {"reason", "count", "ref"}
            assert entry["ref"] == pack_omitted_uri(key, result.pack["pack_id"])
            assert entry is omitted[-1]
            assert entry["count"] >= 2
            assert len(result.omitted) == len(omitted) - 1 + entry["count"]
        else:
            assert set(entry) == {"ref", "qualified_name", "path", "reason"}
            assert parse_ref(entry["ref"]).key == key
            assert resolve_ref(store, entry["ref"]) is not None


@settings(max_examples=25, deadline=None)
@given(files=st.integers(1, 8), lines=st.integers(1, 40), budget=st.integers(MIN_MAX_TOKENS, 3000))
def test_ceiling_holds_with_compaction_enabled(files: int, lines: int, budget: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = corpus(Path(directory), files, lines)
        try:
            result = search(store, Config(), "handler value", SearchOptions(max_tokens=budget))
            assert bytes35(result.text) <= budget
            assert result.pack["budget"]["served_tokens"] <= budget
            check_omitted(store, result)
            for item in result.pack["items"]:
                assert resolve_ref(store, item["ref"]) is not None
        finally:
            store.close()


def test_overflow_tail_when_more_than_cap_are_omitted() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = corpus(Path(directory), 90, 150)
        config = Config(budget=BudgetConfig(omitted_max=4))
        try:
            result = search(store, config, "handler value", SearchOptions(max_tokens=700))
            assert bytes35(result.text) <= 700
            assert result.pack["omitted"][-1]["reason"] == "overflow"
            assert result.pack["budget"]["truncated"] is True
            check_omitted(store, result)
        finally:
            store.close()


def test_compact_stage_runs_between_dedupe_and_coverage() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = corpus(Path(directory), 2, 3)
        try:
            result = search(store, Config(), "handler value")
            stages = [timing.stage for timing in result.timings]
            assert stages[-3:] == ["compact", "coverage", "assemble"]
        finally:
            store.close()
