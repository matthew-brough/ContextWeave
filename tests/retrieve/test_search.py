import tempfile
from dataclasses import replace
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from retrieve_support import ingest

from contextweave.core.budget import bytes35
from contextweave.core.config import Config, PipelineConfig
from contextweave.core.refs import parse_ref, project_key
from contextweave.index.store import Store
from contextweave.retrieve.assemble import MIN_MAX_TOKENS
from contextweave.retrieve.search import SearchOptions, context_search, search

WIDGET = (
    "class Widget:\n"
    "    def render(self):\n"
    "        return 1\n"
    "\n\n"
    "def widget_factory():\n"
    "    return Widget()\n"
)
PARSER = (
    "def parse_tokens(text):\n    return text.split()\n\n\ndef parse_config(text):\n    return {}\n"
)


def seed(store: Store) -> None:
    ingest(store, "pkg/widget.py", WIDGET)
    ingest(store, "pkg/parser.py", PARSER)
    ingest(store, "tests/test_widget.py", "def test_widget():\n    assert True\n")


def test_fast_path_pack(store: Store) -> None:
    seed(store)
    result = search(store, Config(), "Widget")
    pack = result.pack
    assert pack["plan"]["fast_path"] is True
    assert pack["plan"]["intent"] == "lookup"
    assert [i["qualified_name"] for i in pack["items"]] == ["Widget"]
    assert pack["items"][0]["rep"] == "source"
    assert pack["items"][0]["why"] == "exact symbol match"
    assert pack["items"][0]["scores"]["gate"] == 1.0
    assert [t.stage for t in result.timings] == ["plan", "assemble"]


def test_full_pipeline_runs_all_implemented_stages_in_order(store: Store) -> None:
    seed(store)
    result = search(store, Config(), "parse the config text")
    assert [t.stage for t in result.timings] == [
        "plan",
        "retrieve",
        "fuse",
        "gate",
        "expand",
        "dedupe",
        "compact",
        "coverage",
        "assemble",
    ]
    pack = result.pack
    assert pack["plan"]["fast_path"] is False
    assert pack["items"]
    assert pack["items"][0]["path"] == "pkg/parser.py"
    assert pack["errors"] == []
    assert len({i["ref"] for i in pack["items"]}) == len(pack["items"])
    key = project_key(store.root)
    assert all(parse_ref(i["ref"]).key == key for i in pack["items"])


def test_deterministic_for_same_state(store: Store) -> None:
    seed(store)
    first = context_search(store, Config(), "render widget factory")
    second = context_search(store, Config(), "render widget factory")
    assert first == second


def test_pack_id_changes_with_query_and_budget(store: Store) -> None:
    seed(store)
    base = context_search(store, Config(), "parse text")["pack_id"]
    assert context_search(store, Config(), "parse tokens")["pack_id"] != base
    assert (
        context_search(store, Config(), "parse text", SearchOptions(max_tokens=900))["pack_id"]
        != base
    )
    assert len(base) == 12


def test_scope_option_restricts_paths(store: Store) -> None:
    seed(store)
    pack = context_search(store, Config(), "widget", SearchOptions(scope="tests", mode="lexical"))
    assert {i["path"] for i in pack["items"]} == {"tests/test_widget.py"}
    assert pack["plan"]["scope"] == "tests"


def test_intent_option_sets_plan_and_rels(store: Store) -> None:
    seed(store)
    pack = context_search(store, Config(), "parse text here", SearchOptions(intent="debug"))
    assert pack["plan"]["intent"] == "debug"
    assert "callers" in pack["plan"]["rels"]


def test_stale_file_is_reparsed_before_answer(store: Store) -> None:
    seed(store)
    target = store.root / "pkg" / "parser.py"
    target.write_text("def parse_tokens(text):\n    return 'changed marker'\n", encoding="utf-8")
    pack = context_search(store, Config(), "parse_tokens", SearchOptions(mode="lexical"))
    assert any("changed marker" in item["text"] for item in pack["items"])


def test_unmatched_query_gives_empty_valid_pack(store: Store) -> None:
    seed(store)
    pack = context_search(store, Config(), "zzzqqq nothing matches")
    assert pack["items"] == []
    assert pack["budget"]["truncated"] is False


def test_punctuation_query_does_not_raise(store: Store) -> None:
    seed(store)
    pack = context_search(store, Config(), '"((*** ::')
    assert pack["items"] == []


def test_semantic_mode_degrades_but_answers(store: Store) -> None:
    seed(store)
    pack = context_search(store, Config(), "widget", SearchOptions(mode="semantic"))
    assert pack["items"] == []
    assert pack["degraded"] and pack["degraded"][0]["stage"] == "retrieve"


def test_disabled_assemble_still_returns_valid_pack(store: Store) -> None:
    seed(store)
    config = replace(Config(), pipeline=PipelineConfig(assemble=False))
    pack = context_search(store, config, "parse text")
    assert pack["items"] == []
    assert pack["budget"]["ceiling"] == 6000


def test_disabled_plan_falls_back_to_defaults(store: Store) -> None:
    seed(store)
    config = replace(Config(), pipeline=PipelineConfig(plan=False))
    pack = context_search(store, config, "Widget")
    assert pack["plan"]["fast_path"] is False
    assert pack["items"]


def test_test_files_are_scored_with_implementation_zero(store: Store) -> None:
    seed(store)
    pack = context_search(store, Config(), "test widget", SearchOptions(mode="lexical"))
    by_path = {i["path"]: i["scores"]["implementation"] for i in pack["items"]}
    assert by_path["tests/test_widget.py"] == 0.0
    assert by_path["pkg/widget.py"] == 1.0


def test_default_ceiling_is_search_max_tokens(store: Store) -> None:
    seed(store)
    assert context_search(store, Config(), "x")["budget"]["ceiling"] == 6000
    assert (
        context_search(store, Config(), "x", SearchOptions(max_tokens=2000))["budget"]["ceiling"]
        == 2000
    )


NAMES = st.lists(
    st.sampled_from(
        ["alpha", "beta", "gamma", "delta", "parse", "render", "load", "store", "index", "query"]
    ),
    min_size=1,
    max_size=3,
)
FILES = st.lists(
    st.tuples(st.integers(1, 4), NAMES, st.integers(1, 60)),
    min_size=1,
    max_size=8,
)
QUERIES = st.one_of(
    NAMES.map(" ".join),
    st.sampled_from(["alpha", "parse render", "index query store", "load", "beta_0", "zzz"]),
)


def build_corpus(root: Path, files: list[tuple[int, list[str], int]]) -> Store:
    store = Store.open(root)
    for number, (depth, names, lines) in enumerate(files):
        folder = "/".join(f"d{level}" for level in range(depth - 1))
        path = f"{folder}/mod{number}.py".lstrip("/")
        defs = [
            f"def {name}_{number}_{index}():\n" + "    value = 0\n" * lines + "    return value\n"
            for index, name in enumerate(names)
        ]
        ingest(store, path, "\n\n".join(defs))
    return store


@settings(max_examples=60, deadline=None)
@given(files=FILES, query=QUERIES, budget=st.integers(MIN_MAX_TOKENS, 4000))
def test_output_never_exceeds_ceiling(
    files: list[tuple[int, list[str], int]], query: str, budget: int
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = build_corpus(Path(directory), files)
        try:
            result = search(store, Config(), query, SearchOptions(max_tokens=budget))
            unlimited = search(store, Config(), query, SearchOptions(max_tokens=10**7))
        finally:
            store.close()
    pack = result.pack
    assert bytes35(result.text) <= budget
    assert pack["budget"]["ceiling"] == budget
    assert pack["budget"]["served_tokens"] <= budget
    assert len(pack["omitted"]) <= Config().budget.omitted_max
    served = [item["ref"] for item in pack["items"]]
    omitted = [entry["ref"] for entry in result.omitted]
    assert len(set(served) | set(omitted)) == len(served) + len(omitted)
    assert {*served, *omitted} == {item["ref"] for item in unlimited.pack["items"]}
    for ref in omitted:
        parse_ref(ref)
    assert pack["budget"]["truncated"] == (
        len(served) < len(unlimited.pack["items"])
        or any(item["rep"] != "source" for item in pack["items"])
    )
