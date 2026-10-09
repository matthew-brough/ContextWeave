import pytest
from retrieve_support import ingest

from contextweave.index.store import Store
from contextweave.retrieve.structural import structural_search


def seed(store: Store) -> None:
    ingest(store, "pkg/widget.py", "class Widget:\n    def render(self):\n        return 1\n")
    ingest(
        store,
        "pkg/other.py",
        "def widget_factory():\n    return 1\n\n\ndef unrelated():\n    return 2\n",
    )
    ingest(store, "docs/notes.py", "def render_all():\n    return 3\n")


def test_exact_before_prefix(store: Store) -> None:
    seed(store)
    hits = structural_search(store, "Widget", scope=None, limit=10)
    names = [hit.qualified_name for hit in hits]
    assert names[0] == "Widget"
    assert "widget_factory" in names
    assert [hit.ranks.structural for hit in hits] == list(range(1, len(hits) + 1))
    assert all(hit.ranks.lexical is None for hit in hits)


def test_qualified_name_match(store: Store) -> None:
    seed(store)
    hits = structural_search(store, "Widget.render", scope=None, limit=10)
    assert hits[0].qualified_name == "Widget.render" or hits[0].path == "pkg/widget.py"


def test_path_match(store: Store) -> None:
    seed(store)
    hits = structural_search(store, "notes", scope=None, limit=10)
    assert {hit.path for hit in hits} == {"docs/notes.py"}


def test_scope_and_limit(store: Store) -> None:
    seed(store)
    scoped = structural_search(store, "render", scope="docs", limit=10)
    assert {hit.path for hit in scoped} == {"docs/notes.py"}
    assert len(structural_search(store, "render", scope=None, limit=1)) == 1


@pytest.mark.parametrize("query", ["", "%", "_", "'", '"', "the", "a.b.c", "\x00"])
def test_odd_queries_never_raise(store: Store, query: str) -> None:
    seed(store)
    structural_search(store, query, scope=None, limit=10)
