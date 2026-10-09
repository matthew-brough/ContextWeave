from pathlib import Path

import pytest

from contextweave.server.context import ToolContext
from contextweave.server.tools import ToolResult

from .test_tools import error_codes, items, reindex, run

FILES = {
    "graph/base.py": "class Base:\n    def hello(self) -> str:\n        return 'x'\n",
    "graph/core.py": (
        "from graph.base import Base\n\n\n"
        "def leaf() -> int:\n    return 1\n\n\n"
        "def middle() -> int:\n    return leaf()\n\n\n"
        "def top() -> int:\n    return middle()\n\n\n"
        "class Child(Base):\n    pass\n"
    ),
    "graph/other.py": "def leaf() -> int:\n    return 2\n",
    "tests/test_core.py": "from graph.core import top\n\n\ndef test_top() -> None:\n    top()\n",
}


@pytest.fixture
def graph(project: Path, context: ToolContext) -> ToolContext:
    for relative, text in FILES.items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    reindex(project)
    return context


def names(result: ToolResult) -> list[str]:
    return [str(edge["name"]) for edge in items(result, "edges")]


def test_callers_one_hop(graph: ToolContext) -> None:
    result = run(graph, "related", target="middle", rel="callers")
    assert result.structured["found"] is True
    assert names(result) == ["top"]
    (edge,) = items(result, "edges")
    assert edge["resolution"] == "exact" and edge["hops"] == 1
    assert str(edge["ref"]).startswith("cw://")


def test_depth_follows_callers_transitively(graph: ToolContext) -> None:
    deep = run(graph, "related", target="middle", rel="callers", depth=2)
    assert set(names(deep)) == {"top", "test_top"}
    assert {str(edge["hops"]) for edge in items(deep, "edges")} == {"1", "2"}


def test_callees_by_ref(graph: ToolContext) -> None:
    ref = str(
        items(run(graph, "get_symbol", name="top", path="graph/core.py"), "symbols")[0]["ref"]
    )
    result = run(graph, "related", target=ref, rel="callees")
    assert names(result) == ["middle"]


def test_ambiguous_target_lists_candidates(graph: ToolContext) -> None:
    result = run(graph, "related", target="leaf", rel="callers")
    assert result.structured["ambiguous"] is True
    assert result.structured["edges"] == []
    assert {str(c["path"]) for c in items(result, "candidates")} == {
        "graph/core.py",
        "graph/other.py",
    }


def test_inherits_and_importers_and_tests(graph: ToolContext) -> None:
    inherits = run(graph, "related", target="Child", rel="inherits")
    assert names(inherits) == ["Base"]
    importers = run(graph, "related", target="graph/base.py", rel="importers")
    assert names(importers) == ["graph/core.py"]
    covered = run(graph, "related", target="top", rel="tests")
    assert names(covered) == ["test_top"]


def test_name_edges_hidden_unless_requested(graph: ToolContext, project: Path) -> None:
    (project / "graph" / "ext.py").write_text(
        "from lib import Thing\n\n\nclass Ext(Thing):\n    pass\n", encoding="utf-8"
    )
    reindex(project)
    hidden = run(graph, "related", target="Ext", rel="inherits")
    assert hidden.structured["edges"] == []
    shown = run(graph, "related", target="Ext", rel="inherits", include_name_edges=True)
    (edge,) = items(shown, "edges")
    assert edge["resolution"] == "name" and edge["name"] == "Thing" and "ref" not in edge


def test_unknown_target_is_not_found(graph: ToolContext) -> None:
    result = run(graph, "related", target="nope_nothing", rel="callers")
    assert result.structured["found"] is False
    assert not error_codes(result)


def test_bad_ref_and_bad_args_report_errors(graph: ToolContext) -> None:
    assert "bad_ref" in error_codes(run(graph, "related", target="cw://nonsense", rel="callers"))
    assert "bad_args" in error_codes(run(graph, "related", target="top", rel="sideways"))
    assert "bad_args" in error_codes(run(graph, "related", target="top", rel="callers", depth=9))


def test_budget_truncates_with_omitted(graph: ToolContext, project: Path) -> None:
    body = "".join(f"def caller_{n}() -> int:\n    return leaf_one()\n\n\n" for n in range(60))
    (project / "graph" / "many.py").write_text(
        "def leaf_one() -> int:\n    return 1\n\n\n" + body, encoding="utf-8"
    )
    reindex(project)
    result = run(graph, "related", target="leaf_one", rel="callers", max_tokens=300)
    assert result.structured["truncated"] is True
    assert items(result, "omitted") or result.structured.get("hint")


EVENT_FILES = {
    "fxmanifest.lua": "client_script 'client.lua'\nserver_script 'server.lua'\n",
    "client.lua": 'function ask()\n    TriggerServerEvent("res:hello")\nend\n',
    "server.lua": 'RegisterNetEvent("res:hello", function()\n    print("hi")\nend)\n',
}


def test_emitters_and_listeners_carry_event_resolution(project: Path, context: ToolContext) -> None:
    for relative, text in EVENT_FILES.items():
        (project / relative).write_text(text, encoding="utf-8")
    reindex(project)
    emitters = run(context, "related", target="res:hello", rel="emitters")
    (edge,) = items(emitters, "edges")
    assert edge["name"] == "ask" and edge["resolution"] == "event"
    listeners = run(context, "related", target="ask", rel="listeners")
    (heard,) = items(listeners, "edges")
    assert heard["name"] == "res:hello" and heard["resolution"] == "event"
