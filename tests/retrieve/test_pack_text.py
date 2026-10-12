from contextweave.retrieve.pack import ItemWire, ScoresWire
from contextweave.retrieve.pack_text import item_lines


def wire(qualified_name: str | None, kind: str, symbols: list[str]) -> ItemWire:
    return ItemWire(
        ref="r",
        rep="source",
        profile="symbols",
        group="code",
        path="m.py",
        start_line=1,
        end_line=4,
        qualified_name=qualified_name,
        kind=kind,
        symbols=symbols,
        why="",
        scores=ScoresWire(
            final=0.0,
            gate=0.0,
            relevance=0.0,
            answers_query=0.0,
            implementation=0.0,
            structural_value=0.0,
            lexical_match=0.0,
            graph_proximity=0.0,
            redundancy=0.0,
        ),
        edges=[],
        freshness="fresh",
        synthesized=False,
        oversize=False,
        text="body",
    )


def test_group_header_names_members() -> None:
    assert item_lines(wire(None, "group", ["a", "b"]))[0] == "group {a, b} L1-4"


def test_named_item_header_uses_qualified_name() -> None:
    assert item_lines(wire("a", "function", ["a"]))[0] == "function a L1-4"
