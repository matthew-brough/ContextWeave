from hypothesis import given
from hypothesis import strategies as st

from contextweave.compact.chains import Link, render_chains

NODES = st.sampled_from(list("abcdefgh"))
LINKS = st.lists(
    st.builds(Link, NODES, st.sampled_from(["calls", "imports"]), NODES),
    max_size=20,
)


def test_linear_chain_is_one_line() -> None:
    links = [Link("A", "calls", "B"), Link("B", "calls", "C")]
    assert render_chains(links) == ("A -calls-> B -calls-> C",)


def test_branch_stops_chain_at_fan_out() -> None:
    links = [Link("A", "calls", "B"), Link("B", "calls", "C"), Link("B", "calls", "D")]
    assert render_chains(links) == ("A -calls-> B", "B -calls-> C", "B -calls-> D")


def test_relations_never_mix_in_one_chain() -> None:
    links = [Link("A", "calls", "B"), Link("B", "imports", "C")]
    assert render_chains(links) == ("A -calls-> B", "B -imports-> C")


def test_cycle_renders_once() -> None:
    links = [Link("A", "calls", "B"), Link("B", "calls", "A")]
    assert render_chains(links) == ("A -calls-> B -calls-> A",)


def decode(lines: tuple[str, ...]) -> list[Link]:
    links: list[Link] = []
    for line in lines:
        parts = line.split(" ")
        nodes = parts[0::2]
        rels = [part[1:-2] for part in parts[1::2]]
        links.extend(
            Link(nodes[index], rels[index], nodes[index + 1]) for index in range(len(rels))
        )
    return links


@given(LINKS)
def test_chains_preserve_every_edge_exactly_once(links: list[Link]) -> None:
    decoded = decode(render_chains(links))
    assert sorted(decoded) == sorted(set(links))
