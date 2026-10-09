from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class Link:
    """One labelled edge between two named nodes."""

    source: str
    rel: str
    target: str


def render_chains(links: Iterable[Link]) -> tuple[str, ...]:
    """Render edges as maximal single-relation path lines `A -rel-> B -rel-> C`; every edge appears once."""
    unique = sorted(set(links))
    outgoing: defaultdict[tuple[str, str], list[Link]] = defaultdict(list)
    incoming: defaultdict[tuple[str, str], int] = defaultdict(int)
    for link in unique:
        outgoing[(link.rel, link.source)].append(link)
        incoming[(link.rel, link.target)] += 1

    def interior(rel: str, node: str) -> bool:
        return incoming[(rel, node)] == 1 and len(outgoing[(rel, node)]) == 1

    used: set[Link] = set()
    lines: list[str] = []

    def walk(first: Link) -> None:
        path = [first.source, first.target]
        used.add(first)
        current = first
        while interior(current.rel, current.target):
            following = outgoing[(current.rel, current.target)][0]
            if following in used:
                break
            used.add(following)
            path.append(following.target)
            current = following
        lines.append(path[0] + "".join(f" -{first.rel}-> {node}" for node in path[1:]))

    for link in unique:
        if not interior(link.rel, link.source):
            walk(link)
    for link in unique:
        if link not in used:
            walk(link)
    return tuple(lines)
