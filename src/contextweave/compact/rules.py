from contextweave.compact.chains import Link, render_chains
from contextweave.compact.imports import collapse_imports
from contextweave.compact.merge import dedupe_identity, merge_adjacent
from contextweave.compact.rule import Rule
from contextweave.compact.sources import SourceCache
from contextweave.core.registry import Registry
from contextweave.retrieve.candidates import Candidate
from contextweave.retrieve.pack import PackItem

type CandidateRule = Rule[tuple[Candidate, ...]]
type ItemRule = Rule[tuple[PackItem, ...]]

CANDIDATE_RULES: Registry[str, CandidateRule] = Registry()
ITEM_RULES: Registry[str, ItemRule] = Registry()

CANDIDATE_RULES.register("dedupe_identity", Rule("dedupe_identity", True, dedupe_identity))
CANDIDATE_RULES.register("merge_adjacent", Rule("merge_adjacent", True, merge_adjacent))
ITEM_RULES.register("collapse_imports", Rule("collapse_imports", True, collapse_imports))


def compact_candidates(
    sources: SourceCache, cands: tuple[Candidate, ...], *, allow_lossy: bool = False
) -> tuple[Candidate, ...]:
    """Run the candidate rules in registry order, lossless ones always."""
    for rule in CANDIDATE_RULES.values():
        if rule.lossless or allow_lossy:
            cands = rule.apply(sources, cands)
    return cands


def compact_items(
    sources: SourceCache, items: tuple[PackItem, ...], *, allow_lossy: bool = False
) -> tuple[PackItem, ...]:
    """Run the item rules in registry order, lossless ones always."""
    for rule in ITEM_RULES.values():
        if rule.lossless or allow_lossy:
            items = rule.apply(sources, items)
    return items


__all__ = [
    "CANDIDATE_RULES",
    "ITEM_RULES",
    "Link",
    "compact_candidates",
    "compact_items",
    "render_chains",
]
