from dataclasses import replace

from contextweave.core.errors import DegradedEntry, Envelope
from contextweave.core.refs import Anchor
from contextweave.index.records import Fresh, Gone
from contextweave.index.store import Store
from contextweave.retrieve.candidates import Candidate, CandidateSet
from contextweave.retrieve.hydrate import load_file_candidates


def freshen(store: Store, cands: CandidateSet, envelope: Envelope, stage: str) -> CandidateSet:
    """Stat-check candidate files (V3); reparse changed ones, remap by anchor or drop."""
    remaps = {path: _remap(store, path, envelope, stage) for path in _paths(cands)}
    kept: list[Candidate] = []
    for cand in cands:
        mapping = remaps[cand.path]
        if mapping is None:
            kept.append(cand)
        elif (target := mapping.get(cand.anchor)) is not None:
            kept.append(replace(target, ranks=cand.ranks, scores=cand.scores))
    return CandidateSet.of(kept)


def _paths(cands: CandidateSet) -> tuple[str, ...]:
    return tuple(dict.fromkeys(cand.path for cand in cands))


def _remap(
    store: Store, path: str, envelope: Envelope, stage: str
) -> dict[Anchor, Candidate] | None:
    result = store.refresh(path)
    if isinstance(result.check, Fresh):
        return None
    if isinstance(result.check, Gone) or result.data is None:
        return {}
    if not result.persisted:
        envelope.add_degraded(DegradedEntry(stage=stage, reason="stale_readonly"))
        return {}
    mapping: dict[Anchor, Candidate] = {}
    for cand in load_file_candidates(store, path):
        mapping.setdefault(cand.anchor, cand)
    return mapping
