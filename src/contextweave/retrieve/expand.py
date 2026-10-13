from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from contextweave.retrieve.assemble import FileView, load_view, make_item
from contextweave.retrieve.candidates import Candidate, CandidateEdge, CandidateSet
from contextweave.retrieve.context import QueryContext
from contextweave.retrieve.freshness import freshen
from contextweave.retrieve.graph import Endpoint, Hop, Link, walk
from contextweave.retrieve.hydrate import load_candidates
from contextweave.retrieve.rows import as_int
from contextweave.retrieve.score import role_of

STAGE_NAME = "expand"
ENDPOINTS_PER_CHUNK = 8
EDGES_PER_ITEM = 6


@dataclass(frozen=True, slots=True)
class Expanded:
    """A chunk the graph walk reached: how far, over which rel, from which seed."""

    cand: Candidate
    hops: int
    rel: str
    seed: Candidate


type SeedValue = Callable[[Candidate], float]


def gate_of(cand: Candidate) -> float:
    return cand.scores.gate


def final_of(cand: Candidate) -> float:
    return cand.scores.final


class GatePassTwo(Protocol):
    """Gate pass 2: scores freshly reached chunks, the slot a classifier fills."""

    def __call__(self, ctx: QueryContext, expanded: Sequence[Expanded]) -> list[Candidate]: ...


def fallback_gate_pass(ctx: QueryContext, expanded: Sequence[Expanded]) -> list[Candidate]:
    """Classify-skipped pass 2: gate = 0.5 * seed gate / hops."""
    scored: list[Candidate] = []
    for item in expanded:
        role = role_of(item.cand, ctx.config.index.test_globs)
        scores = replace(
            item.cand.scores,
            gate=0.5 * item.seed.scores.gate / item.hops,
            graph_proximity=proximity(item.hops),
            implementation=1.0 if role == "implementation" else 0.0,
        )
        origin = item.seed.qualified_name or item.seed.path
        scored.append(replace(item.cand, scores=scores, why=f"{item.rel} of {origin}"))
    return scored


def proximity(hops: int) -> float:
    """1 / (1 + hops from the nearest seed); a seed is 1.0."""
    return 1 / (1 + hops)


class ExpandStage:
    """Graph walk from confident candidates over plan.rels, new chunks scored by gate pass 2."""

    name = STAGE_NAME

    def __init__(self, gate_pass: GatePassTwo = fallback_gate_pass) -> None:
        self._gate_pass = gate_pass

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        plan = ctx.plan.value
        if plan is None:
            return cands
        return self.expand(ctx, cands, plan.rels, ctx.config.expand.max_depth, gate_of)

    def expand(
        self,
        ctx: QueryContext,
        cands: CandidateSet,
        rels: tuple[str, ...],
        max_depth: int,
        seed_value: SeedValue,
    ) -> CandidateSet:
        """Walk rels from candidates whose seed_value reaches expand.min_score; reached chunks join."""
        if not rels or len(cands) == 0:
            return cands
        seeds = [cand for cand in cands if seed_value(cand) >= ctx.config.expand.min_score]
        if not seeds:
            return cands
        owners = _seed_endpoints(ctx, seeds)
        reach = walk(ctx.store, list(owners), rels, max_depth)
        origin = _origins(owners, reach.hops)
        placed = _place_hops(ctx, reach.hops)
        known = {cand.chunk_id for cand in cands}
        fresh = self._reach_new(ctx, placed, origin, known)
        seed_ids = {seed.chunk_id for seed in seeds}
        nearest = {chunk_id: reached.hops for chunk_id, reached in placed.items()}
        merged = [_with_proximity(cand, nearest, seed_ids) for cand in cands]
        merged.extend(fresh)
        chunks = {endpoint: seed.chunk_id for endpoint, seed in owners.items()}
        chunks.update({reached.endpoint: cid for cid, reached in placed.items()})
        return _attach_edges(ctx, CandidateSet.of(merged), reach.links, chunks)

    def _reach_new(
        self,
        ctx: QueryContext,
        placed: dict[int, _Reached],
        origin: dict[Endpoint, Candidate],
        known: set[int],
    ) -> list[Candidate]:
        loaded = load_candidates(ctx.store, [cid for cid in placed if cid not in known])
        ranked = sorted(
            (
                Expanded(loaded[reached.chunk_id], reached.hops, reached.rel, seed)
                for reached in placed.values()
                if reached.chunk_id in loaded and (seed := origin.get(reached.endpoint)) is not None
            ),
            key=lambda item: (item.hops, -item.seed.scores.gate, *item.cand.order_key),
        )[: ctx.config.gate.expand_window]
        if not ranked:
            return []
        fresh = freshen(
            ctx.store, CandidateSet.of(item.cand for item in ranked), ctx.envelope, STAGE_NAME
        )
        by_chunk = {cand.chunk_id: cand for cand in fresh}
        kept = [
            replace(item, cand=by_chunk[item.cand.chunk_id])
            for item in ranked
            if item.cand.chunk_id in by_chunk
        ]
        return self._gate_pass(ctx, kept)


def _seed_endpoints(ctx: QueryContext, seeds: Sequence[Candidate]) -> dict[Endpoint, Candidate]:
    owners: dict[Endpoint, Candidate] = {}
    for seed in seeds:
        for endpoint in _endpoints_of(ctx, seed):
            owners.setdefault(endpoint, seed)
    return owners


def _endpoints_of(ctx: QueryContext, cand: Candidate) -> list[Endpoint]:
    store = ctx.store
    symbols = store.query(
        "SELECT symbol_id FROM chunk_symbols WHERE chunk_id = ? ORDER BY rowid LIMIT ?",
        (cand.chunk_id, ENDPOINTS_PER_CHUNK),
    )
    if symbols:
        return [Endpoint("symbol", as_int(row[0])) for row in symbols]
    if cand.profile == "schema":
        nodes = store.query(
            "SELECT n.id FROM schema_nodes n JOIN files f ON f.id = n.file_id"
            " WHERE f.path = ? AND n.start_byte >= ? AND n.end_byte <= ?"
            " ORDER BY n.start_byte, n.id LIMIT ?",
            (cand.path, cand.start_byte, cand.end_byte, ENDPOINTS_PER_CHUNK),
        )
        return [Endpoint("node", as_int(row[0])) for row in nodes]
    files = store.query("SELECT id FROM files WHERE path = ?", (cand.path,))
    return [Endpoint("file", as_int(files[0][0]))] if files else []


def chunk_of(ctx: QueryContext, endpoint: Endpoint) -> int | None:
    """The chunk that serves an endpoint."""
    match endpoint.kind:
        case "symbol":
            rows = ctx.store.query(
                "SELECT cs.chunk_id FROM chunk_symbols cs JOIN chunks c ON c.id = cs.chunk_id"
                " WHERE cs.symbol_id = ? ORDER BY COALESCE(c.part, 0), c.id LIMIT 1",
                (endpoint.id,),
            )
        case "file":
            rows = ctx.store.query(
                "SELECT id FROM chunks WHERE file_id = ? ORDER BY start_byte, id LIMIT 1",
                (endpoint.id,),
            )
        case "node":
            rows = ctx.store.query(
                "SELECT c.id FROM chunks c JOIN schema_nodes n ON n.file_id = c.file_id"
                " WHERE n.id = ? ORDER BY"
                " (c.start_byte <= n.start_byte AND c.end_byte > n.start_byte) DESC,"
                " c.start_byte, c.id LIMIT 1",
                (endpoint.id,),
            )
    return as_int(rows[0][0]) if rows else None


def _origins(owners: dict[Endpoint, Candidate], hops: Sequence[Hop]) -> dict[Endpoint, Candidate]:
    origin = dict(owners)
    for hop in hops:
        seed = origin.get(hop.via)
        if seed is not None:
            origin[hop.endpoint] = seed
    return origin


@dataclass(frozen=True, slots=True)
class _Reached:
    chunk_id: int
    hops: int
    rel: str
    endpoint: Endpoint


def _place_hops(ctx: QueryContext, hops: Sequence[Hop]) -> dict[int, _Reached]:
    placed: dict[int, _Reached] = {}
    for hop in hops:
        chunk_id = chunk_of(ctx, hop.endpoint)
        if chunk_id is not None and chunk_id not in placed:
            placed[chunk_id] = _Reached(chunk_id, hop.hops, hop.rel, hop.endpoint)
    return placed


def _with_proximity(cand: Candidate, nearest: dict[int, int], seed_ids: set[int]) -> Candidate:
    if cand.chunk_id in seed_ids:
        value = 1.0
    elif cand.chunk_id in nearest:
        value = proximity(nearest[cand.chunk_id])
    else:
        value = 0.0
    return cand.with_scores(replace(cand.scores, graph_proximity=value))


def _attach_edges(
    ctx: QueryContext, cands: CandidateSet, links: Sequence[Link], chunks: dict[Endpoint, int]
) -> CandidateSet:
    by_id = {cand.chunk_id: cand for cand in cands}
    views: dict[str, FileView | None] = {}
    refs: dict[int, str | None] = {}

    def ref_of_chunk(chunk_id: int) -> str | None:
        if chunk_id not in refs:
            target = by_id[chunk_id]
            if target.path not in views:
                views[target.path] = load_view(ctx.store, target.path)
            view = views[target.path]
            item = None if view is None else make_item(ctx.project_key, target, view)
            refs[chunk_id] = None if item is None else item.ref
        return refs[chunk_id]

    gathered: dict[int, list[CandidateEdge]] = {}
    for link in links:
        source, target = chunks.get(link.source), chunks.get(link.target)
        if source is None or target is None or source == target:
            continue
        if source not in by_id or target not in by_id:
            continue
        to_ref = ref_of_chunk(target)
        edge = None if to_ref is None else CandidateEdge(link.rel, to_ref, link.resolution)
        bucket = gathered.setdefault(source, [])
        if edge is not None and edge not in bucket and len(bucket) < EDGES_PER_ITEM:
            bucket.append(edge)
    return cands.map(
        lambda cand: (
            replace(cand, edges=tuple(gathered[cand.chunk_id]))
            if cand.chunk_id in gathered
            else cand
        )
    )
