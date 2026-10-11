import hashlib
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from contextweave.compact.lossy import prepare_lossy
from contextweave.compact.rules import compact_items
from contextweave.compact.sources import SourceCache
from contextweave.core.budget import Ceiling, Estimator, bytes35
from contextweave.core.errors import DegradedEntry, ErrorEntry, OmittedEntry, cap_omitted
from contextweave.core.refs import (
    ByteAnchor,
    PartAnchor,
    Ref,
    RefError,
    SymbolAnchor,
    encode_ref,
)
from contextweave.index.records import Chunk, SymbolRecord
from contextweave.index.store import Store, hash_bytes
from contextweave.models.providers import Device, Slot
from contextweave.retrieve.candidates import Candidate, CandidateSet
from contextweave.retrieve.context import QueryContext
from contextweave.retrieve.ladder import next_rung
from contextweave.retrieve.pack import (
    LOSSLESS_REPS,
    BudgetWire,
    ContextPack,
    CoverageWire,
    DevicesWire,
    PackItem,
    PackResult,
    PlanWire,
    RrfWire,
)
from contextweave.retrieve.pack_text import TextRender, item_lines, render_pack_text
from contextweave.retrieve.plan import DEFAULT_PLAN

STAGE_NAME = "assemble"
MIN_MAX_TOKENS = 512
HASH_LENGTH = 12
OMITTED_REASON = "budget"
HINT = "narrow with scope or raise max_tokens; omitted refs expand losslessly"
ENVELOPE_REASON_LENGTH = 60


@dataclass(slots=True)
class FileView:
    """A file's current bytes with its chunks and symbols, loaded once per assembly."""

    source: bytes
    chunks: dict[int, Chunk]
    spans: dict[tuple[int, int], Chunk]
    symbols: dict[str, list[SymbolRecord]]


def load_view(store: Store, path: str) -> FileView | None:
    try:
        source = (store.root / path).read_bytes()
    except OSError:
        return None
    chunks = store.chunks_in_file(path)
    symbols: defaultdict[str, list[SymbolRecord]] = defaultdict(list)
    for symbol in store.symbols_in_file(path):
        symbols[symbol.qualified_name].append(symbol)
    return FileView(
        source,
        {chunk.id: chunk for chunk in chunks},
        {(chunk.start_byte, chunk.end_byte): chunk for chunk in chunks},
        dict(symbols),
    )


def pick_symbol(view: FileView, name: str, chunk: Chunk) -> SymbolRecord | None:
    options = view.symbols.get(name, [])
    inside = [
        symbol
        for symbol in options
        if chunk.start_byte <= symbol.start_byte and symbol.end_byte <= chunk.end_byte
    ]
    return next(iter(inside or options), None)


def ref_of(
    key: str, cand: Candidate, chunk: Chunk, symbol: SymbolRecord | None, body: bytes
) -> str:
    """Ref per the anchor table; a symbol ref only when the symbol is exactly the served bytes (V6)."""
    sliced = hash_bytes(body)[:HASH_LENGTH]
    exact = (
        symbol is not None
        and symbol.start_byte == cand.start_byte
        and symbol.end_byte == cand.end_byte
    )
    match cand.anchor:
        case SymbolAnchor() if symbol is not None and exact:
            ref = Ref(key, cand.path, cand.anchor, symbol.content_hash[:HASH_LENGTH])
        case PartAnchor():
            ref = Ref(key, cand.path, cand.anchor, chunk.content_hash[:HASH_LENGTH])
        case _:
            ref = Ref(key, cand.path, ByteAnchor(cand.start_byte, cand.end_byte), sliced)
    try:
        return encode_ref(ref)
    except RefError:
        fallback = Ref(key, cand.path, ByteAnchor(cand.start_byte, cand.end_byte), sliced)
        return encode_ref(fallback)


def make_item(key: str, cand: Candidate, view: FileView) -> PackItem | None:
    """The served form of a candidate, or None when its chunk or bytes are gone."""
    chunk = view.chunks.get(cand.chunk_id) or view.spans.get((cand.start_byte, cand.end_byte))
    if chunk is None or cand.end_byte > len(view.source):
        return None
    body = view.source[cand.start_byte : cand.end_byte]
    symbol = None if cand.qualified_name is None else pick_symbol(view, cand.qualified_name, chunk)
    return PackItem(
        cand=cand,
        ref=ref_of(key, cand, chunk, symbol, body),
        rep="source",
        text=body.decode(errors="replace"),
        signature=symbol.signature if symbol and symbol.signature else None,
        start_line=view.source.count(b"\n", 0, cand.start_byte) + 1,
        end_line=view.source.count(b"\n", 0, max(cand.end_byte - 1, cand.start_byte)) + 1,
        oversize=chunk.oversize,
    )


def ordered_items(items: list[PackItem]) -> list[PackItem]:
    """Files by max final desc, items in a file by start_byte."""
    best: dict[str, float] = {}
    for item in items:
        best[item.cand.path] = max(best.get(item.cand.path, 0.0), item.cand.scores.final)
    return sorted(
        items,
        key=lambda item: (-best[item.cand.path], item.cand.path, *item.cand.order_key[1:]),
    )


def omitted_entries(dropped: list[PackItem]) -> list[OmittedEntry]:
    ranked = sorted(dropped, key=lambda item: (-item.cand.scores.final, *item.cand.order_key))
    return [item.to_omitted(OMITTED_REASON) for item in ranked]


def safe_generation(store: Store) -> int:
    try:
        return store.generation
    except sqlite3.Error:
        return 0


def pack_id_of(ctx: QueryContext, limit: int) -> str:
    """sha256 of query, options, config and index generation, first 12 hex."""
    parts = (
        ctx.query,
        ctx.scope or "",
        ctx.intent or "",
        ctx.mode,
        str(limit),
        repr(ctx.config),
        str(safe_generation(ctx.store)),
    )
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:HASH_LENGTH]


def device_of(ctx: QueryContext, slot: Slot) -> str:
    models = ctx.models
    active = None if models is None else models.healths[slot].active
    return Device.NONE.value if active is None else active.value


def devices_wire(ctx: QueryContext) -> DevicesWire:
    return DevicesWire(embed=device_of(ctx, Slot.EMBED), classify=device_of(ctx, Slot.CLASSIFY))


def coverage_wire(ctx: QueryContext) -> CoverageWire:
    found = ctx.coverage.value
    if found is None:
        return CoverageWire(score=0.0, sufficient=False, iterations=0, missing=[])
    return CoverageWire(
        score=found.score,
        sufficient=found.sufficient,
        iterations=found.iterations,
        missing=list(found.missing),
    )


def plan_wire(ctx: QueryContext) -> PlanWire:
    plan = ctx.plan.value or DEFAULT_PLAN
    weights = plan.rrf_weights
    return PlanWire(
        intent=ctx.intent if ctx.plan.value is None and ctx.intent else plan.intent,
        scope=plan.scope,
        rrf_weights=RrfWire(
            lexical=weights.lexical, semantic=weights.semantic, structural=weights.structural
        ),
        rels=list(plan.rels),
        compaction=plan.compaction,
        fast_path=plan.fast_path,
        planner=plan.planner,
        devices=devices_wire(ctx),
    )


class Fitter:
    """Builds the pack for a layout and trims it down the ladder until it fits the ceiling."""

    def __init__(
        self, ctx: QueryContext, ceiling: Ceiling, render: TextRender, estimator: Estimator
    ) -> None:
        self.ctx = ctx
        self.ceiling = ceiling
        self.render = render
        self.estimator = estimator
        self.pack_id = pack_id_of(ctx, ceiling.limit)
        self.plan = plan_wire(ctx)
        self.compact_envelope = False

    def cost(self, item: PackItem) -> int:
        return self.estimator("\n".join(item_lines(item.to_wire())))

    def build(
        self, live: list[PackItem], dropped: list[PackItem], cap: int
    ) -> tuple[ContextPack, str, int]:
        entries = cap_omitted(
            omitted_entries(dropped),
            omitted_max=cap,
            key=self.ctx.project_key,
            pack_id=self.pack_id,
        )
        truncated = bool(dropped) or any(item.rep not in LOSSLESS_REPS for item in live)
        budget = BudgetWire(
            served_tokens=self.ceiling.limit, ceiling=self.ceiling.limit, truncated=truncated
        )
        if truncated:
            budget["hint"] = HINT
        envelope = self.ctx.envelope
        errors = envelope.errors
        degraded = envelope.degraded
        if self.compact_envelope:
            errors = [ErrorEntry(code=error["code"]) for error in errors]
            degraded = [
                DegradedEntry(stage=entry["stage"], reason=entry["reason"][:ENVELOPE_REASON_LENGTH])
                for entry in degraded
            ]
        pack = ContextPack(
            pack_id=self.pack_id,
            plan=self.plan,
            coverage=coverage_wire(self.ctx),
            items=[item.to_wire() for item in ordered_items(live)],
            omitted=entries,
            budget=budget,
            degraded=list(degraded),
            errors=list(errors),
        )
        text = self.render(pack)
        return pack, text, self.estimator(text)

    def trim(
        self, live: list[PackItem], dropped: list[PackItem], over: int, allow_critical: bool
    ) -> bool:
        """Apply next rungs to the lowest-final items until the estimated saving covers `over`."""
        saved = 0
        progressed = False
        while saved < over:
            options = [
                (item, rung)
                for item in live
                if (rung := next_rung(item, allow_critical)) is not None
            ]
            if not options:
                break
            item, rung = min(
                options, key=lambda pair: (pair[0].cand.scores.final, *pair[0].cand.order_key)
            )
            index = live.index(item)
            after = rung.apply(item)
            saved += self.cost(item) - (0 if after is None else self.cost(after))
            if after is None:
                dropped.append(live.pop(index))
            else:
                live[index] = after
            progressed = True
        return progressed

    def largest_cap(self, live: list[PackItem], dropped: list[PackItem], omitted_max: int) -> int:
        low, high = 1, omitted_max
        while low < high:
            mid = (low + high + 1) // 2
            if self.ceiling.fits(self.build(live, dropped, mid)[2]):
                low = mid
            else:
                high = mid - 1
        return low

    def fit(self, items: list[PackItem]) -> PackResult:
        live = list(items)
        dropped: list[PackItem] = []
        omitted_max = max(1, self.ctx.config.budget.omitted_max)
        allow_critical = False
        _, _, tokens = self.build(live, dropped, 1)
        if not self.ceiling.fits(tokens):
            live = prepare_lossy(self.ctx.store, self.ctx.config.models.annotate, live)
        while not self.ceiling.fits(tokens):
            if not self.trim(live, dropped, self.ceiling.over_by(tokens), allow_critical):
                if allow_critical:
                    break
                allow_critical = True
            _, _, tokens = self.build(live, dropped, 1)
        if not self.ceiling.fits(tokens):
            self.compact_envelope = True
        cap = self.largest_cap(live, dropped, omitted_max)
        pack, text, tokens = self.build(live, dropped, cap)
        pack["budget"]["served_tokens"] = tokens
        return PackResult(pack, text, tuple(omitted_entries(dropped)))


def effective_limit(ctx: QueryContext) -> int:
    """The requested ceiling, raised to the smallest pack that can carry the envelope."""
    if ctx.max_tokens >= MIN_MAX_TOKENS:
        return ctx.max_tokens
    if not any(error["code"] == "max_tokens_raised" for error in ctx.envelope.errors):
        ctx.envelope.add_error(
            ErrorEntry(
                code="max_tokens_raised",
                message=f"max_tokens {ctx.max_tokens} below minimum {MIN_MAX_TOKENS}",
                stage=STAGE_NAME,
            )
        )
    return MIN_MAX_TOKENS


class AssembleStage:
    """Groups served candidates by file, renders the ContextPack and enforces the ceiling."""

    name = STAGE_NAME

    def __init__(
        self, render: TextRender = render_pack_text, estimator: Estimator = bytes35
    ) -> None:
        self.render = render
        self.estimator = estimator

    def assemble(self, ctx: QueryContext, cands: CandidateSet) -> PackResult:
        ceiling = Ceiling(effective_limit(ctx))
        views: dict[str, FileView | None] = {}
        sources = SourceCache(ctx.root)
        items: list[PackItem] = []
        for cand in cands:
            if cand.path not in views:
                views[cand.path] = load_view(ctx.store, cand.path)
                loaded = views[cand.path]
                if loaded is not None:
                    sources.preload(cand.path, loaded.source)
            view = views[cand.path]
            item = None if view is None else make_item(ctx.project_key, cand, view)
            if item is None:
                ctx.envelope.add_degraded(
                    DegradedEntry(stage=STAGE_NAME, reason=f"source_unavailable: {cand.path}")
                )
                continue
            items.append(item)
        compacted = compact_items(sources, tuple(ordered_items(items)))
        return Fitter(ctx, ceiling, self.render, self.estimator).fit(list(compacted))

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        ctx.pack.value = self.assemble(ctx, cands)
        return cands
