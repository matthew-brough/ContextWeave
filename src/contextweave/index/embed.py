import hashlib
import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from contextweave.core.config import Config, EmbedConfig
from contextweave.core.errors import DegradedEntry
from contextweave.index.chunk_tokens import ChunkTokenizer, FileTokenizer, default_tokenizer
from contextweave.index.derived import WHOLE, DerivedCache
from contextweave.index.progress import Progress, ProgressSink, discard
from contextweave.index.store import Store
from contextweave.index.vectors import (
    ANNOT,
    CODE,
    KINDS,
    Space,
    SpaceKind,
    VectorInput,
    VectorPart,
    ensure_space,
    find_space,
    mark_live,
    pending_inputs,
    purge_orphans,
    vec_available,
    write_vectors,
)
from contextweave.models.errors import ModelContextError
from contextweave.models.providers import EmbedClient, EmbedKind, Slot
from contextweave.models.slots import ModelSlots, Skipped

STAGE = "embed"
DEFAULT_BATCH = 16
DEFAULT_MAX_INPUT_TOKENS = 8192
SERVER_RESERVED_TOKENS = 1
WRITE_ROWS = 256
OVERSIZE = "#parts"
CALL_PARTS = 4
QUEUE_DEPTH = 4


@dataclass(frozen=True, slots=True)
class EmbedReport:
    """What one embed pass did; pending > 0 or degraded entries mean it should run again."""

    embedded: int = 0
    cached: int = 0
    pending: int = 0
    live: tuple[str, ...] = ()
    degraded: tuple[DegradedEntry, ...] = field(default=())


def embed_model_id(config: EmbedConfig) -> str:
    """Cache identity of document vectors: the model plus a digest of the document prefix."""
    digest = hashlib.sha256(config.doc_prefix.encode()).hexdigest()[:8]
    return f"{config.model}@{digest}"


def truncate_to_tokens(text: str, tokenizer: ChunkTokenizer, limit: int) -> str:
    """The longest prefix of text counting at most limit tokens; a model-context cut, not a chunk cut."""
    if tokenizer.count(text) <= limit:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if tokenizer.count(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def split_spans(text: str, tokenizer: ChunkTokenizer, limit: int) -> list[tuple[int, int]]:
    """Consecutive (start, end) spans covering text, each at most limit tokens."""
    if isinstance(tokenizer, FileTokenizer):
        return tokenizer.spans(text, limit)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        window = truncate_to_tokens(text[start:], tokenizer, limit) or text[start : start + 1]
        spans.append((start, start + len(window)))
        start += len(window)
    return spans or [(0, len(text))]


def _shifted(parts: Iterable[VectorPart], offset: int) -> list[VectorPart]:
    return [VectorPart(offset + part.start, offset + part.end, part.vector) for part in parts]


def embed_fitting(
    client: EmbedClient, texts: Sequence[str], kind: EmbedKind
) -> list[list[VectorPart]]:
    """Embed texts as parts; on a context overflow retry one by one, splitting an overlong text."""
    try:
        vectors = client.embed(texts, kind=kind)
    except ModelContextError as exc:
        if len(texts) > 1:
            return [parts for text in texts for parts in embed_fitting(client, [text], kind)]
        text = texts[0]
        if len(text) < 2:
            raise
        size = math.ceil(len(text) / max(2, math.ceil(1 / exc.fit_ratio())))
        return [
            [
                part
                for start in range(0, len(text), size)
                for parts in embed_fitting(client, [text[start : start + size]], kind)
                for part in _shifted(parts, start)
            ]
        ]
    return [
        [VectorPart(0, len(text), vector)] for text, vector in zip(texts, vectors, strict=False)
    ]


def oversize(
    inputs: Sequence[VectorInput], tokenizer: ChunkTokenizer, budget: int
) -> tuple[VectorInput, ...]:
    """Inputs over budget rekeyed, so vectors truncated from them before parts existed recompute."""
    return tuple(
        replace(item, key=item.key + OVERSIZE)
        if len(item.text) > budget and tokenizer.count(item.text) > budget
        else item
        for item in inputs
    )


def code_inputs(store: Store) -> tuple[VectorInput, ...]:
    """Every chunk's served text keyed by its content_hash."""
    rows = store.query(
        "SELECT c.id, c.content_hash, c.context_header,"
        " (SELECT body FROM fts_code WHERE rowid = c.id) FROM chunks c ORDER BY c.id"
    )
    return tuple(
        VectorInput(
            int(str(row[0])),
            str(row[1]),
            f"{row[2]}\n{row[3]}" if row[2] else str(row[3] or ""),
        )
        for row in rows
    )


def annot_inputs(store: Store, config: Config) -> tuple[VectorInput, ...]:
    """Annotation purpose, tags and signatures of annotated chunks keyed by a hash of that text."""
    annotate = config.models.annotate
    rows = store.query(
        "SELECT a.chunk_id, a.purpose,"
        " (SELECT group_concat(t.value, ' ') FROM json_each(a.tags) t),"
        " (SELECT group_concat(s.signature, char(10)) FROM chunk_symbols cs"
        "  JOIN symbols s ON s.id = cs.symbol_id WHERE cs.chunk_id = a.chunk_id)"
        " FROM annotations a WHERE a.model_id = ? AND a.prompt_version = ? ORDER BY a.chunk_id",
        (annotate.model, annotate.prompt_version),
    )
    inputs: list[VectorInput] = []
    for row in rows:
        text = f"{row[1]}\n{row[2] or ''}\n{row[3] or ''}"
        inputs.append(
            VectorInput(int(str(row[0])), hashlib.sha256(text.encode()).hexdigest(), text)
        )
    return tuple(inputs)


def _inputs(store: Store, config: Config, kind: SpaceKind) -> tuple[VectorInput, ...]:
    return code_inputs(store) if kind == CODE else annot_inputs(store, config)


@dataclass(slots=True)
class _Tally:
    embedded: int = 0
    cached: int = 0
    degraded: list[DegradedEntry] = field(default_factory=list[DegradedEntry])


@dataclass(frozen=True, slots=True)
class _Spec:
    kind: SpaceKind
    model_id: str
    dim: int


type Parts = tuple[VectorPart, ...]


@dataclass(frozen=True, slots=True)
class _Computed:
    parts: dict[str, Parts] = field(default_factory=dict[str, Parts])
    degraded: DegradedEntry | None = None


def call_groups(texts: Sequence[str], tokenizer: ChunkTokenizer, limit: int) -> list[list[str]]:
    """Consecutive texts grouped so each model call carries at most limit tokens, or one text."""
    groups: list[list[str]] = []
    held = limit
    for text in texts:
        tokens = tokenizer.count(text)
        if not groups or held + tokens > limit:
            groups.append([])
            held = 0
        groups[-1].append(text)
        held += tokens
    return groups


def _request(client: EmbedClient, texts: Sequence[str]) -> Callable[[], list[list[VectorPart]]]:
    return lambda: embed_fitting(client, texts, EmbedKind.DOC)


def _compute(
    slots: ModelSlots,
    texts: Sequence[str],
    keys: Sequence[str],
    spec: _Spec,
    tokenizer: ChunkTokenizer,
    budget: int,
) -> _Computed:
    spans = [split_spans(text, tokenizer, budget) for text in texts]
    flat = [
        text[start:end] for text, group in zip(texts, spans, strict=True) for start, end in group
    ]
    embedded: list[list[VectorPart]] = []
    for group in call_groups(flat, tokenizer, budget * CALL_PARTS):
        outcome = slots.call(
            Slot.EMBED,
            _request(slots.clients.embed, group),
            stage=STAGE,
            budget=slots.embed_budget(EmbedKind.DOC),
        )
        if isinstance(outcome, Skipped):
            return _Computed(degraded=outcome.degraded)
        embedded.extend(outcome.value)
    if len(embedded) != len(flat) or any(
        not parts or any(len(part.vector) != spec.dim for part in parts) for parts in embedded
    ):
        return _Computed(degraded=DegradedEntry(stage=STAGE, reason="embed_bad_vectors"))
    cursor = iter(embedded)
    computed: dict[str, Parts] = {}
    for key, group in zip(keys, spans, strict=True):
        computed[key] = tuple(part for start, _ in group for part in _shifted(next(cursor), start))
    return _Computed(computed)


@dataclass(frozen=True, slots=True)
class _Flight:
    batch: Sequence[VectorInput]
    found: Mapping[str, Parts]
    future: Future[_Computed] | None


@dataclass(slots=True)
class _Sink:
    """Buffers vectors and writes them to the project store one transaction per WRITE_ROWS."""

    store: Store
    spec: _Spec
    space: Space | None
    rows: list[tuple[VectorInput, Parts]] = field(default_factory=list[tuple[VectorInput, Parts]])

    def add(self, rows: Iterable[tuple[VectorInput, Parts]]) -> None:
        self.rows.extend(rows)
        if len(self.rows) >= WRITE_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.space = self.space or ensure_space(
            self.store, self.spec.kind, self.spec.model_id, self.spec.dim
        )
        with self.store.writer_lock():
            write_vectors(self.store, self.space, self.rows)
        self.rows.clear()


def _settle(flight: _Flight, sink: _Sink, cache: DerivedCache, tally: _Tally) -> None:
    merged = dict(flight.found)
    computed = flight.future.result() if flight.future is not None else _Computed()
    if computed.degraded is not None:
        tally.degraded.append(computed.degraded)
    elif computed.parts:
        spec = sink.spec
        cache.put_parts(
            {
                key: [(part.start, part.end, part.vector) for part in parts]
                for key, parts in computed.parts.items()
            },
            spec.kind,
            spec.model_id,
            spec.dim,
        )
        tally.embedded += len(computed.parts)
        merged.update(computed.parts)
    sink.add((item, merged[item.key]) for item in flight.batch if item.key in merged)


def _cached(cache: DerivedCache, batch: Sequence[VectorInput], spec: _Spec) -> dict[str, Parts]:
    texts = {item.key: item.text for item in batch}
    found = cache.get_parts(list(texts), spec.kind, spec.model_id, spec.dim)
    return {
        key: tuple(
            VectorPart(start, len(texts[key]) if end == WHOLE else end, vector)
            for start, end, vector in parts
        )
        for key, parts in found.items()
    }


def _embed_work(
    work: Sequence[VectorInput],
    sink: _Sink,
    slots: ModelSlots,
    cache: DerivedCache,
    tally: _Tally,
    pace: _Pace,
    on_progress: ProgressSink,
) -> None:
    """Embed work in batches, pace.parallel at once and QUEUE_DEPTH times that queued, settled in order."""
    spec = sink.spec
    stage = f"{STAGE} {spec.kind}"
    done = 0

    def settle(flight: _Flight) -> None:
        nonlocal done
        _settle(flight, sink, cache, tally)
        done += len(flight.batch)
        on_progress(Progress(stage, done, len(work)))

    on_progress(Progress(stage, done, len(work)))
    with ThreadPoolExecutor(pace.parallel, thread_name_prefix=STAGE) as pool:
        in_flight: deque[_Flight] = deque()
        for start in range(0, len(work), pace.batch_size):
            if tally.degraded:
                break
            batch = work[start : start + pace.batch_size]
            keys = [item.key for item in batch]
            found = _cached(cache, batch, spec)
            tally.cached += len(found)
            missing = [key for key in dict.fromkeys(keys) if key not in found]
            texts = {item.key: item.text for item in batch}
            future = (
                pool.submit(
                    _compute,
                    slots,
                    [texts[key] for key in missing],
                    missing,
                    spec,
                    pace.tokenizer,
                    pace.budget,
                )
                if missing
                else None
            )
            in_flight.append(_Flight(batch, found, future))
            if len(in_flight) >= pace.parallel * QUEUE_DEPTH:
                settle(in_flight.popleft())
        while in_flight:
            settle(in_flight.popleft())
    sink.flush()


@dataclass(frozen=True, slots=True)
class _Pace:
    tokenizer: ChunkTokenizer
    budget: int
    batch_size: int
    parallel: int


def embed_project(
    store: Store,
    slots: ModelSlots,
    cache: DerivedCache,
    config: Config,
    *,
    tokenizer: ChunkTokenizer | None = None,
    batch_size: int = DEFAULT_BATCH,
    max_chunks: int | None = None,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    on_progress: ProgressSink = discard,
) -> EmbedReport:
    """Embed chunks lacking a current vector; doubles as the background space rebuild job.

    A new model or dim gets its own building space, filled here; once nothing is pending it
    goes live and older spaces of its kind are evicted. A skipped model call stops the pass and
    is reported, never raised; the scheduler calls again, bounding each slice with max_chunks.
    """
    if not vec_available(store):
        return EmbedReport(degraded=(DegradedEntry(stage=STAGE, reason="vec_unavailable"),))
    embed_config = config.models.embed
    model_id = embed_model_id(embed_config)
    counter = tokenizer or default_tokenizer()
    tally = _Tally()
    pace = _Pace(
        counter,
        max(max_input_tokens - SERVER_RESERVED_TOKENS - counter.count(embed_config.doc_prefix), 1),
        batch_size,
        max(embed_config.parallel, 1),
    )
    remaining = 0
    live: list[str] = []
    for kind in KINDS:
        inputs = oversize(_inputs(store, config, kind), counter, pace.budget)
        space = find_space(store, kind, model_id, embed_config.dim)
        if kind == ANNOT and not inputs and space is None:
            continue
        spec = _Spec(kind, model_id, embed_config.dim)
        todo = pending_inputs(store, space, inputs) if space else inputs
        work = todo if max_chunks is None else todo[: max(max_chunks - tally.embedded, 0)]
        sink = _Sink(store, spec, space)
        _embed_work(work, sink, slots, cache, tally, pace, on_progress)
        space = sink.space
        left = len(pending_inputs(store, space, inputs)) if space else len(inputs)
        remaining += left
        if left == 0:
            space = space or ensure_space(store, kind, model_id, embed_config.dim)
            with store.writer_lock():
                purge_orphans(store, space)
                if space.status != "live":
                    space = mark_live(store, space)
        if space is not None and space.status == "live":
            live.append(space.name)
    return EmbedReport(tally.embedded, tally.cached, remaining, tuple(live), tuple(tally.degraded))
