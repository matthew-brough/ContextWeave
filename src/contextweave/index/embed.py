import hashlib
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from contextweave.core.config import Config, EmbedConfig
from contextweave.core.errors import DegradedEntry
from contextweave.index.chunk_tokens import ChunkTokenizer, default_tokenizer
from contextweave.index.derived import DerivedCache
from contextweave.index.store import Store
from contextweave.index.vectors import (
    ANNOT,
    CODE,
    KINDS,
    Space,
    SpaceKind,
    VectorInput,
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
WRITE_ROWS = 256


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


def embed_fitting(client: EmbedClient, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
    """Embed texts; on a context overflow retry one by one, re-cutting by the server's count."""
    try:
        return client.embed(texts, kind=kind)
    except ModelContextError as exc:
        if len(texts) == 1:
            return client.embed([texts[0][: int(len(texts[0]) * exc.fit_ratio())]], kind=kind)
    return [vector for text in texts for vector in embed_fitting(client, [text], kind)]


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


@dataclass(frozen=True, slots=True)
class _Computed:
    vectors: dict[str, Sequence[float]] = field(default_factory=dict[str, Sequence[float]])
    degraded: DegradedEntry | None = None


def _compute(
    slots: ModelSlots,
    texts: Sequence[str],
    keys: Sequence[str],
    spec: _Spec,
    tokenizer: ChunkTokenizer,
    budget: int,
) -> _Computed:
    inputs = [truncate_to_tokens(text, tokenizer, budget) for text in texts]
    outcome = slots.call(
        Slot.EMBED,
        lambda: embed_fitting(slots.clients.embed, inputs, EmbedKind.DOC),
        stage=STAGE,
        budget=slots.embed_budget(EmbedKind.DOC),
    )
    if isinstance(outcome, Skipped):
        return _Computed(degraded=outcome.degraded)
    vectors = outcome.value
    if len(vectors) != len(keys) or any(len(vector) != spec.dim for vector in vectors):
        return _Computed(degraded=DegradedEntry(stage=STAGE, reason="embed_bad_vectors"))
    return _Computed(dict(zip(keys, vectors, strict=True)))


@dataclass(frozen=True, slots=True)
class _Flight:
    batch: Sequence[VectorInput]
    found: Mapping[str, Sequence[float]]
    future: Future[_Computed] | None


@dataclass(slots=True)
class _Sink:
    """Buffers vectors and writes them to the project store one transaction per WRITE_ROWS."""

    store: Store
    spec: _Spec
    space: Space | None
    rows: list[tuple[VectorInput, Sequence[float]]] = field(
        default_factory=list[tuple[VectorInput, Sequence[float]]]
    )

    def add(self, rows: Iterable[tuple[VectorInput, Sequence[float]]]) -> None:
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
    elif computed.vectors:
        spec = sink.spec
        cache.put_vectors(computed.vectors, spec.kind, spec.model_id, spec.dim)
        tally.embedded += len(computed.vectors)
        merged.update(computed.vectors)
    sink.add((item, merged[item.key]) for item in flight.batch if item.key in merged)


def _embed_work(
    work: Sequence[VectorInput],
    sink: _Sink,
    slots: ModelSlots,
    cache: DerivedCache,
    tally: _Tally,
    pace: _Pace,
) -> None:
    """Embed work in batches, keeping up to pace.parallel batches in flight, settled in order."""
    spec = sink.spec
    with ThreadPoolExecutor(pace.parallel, thread_name_prefix=STAGE) as pool:
        in_flight: deque[_Flight] = deque()
        for start in range(0, len(work), pace.batch_size):
            if tally.degraded:
                break
            batch = work[start : start + pace.batch_size]
            keys = [item.key for item in batch]
            found = cache.get_vectors(keys, spec.kind, spec.model_id, spec.dim)
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
            if len(in_flight) >= pace.parallel:
                _settle(in_flight.popleft(), sink, cache, tally)
        while in_flight:
            _settle(in_flight.popleft(), sink, cache, tally)
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
        max(max_input_tokens - counter.count(embed_config.doc_prefix), 1),
        batch_size,
        max(embed_config.parallel, 1),
    )
    remaining = 0
    live: list[str] = []
    for kind in KINDS:
        inputs = _inputs(store, config, kind)
        space = find_space(store, kind, model_id, embed_config.dim)
        if kind == ANNOT and not inputs and space is None:
            continue
        spec = _Spec(kind, model_id, embed_config.dim)
        todo = pending_inputs(store, space, inputs) if space else inputs
        work = todo if max_chunks is None else todo[: max(max_chunks - tally.embedded, 0)]
        sink = _Sink(store, spec, space)
        _embed_work(work, sink, slots, cache, tally, pace)
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
