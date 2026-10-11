import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from contextweave.core.config import Config, EmbedConfig
from contextweave.core.errors import DegradedEntry
from contextweave.index.chunk_tokens import ChunkTokenizer, default_tokenizer
from contextweave.index.derived import DerivedCache, cached_or_compute
from contextweave.index.store import Store
from contextweave.index.vectors import (
    ANNOT,
    CODE,
    KINDS,
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


def _embed_batch(
    slots: ModelSlots,
    cache: DerivedCache,
    spec: _Spec,
    batch: Sequence[VectorInput],
    config: EmbedConfig,
    tokenizer: ChunkTokenizer,
    max_input_tokens: int,
    tally: _Tally,
) -> Mapping[str, Sequence[float]]:
    texts = {item.key: item.text for item in batch}
    budget = max(max_input_tokens - tokenizer.count(config.doc_prefix), 1)

    def lookup(keys: Sequence[str]) -> Mapping[str, Sequence[float]]:
        found = cache.get_vectors(keys, spec.kind, spec.model_id, spec.dim)
        tally.cached += len(found)
        return found

    def compute(keys: Sequence[str]) -> Mapping[str, Sequence[float]]:
        inputs = [truncate_to_tokens(texts[key], tokenizer, budget) for key in keys]
        outcome = slots.call(
            Slot.EMBED,
            lambda: embed_fitting(slots.clients.embed, inputs, EmbedKind.DOC),
            stage=STAGE,
            budget=slots.embed_budget(EmbedKind.DOC),
        )
        if isinstance(outcome, Skipped):
            tally.degraded.append(outcome.degraded)
            return {}
        vectors = outcome.value
        if len(vectors) != len(keys) or any(len(vector) != spec.dim for vector in vectors):
            tally.degraded.append(DegradedEntry(stage=STAGE, reason="embed_bad_vectors"))
            return {}
        return dict(zip(keys, vectors, strict=True))

    def store_vector(key: str, vector: Sequence[float]) -> None:
        cache.put_vector(key, spec.kind, spec.model_id, spec.dim, vector)
        tally.embedded += 1

    return cached_or_compute([item.key for item in batch], lookup, compute, store_vector)


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
        for start in range(0, len(work), batch_size):
            batch = work[start : start + batch_size]
            vectors = _embed_batch(
                slots, cache, spec, batch, embed_config, counter, max_input_tokens, tally
            )
            ready = [(item, vectors[item.key]) for item in batch if item.key in vectors]
            if ready:
                space = space or ensure_space(store, kind, model_id, embed_config.dim)
                with store.writer_lock():
                    write_vectors(store, space, ready)
            if tally.degraded:
                break
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
