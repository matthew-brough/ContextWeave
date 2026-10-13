import hashlib
import json
import re
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from contextweave.core.config import Config
from contextweave.core.errors import DegradedEntry
from contextweave.core.types import JSON
from contextweave.index.chunk_tokens import ChunkTokenizer, default_tokenizer
from contextweave.index.derived import PART_SEP, DerivedAnnotation, DerivedCache
from contextweave.index.embed import truncate_to_tokens
from contextweave.index.progress import Progress, ProgressSink
from contextweave.index.store import Store
from contextweave.models.errors import ModelContextError, ModelParseError, ModelRequestError
from contextweave.models.prompt_store import PROMPTS_DIR, PromptError, PromptTemplate, load_prompt
from contextweave.models.providers import Slot
from contextweave.models.slots import Done, ModelSlots, Outcome

STAGE = "annotate"
PROMPT_NAME = "annotate"
ANNOTATED_PROFILES = ("symbols", "defs", "outline")
MAX_CONSECUTIVE_FAULTS = 8
QUEUE_DEPTH = 4
DEFAULT_MAX_CODE_TOKENS = 6144

_ELIGIBLE = "f.profile IN (" + ", ".join(f"'{profile}'" for profile in ANNOTATED_PROFILES) + ")"
_UNANNOTATED = (
    "NOT EXISTS (SELECT 1 FROM annotations a WHERE a.chunk_id = c.id"
    " AND a.model_id = ? AND a.prompt_version = ?)"
)


@dataclass(frozen=True, slots=True)
class AnnotateProgress:
    """Running state of one pass: chunks annotated so far of the eligible total, with an ETA."""

    done: int
    total: int
    rate: float | None
    eta_s: float | None


@dataclass(frozen=True, slots=True)
class AnnotateReport:
    """What one annotate pass did; pending > 0 or degraded entries mean it should run again.

    rate is model-annotated chunks per second measured this pass; eta_s is pending over rate.
    """

    annotated: int = 0
    cached: int = 0
    pending: int = 0
    total: int = 0
    rate: float | None = None
    eta_s: float | None = None
    degraded: tuple[DegradedEntry, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class PendingChunk:
    chunk_id: int
    content_hash: str
    kind: str
    path: str
    name: str
    code: str


@dataclass(slots=True)
class _Tally:
    annotated: int = 0
    cached: int = 0
    model_seconds: float = 0.0
    consecutive_faults: int = 0
    degraded: list[DegradedEntry] = field(default_factory=list[DegradedEntry])

    @property
    def rate(self) -> float | None:
        return (
            self.annotated / self.model_seconds if self.annotated and self.model_seconds else None
        )


@dataclass(slots=True)
class _Attempt:
    chunk_fault: bool = False


@dataclass(frozen=True, slots=True)
class _InFlight:
    chunk: PendingChunk
    attempt: _Attempt
    future: Future[Outcome[DerivedAnnotation]]


def prompt_cache_version(prompt: PromptTemplate, max_code_tokens: int) -> str:
    """Cache identity of the prompt: its version plus a digest of messages, schema and code cap."""
    payload = json.dumps(
        {
            "messages": [[message.role, message.content] for message in prompt.messages],
            "schema": prompt.schema,
            "max_code_tokens": max_code_tokens,
        },
        sort_keys=True,
    )
    return f"{prompt.version}@{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def annotation_cache_key(chunk: PendingChunk) -> str:
    """Cache identity of a chunk's prompt input: its content plus the path, kind and name shown."""
    shown = "\0".join((chunk.path, chunk.kind, chunk.name))
    return f"{chunk.content_hash}{PART_SEP}{hashlib.sha256(shown.encode()).hexdigest()[:16]}"


def annotation_text(annotation: DerivedAnnotation) -> str:
    """The annotation as indexed in the fts annotation column."""
    return " ".join((annotation.purpose, *annotation.tags))


def total_chunks(store: Store) -> int:
    """Chunks whose file profile is annotated."""
    rows = store.query(
        "SELECT count(*) FROM chunks c JOIN files f ON f.id = c.file_id WHERE " + _ELIGIBLE
    )
    return int(str(rows[0][0]))


def pending_chunks(store: Store, config: Config) -> tuple[PendingChunk, ...]:
    """Annotatable chunks lacking an annotation for the configured model and prompt version."""
    annotate = config.models.annotate
    rows = store.query(
        "SELECT c.id, c.content_hash, c.kind, f.path,"
        " (SELECT s.qualified_name FROM chunk_symbols cs JOIN symbols s ON s.id = cs.symbol_id"
        "  WHERE cs.chunk_id = c.id ORDER BY cs.rowid LIMIT 1),"
        " c.context_header, (SELECT body FROM fts_code WHERE rowid = c.id)"
        f" FROM chunks c JOIN files f ON f.id = c.file_id WHERE {_ELIGIBLE} AND {_UNANNOTATED}"
        " ORDER BY c.id",
        (annotate.model, annotate.prompt_version),
    )
    return tuple(
        PendingChunk(
            int(str(row[0])),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4] or ""),
            f"{row[5]}\n{row[6]}" if row[5] else str(row[6] or ""),
        )
        for row in rows
    )


def write_annotation(
    store: Store,
    chunk: PendingChunk,
    model_id: str,
    prompt_version: int,
    annotation: DerivedAnnotation,
) -> None:
    """Replace the chunk's annotations with this one and update its fts annotation column."""
    with store.writer_lock(), store.write_transaction():
        store.execute("DELETE FROM annotations WHERE chunk_id = ?", (chunk.chunk_id,))
        store.execute(
            "INSERT INTO annotations (chunk_id, content_hash, model_id, prompt_version,"
            " purpose, tags) VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk.chunk_id,
                chunk.content_hash,
                model_id,
                prompt_version,
                annotation.purpose,
                json.dumps(list(annotation.tags)),
            ),
        )
        store.execute(
            "UPDATE fts_code SET annotation = ? WHERE rowid = ?",
            (annotation_text(annotation), chunk.chunk_id),
        )


def _spec(schema: Mapping[str, JSON], key: str) -> Mapping[str, JSON]:
    properties = schema.get("properties")
    spec = properties.get(key) if isinstance(properties, dict) else None
    return spec if isinstance(spec, dict) else {}


def _limit(spec: Mapping[str, JSON], key: str) -> int | None:
    value = spec.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def validate_annotation(
    output: Mapping[str, JSON], schema: Mapping[str, JSON]
) -> DerivedAnnotation:
    """Check model output against the prompt schema; ModelParseError when it does not conform."""
    properties = schema.get("properties")
    if schema.get("additionalProperties") is False and isinstance(properties, dict):
        extra = sorted(set(output) - set(properties))
        if extra:
            raise ModelParseError(f"annotation has unexpected keys: {', '.join(extra)}")
    purpose = output.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        raise ModelParseError("annotation purpose is not a non-empty string")
    purpose = " ".join(purpose.split())
    purpose_limit = _limit(_spec(schema, "purpose"), "maxLength")
    if purpose_limit is not None and len(purpose) > purpose_limit:
        raise ModelParseError(f"annotation purpose exceeds {purpose_limit} characters")
    raw_tags = output.get("tags")
    if not isinstance(raw_tags, list):
        raise ModelParseError("annotation tags is not an array")
    tags_spec = _spec(schema, "tags")
    items = tags_spec.get("items")
    pattern = items.get("pattern") if isinstance(items, dict) else None
    tags: list[str] = []
    for raw in raw_tags:
        if not isinstance(raw, str):
            raise ModelParseError("annotation tag is not a string")
        tag = raw.strip().lower()
        if isinstance(pattern, str) and re.fullmatch(pattern, tag) is None:
            raise ModelParseError(f"annotation tag '{raw}' does not match {pattern}")
        if tag and tag not in tags:
            tags.append(tag)
    tags_limit = _limit(tags_spec, "maxItems")
    if tags_limit is not None and len(tags) > tags_limit:
        raise ModelParseError(f"annotation has more than {tags_limit} tags")
    return DerivedAnnotation(purpose, tuple(tags))


def _generate(
    slots: ModelSlots, prompt: PromptTemplate, chunk: PendingChunk, code: str, attempt: _Attempt
) -> Outcome[DerivedAnnotation]:
    def ask(text: str) -> DerivedAnnotation:
        messages = prompt.render(path=chunk.path, kind=chunk.kind, name=chunk.name, code=text)
        output = slots.clients.annotate.annotate(messages, prompt.schema)
        return validate_annotation(output, prompt.schema)

    def call() -> DerivedAnnotation:
        try:
            try:
                return ask(code)
            except ModelContextError as exc:
                return ask(code[: int(len(code) * exc.fit_ratio())])
        except ModelParseError, ModelRequestError:
            attempt.chunk_fault = True
            raise

    return slots.call(Slot.ANNOTATE, call, stage=STAGE)


def _eta(left: int, rate: float | None) -> float | None:
    return left / rate if rate else None


def annotate_project(
    store: Store,
    slots: ModelSlots,
    cache: DerivedCache,
    config: Config,
    *,
    max_chunks: int | None = None,
    on_progress: Callable[[AnnotateProgress], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    prompts_dir: Path = PROMPTS_DIR,
    tokenizer: ChunkTokenizer | None = None,
    max_code_tokens: int = DEFAULT_MAX_CODE_TOKENS,
) -> AnnotateReport:
    """Annotate chunks lacking an annotation for the current model and prompt version.

    Cache first, so unchanged content costs no model call; max_chunks bounds model calls per
    slice. Up to models.annotate.parallel calls run at once, QUEUE_DEPTH times that queued so a slow
    call does not idle the other slots; results are written in chunk order on this thread. Code past max_code_tokens is cut, and cut again to the server's own count
    when it still overflows the context. A model that is down stops the pass; a malformed answer
    or a rejected request leaves that chunk pending for the next pass. Both are reported as
    degraded, never raised.
    """
    annotate = config.models.annotate
    total = total_chunks(store)
    todo = pending_chunks(store, config)
    if not todo:
        return AnnotateReport(total=total)
    try:
        prompt = load_prompt(PROMPT_NAME, annotate.prompt_version, prompts_dir)
    except PromptError as exc:
        entry = DegradedEntry(stage=STAGE, reason=f"annotate_prompt: {exc}")
        return AnnotateReport(pending=len(todo), total=total, degraded=(entry,))
    version = prompt_cache_version(prompt, max_code_tokens)
    counter = tokenizer or default_tokenizer()
    tally = _Tally()
    left = len(todo)
    submitted = 0
    started: float | None = None
    in_flight: deque[_InFlight] = deque()
    if on_progress is not None:
        on_progress(AnnotateProgress(total - left, total, None, None))

    def record(chunk: PendingChunk, found: DerivedAnnotation) -> None:
        nonlocal left
        write_annotation(store, chunk, annotate.model, annotate.prompt_version, found)
        left -= 1
        if on_progress is not None:
            on_progress(AnnotateProgress(total - left, total, tally.rate, _eta(left, tally.rate)))

    def settle(flight: _InFlight, *, counting: bool) -> bool:
        if flight.future.cancelled():
            return True
        outcome = flight.future.result()
        if isinstance(outcome, Done):
            tally.consecutive_faults = 0
            tally.annotated += 1
            if started is not None:
                tally.model_seconds = clock() - started
            cache.put_annotation(
                annotation_cache_key(flight.chunk), annotate.model, version, outcome.value
            )
            record(flight.chunk, outcome.value)
            return True
        if not counting:
            return True
        tally.degraded.append(outcome.degraded)
        if not flight.attempt.chunk_fault:
            return False
        tally.consecutive_faults += 1
        return tally.consecutive_faults < MAX_CONSECUTIVE_FAULTS

    with ThreadPoolExecutor(max(annotate.parallel, 1), thread_name_prefix=STAGE) as pool:
        going = True
        for chunk in todo:
            found = cache.get_annotation(annotation_cache_key(chunk), annotate.model, version)
            if found is not None:
                tally.cached += 1
                record(chunk, found)
                continue
            if max_chunks is not None and submitted >= max_chunks:
                continue
            if started is None:
                started = clock()
            code = truncate_to_tokens(chunk.code, counter, max_code_tokens)
            attempt = _Attempt()
            future = pool.submit(_generate, slots, prompt, chunk, code, attempt)
            in_flight.append(_InFlight(chunk, attempt, future))
            submitted += 1
            if len(in_flight) >= max(annotate.parallel, 1) * QUEUE_DEPTH:
                going = settle(in_flight.popleft(), counting=True)
                if not going:
                    break
        if not going:
            for flight in in_flight:
                flight.future.cancel()
        while in_flight:
            going = settle(in_flight.popleft(), counting=going) and going
    pending = len(pending_chunks(store, config))
    return AnnotateReport(
        tally.annotated,
        tally.cached,
        pending,
        total,
        tally.rate,
        _eta(pending, tally.rate),
        tuple(tally.degraded),
    )


def annotate_progress(sink: ProgressSink) -> Callable[[AnnotateProgress], None]:
    """Forward annotate progress to a generic stage progress sink."""

    def forward(progress: AnnotateProgress) -> None:
        sink(Progress(STAGE, progress.done, progress.total, progress.eta_s))

    return forward


def format_eta(seconds: float) -> str:
    """Seconds as a short h/m/s duration."""
    whole = max(round(seconds), 0)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def annotate_lines(report: AnnotateReport) -> list[str]:
    """Summary lines of an annotate pass, with the ETA while chunks remain."""
    line = f"annotated {report.annotated} chunks, {report.cached} cached, {report.pending} pending"
    if report.pending and report.eta_s is not None:
        line += f", eta {format_eta(report.eta_s)}"
    lines = [line]
    if report.pending:
        lines.append("rerun `cw index` to finish annotating")
    lines.extend(f"degraded {entry['stage']}: {entry['reason']}" for entry in report.degraded)
    return lines
