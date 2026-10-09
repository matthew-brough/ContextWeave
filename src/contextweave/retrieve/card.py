from collections.abc import Sequence

from contextweave.compact.sources import SourceCache
from contextweave.index.store import Store
from contextweave.retrieve.candidates import Candidate

BYTES_PER_TOKEN = 3.5
ANNOTATION_BATCH = 500


def cached_purposes(
    store: Store, model: str, prompt_version: int, chunk_ids: Sequence[int]
) -> dict[int, str]:
    """Annotation purposes cached for the current chunk hash, model and prompt version."""
    found: dict[int, str] = {}
    for offset in range(0, len(chunk_ids), ANNOTATION_BATCH):
        batch = chunk_ids[offset : offset + ANNOTATION_BATCH]
        marks = ", ".join("?" for _ in batch)
        rows = store.query(
            "SELECT a.chunk_id, a.purpose FROM annotations a"
            " JOIN chunks c ON c.id = a.chunk_id AND c.content_hash = a.content_hash"
            f" WHERE a.model_id = ? AND a.prompt_version = ? AND a.chunk_id IN ({marks})",
            [model, prompt_version, *batch],
        )
        for row in rows:
            chunk_id, purpose = row[0], row[1]
            if isinstance(chunk_id, int) and isinstance(purpose, str) and purpose.strip():
                found[chunk_id] = purpose.strip()
    return found


def truncate_body(body: bytes, card_tokens: int) -> str:
    """The body cut to card_tokens at the byte estimator; a cut mid-character is dropped."""
    limit = int(card_tokens * BYTES_PER_TOKEN)
    return body[:limit].decode(errors="ignore")


def card_of(
    cand: Candidate, sources: SourceCache, purposes: dict[int, str], card_tokens: int
) -> str:
    """path, symbol, cached annotation and the first card_tokens of the body."""
    source = sources.read(cand.path)
    body = b"" if source is None else source[cand.start_byte : cand.end_byte]
    lines = [f"path: {cand.path}"]
    if cand.qualified_name:
        lines.append(f"symbol: {cand.kind or 'symbol'} {cand.qualified_name}")
    purpose = purposes.get(cand.chunk_id)
    if purpose:
        lines.append(f"purpose: {purpose}")
    lines.append(truncate_body(body, card_tokens))
    return "\n".join(lines)
