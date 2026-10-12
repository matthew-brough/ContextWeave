from dataclasses import replace

from contextweave.compact.sources import SourceCache
from contextweave.core.refs import ByteAnchor
from contextweave.retrieve.candidates import Candidate

MAX_SEAM_NEWLINES = 1


def rank_key(cand: Candidate) -> tuple[float, str, int, int]:
    return (-cand.scores.final, *cand.order_key)


def dedupe_identity(sources: SourceCache, cands: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    """Keep the best-ranked candidate of each (path, byte range)."""
    kept: dict[tuple[str, int, int], Candidate] = {}
    for cand in sorted(cands, key=rank_key):
        kept.setdefault(cand.order_key, cand)
    return tuple(kept.values())


def is_seam(gap: bytes) -> bool:
    return not gap.strip() and gap.count(b"\n") <= MAX_SEAM_NEWLINES


def combine(run: list[Candidate]) -> Candidate:
    """One candidate over the union of a run; metadata of its best-ranked member."""
    if len(run) == 1:
        return run[0]
    best = min(run, key=rank_key)
    start = min(cand.start_byte for cand in run)
    end = max(cand.end_byte for cand in run)
    same = len({cand.qualified_name for cand in run}) == 1
    note = f"merged {len(run)} adjacent ranges"
    return replace(
        best,
        start_byte=start,
        end_byte=end,
        anchor=ByteAnchor(start, end),
        qualified_name=best.qualified_name if same else None,
        kind=best.kind if same else "group",
        symbols=tuple(dict.fromkeys(name for cand in run for name in cand.symbols)),
        critical=any(cand.critical for cand in run),
        edges=tuple(dict.fromkeys(edge for cand in run for edge in cand.edges)),
        fused=max(cand.fused for cand in run),
        why=f"{best.why}; {note}" if best.why else note,
    )


def merge_file(source: bytes, cands: list[Candidate]) -> list[Candidate]:
    merged: list[Candidate] = []
    run: list[Candidate] = []
    run_end = 0
    for cand in sorted(cands, key=lambda item: item.order_key[1:]):
        if run and (cand.start_byte <= run_end or is_seam(source[run_end : cand.start_byte])):
            run.append(cand)
            run_end = max(run_end, cand.end_byte)
            continue
        if run:
            merged.append(combine(run))
        run = [cand]
        run_end = cand.end_byte
    if run:
        merged.append(combine(run))
    return merged


def merge_adjacent(sources: SourceCache, cands: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    """Merge overlapping or touching ranges of one file into a single byte-range candidate."""
    by_path: dict[str, list[Candidate]] = {}
    for cand in cands:
        by_path.setdefault(cand.path, []).append(cand)
    out: list[Candidate] = []
    for path, group in by_path.items():
        source = sources.read(path)
        out.extend(group if source is None else merge_file(source, group))
    return tuple(sorted(out, key=rank_key))
