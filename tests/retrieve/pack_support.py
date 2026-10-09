from contextweave.core.refs import ByteAnchor, SymbolAnchor
from contextweave.retrieve.candidates import Candidate, Ranks, Scores


def make_cand(
    chunk_id: int,
    path: str = "a.py",
    start: int = 0,
    end: int = 10,
    *,
    name: str | None = None,
    ranks: Ranks | None = None,
    scores: Scores | None = None,
    fused: float = 0.0,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        path=path,
        start_byte=start,
        end_byte=end,
        anchor=SymbolAnchor(name) if name else ByteAnchor(start, end),
        profile="symbols",
        group="core",
        qualified_name=name,
        kind="function" if name else "block",
        ranks=ranks or Ranks(),
        scores=scores or Scores(),
        fused=fused,
    )
