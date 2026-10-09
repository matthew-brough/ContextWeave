from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from contextweave.core.budget import Ceiling
from contextweave.core.errors import Envelope, OmittedEntry, OverflowEntry, cap_omitted
from contextweave.core.project import ProjectEcho, ResolvedProject
from contextweave.server.context import PackCache, ToolContext
from contextweave.server.snapshot import Snapshot

type Omitted = list[OmittedEntry | OverflowEntry]
type Trimmed[P] = tuple[P, list[OmittedEntry], int]
type Assemble[I, P] = Callable[[Sequence[I], Omitted, str | None], P]


@dataclass(frozen=True, slots=True)
class Call[P: Mapping[str, object]]:
    """One tool invocation: its context, resolved project, fresh snapshot and budget."""

    ctx: ToolContext
    project: ResolvedProject
    snap: Snapshot
    env: Envelope
    ceiling: Ceiling
    measure: Callable[[P], int]

    @property
    def echo(self) -> ProjectEcho:
        return self.project.echo()

    def fits(self, payload: P) -> bool:
        return self.ceiling.fits(self.measure(payload))


@dataclass(frozen=True, slots=True)
class Clip[I]:
    """How to shorten one item's body when even a single item overflows."""

    size: Callable[[I], int]
    cut: Callable[[I, int], I]


def fit_items[I, P: Mapping[str, object]](
    call: Call[P],
    items: Sequence[I],
    assemble: Assemble[I, P],
    omit: Callable[[I], OmittedEntry],
    hint: str,
    clip: Clip[I] | None = None,
) -> P:
    """Largest prefix of items that fits the ceiling; the rest become omitted entries (V1, V6)."""
    full = assemble(items, [], None)
    if call.fits(full):
        return full
    omitted_max = call.ctx.config.budget.omitted_max
    key = call.project.key

    def trimmed(kept: Sequence[I], rest: Sequence[I]) -> Trimmed[P]:
        dropped = [omit(item) for item in rest]
        pack_id = PackCache.pack_id(dropped)
        cap = omitted_max
        while True:
            capped = cap_omitted(dropped, omitted_max=cap, key=key, pack_id=pack_id)
            payload = assemble(kept, capped, hint)
            if cap <= 1 or call.fits(payload):
                return payload, dropped, cap
            cap = max(cap // 2, 1)

    best: Trimmed[P] | None = None
    low, high = 0, len(items) - 1
    while low <= high:
        mid = (low + high) // 2
        candidate = trimmed(items[:mid], items[mid:])
        if call.fits(candidate[0]):
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    if best is None and clip is not None and items:
        best = _clipped(call, items, trimmed, clip)
    if best is None:
        best = trimmed([], items)
    payload, dropped, cap = best
    if len(dropped) > cap:
        call.ctx.packs.put(key, PackCache.pack_id(dropped), dropped)
    return payload


def _clipped[I, P: Mapping[str, object]](
    call: Call[P],
    items: Sequence[I],
    trimmed: Callable[[Sequence[I], Sequence[I]], Trimmed[P]],
    clip: Clip[I],
) -> Trimmed[P]:
    head, rest = items[0], items[1:]
    result = trimmed([clip.cut(head, 0)], rest)
    low, high = 1, clip.size(head)
    while low <= high:
        mid = (low + high) // 2
        candidate = trimmed([clip.cut(head, mid)], rest)
        if call.fits(candidate[0]):
            result = candidate
            low = mid + 1
        else:
            high = mid - 1
    return result


def fit_text[P: Mapping[str, object]](
    call: Call[P], text: str, assemble: Callable[[str, str | None], P], hint: str
) -> P:
    """Longest prefix of text that fits the ceiling."""
    full = assemble(text, None)
    if call.fits(full):
        return full
    best = assemble("", hint)
    low, high = 1, len(text)
    while low <= high:
        mid = (low + high) // 2
        candidate = assemble(text[:mid], hint)
        if call.fits(candidate):
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best
