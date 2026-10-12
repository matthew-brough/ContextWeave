import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from contextweave.core.budget import bytes35
from contextweave.eval.adapters import AdapterContext, Served, degraded
from contextweave.eval.keywords import keywords
from contextweave.lang.extract_symbols import extract_symbols
from contextweave.lang.parse import parse_bytes
from contextweave.lang.registry import spec_for_path

RG_COMMAND = "rg"
HEAD_LINES = 50
OUTPUT_CAP_CHARS = 30000
SED_FILES = 3
SED_BEFORE = 20
SED_AFTER = 60
CODE_PROFILES = frozenset({"symbols", "defs"})


@dataclass(slots=True)
class _FileHits:
    """Matching lines of one file and which keywords they contain."""

    lines: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    words: set[str] = field(default_factory=set[str])


def _word_count(text: str, words: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(word in lowered for word in words)


def parse_rg(output: str, words: Sequence[str]) -> dict[str, _FileHits]:
    """Hits per path from `rg --null -n` output."""
    hits: dict[str, _FileHits] = {}
    for raw in output.splitlines():
        path, _, rest = raw.partition("\0")
        number, _, text = rest.partition(":")
        if not number.isdigit():
            continue
        entry = hits.setdefault(path, _FileHits())
        entry.lines.append((int(number), text))
        lowered = text.lower()
        entry.words.update(word for word in words if word in lowered)
    return hits


def rank_files(hits: dict[str, _FileHits]) -> list[str]:
    """Paths by distinct keywords matched, then hit count, then path."""
    return sorted(hits, key=lambda path: (-len(hits[path].words), -len(hits[path].lines), path))


def sed_window(entry: _FileHits, words: Sequence[str]) -> tuple[int, int]:
    """The 1-based inclusive line range read around the hit naming the most keywords."""
    anchor = max(entry.lines, key=lambda hit: (_word_count(hit[1], words), -hit[0]))[0]
    return max(1, anchor - SED_BEFORE), anchor + SED_AFTER


def is_code(path: str) -> bool:
    """Whether path is source an agent would sed open, not docs or data."""
    spec = spec_for_path(path)
    return spec is not None and spec.profile in CODE_PROFILES


def _symbols_in(path: str, source: bytes, start: int, end: int) -> tuple[str, ...]:
    parsed = parse_bytes(path, source).parsed
    if parsed is None:
        return ()
    return tuple(
        symbol.qualified_name
        for symbol in extract_symbols(parsed).symbols
        if symbol.start_line <= end and symbol.end_line >= start
    )


class GrepAdapter:
    """Baseline of an agent's default search: rg the query keywords, head the output, sed the best code hits.

    Each command's output is capped like the agent's shell tool truncates it.
    """

    def __init__(self, context: AdapterContext) -> None:
        self._root = context.root

    def _serve(self, query: str) -> Served:
        words = keywords(query)
        if not words:
            return Served(error="no keywords")
        started = time.perf_counter()
        pattern = [arg for word in words for arg in ("-e", word)]
        done = subprocess.run(
            [RG_COMMAND, "--null", "-n", "-i", "--no-heading", "--color", "never", *pattern, "."],
            cwd=self._root,
            capture_output=True,
            check=False,
        )
        hits = parse_rg(done.stdout.decode("utf-8", errors="replace"), words)
        ranked = rank_files(hits)
        shown = [f"{path}:{number}:{text}" for path in ranked for number, text in hits[path].lines][
            :HEAD_LINES
        ]
        tokens = bytes35("\n".join(shown)[:OUTPUT_CAP_CHARS])
        items: dict[str, None] = {}
        for path in [path for path in ranked if is_code(path)][:SED_FILES]:
            start, end = sed_window(hits[path], words)
            relative = path.removeprefix("./")
            try:
                source = (self._root / relative).read_bytes()
            except OSError:
                continue
            lines = source.decode("utf-8", errors="replace").splitlines()
            tokens += bytes35("\n".join(lines[start - 1 : end])[:OUTPUT_CAP_CHARS])
            items.update(dict.fromkeys(_symbols_in(relative, source, start, end)))
        elapsed = (time.perf_counter() - started) * 1000
        return Served(tuple(items), tokens, elapsed)

    def serve_all(self, queries: Sequence[str]) -> tuple[Served, ...]:
        if shutil.which(RG_COMMAND) is None:
            return degraded(len(queries), f"command not found: {RG_COMMAND}")
        return tuple(self._serve(query) for query in queries)
