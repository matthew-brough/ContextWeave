from dataclasses import replace

from contextweave.compact.sources import SourceCache
from contextweave.retrieve.pack import PackItem

COLLAPSED_PREFIX = "imports as in "

type StatementKey = tuple[str, str]


def directory_of(path: str) -> str:
    return path.rpartition("/")[0]


def statement_keys(sources: SourceCache, item: PackItem) -> tuple[StatementKey, ...] | None:
    """Normalised import statements when the item is nothing but imports, else None."""
    cand = item.cand
    source = sources.read(cand.path)
    if item.rep != "source" or source is None:
        return None
    inside = [
        statement
        for statement in sources.imports(cand.path)
        if cand.start_byte <= statement.start_byte and statement.end_byte <= cand.end_byte
    ]
    if not inside:
        return None
    position = cand.start_byte
    keys: list[StatementKey] = []
    for statement in inside:
        if statement.start_byte > position and source[position : statement.start_byte].strip():
            return None
        position = max(position, statement.end_byte)
        text = " ".join(
            source[statement.start_byte : statement.end_byte].decode(errors="replace").split()
        )
        scope = directory_of(cand.path) if statement.module.startswith(".") else ""
        keys.append((scope, text))
    if source[position : cand.end_byte].strip():
        return None
    return tuple(keys)


def collapse_imports(sources: SourceCache, items: tuple[PackItem, ...]) -> tuple[PackItem, ...]:
    """Replace an import-only item whose statements all appeared earlier with refs to those items."""
    shown: dict[StatementKey, str] = {}
    out: list[PackItem] = []
    for item in items:
        keys = statement_keys(sources, item)
        if keys is None:
            out.append(item)
            continue
        refs = [shown.get(key) for key in keys]
        if all(ref is not None for ref in refs):
            unique = dict.fromkeys(ref for ref in refs if ref is not None)
            text = COLLAPSED_PREFIX + " ".join(unique)
            if len(text) < len(item.text):
                out.append(replace(item, rep="collapsed", text=text))
                continue
        for key in keys:
            shown.setdefault(key, item.ref)
        out.append(item)
    return tuple(out)
