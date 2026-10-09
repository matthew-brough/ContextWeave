from collections.abc import Sequence
from dataclasses import replace
from pathlib import PurePosixPath

from contextweave.core.config import AnnotateConfig
from contextweave.index.records import FileData, SchemaNodeRecord, SymbolRecord
from contextweave.index.store import Store
from contextweave.retrieve.pack import CachedAnnotation, Folds, PackItem, Rep

BOILERPLATE_DUNDERS = frozenset(
    {"__repr__", "__str__", "__eq__", "__ne__", "__hash__", "__lt__", "__le__", "__gt__", "__ge__"}
)
ACCESSOR_PREFIXES = ("get_", "set_")
ACCESSOR_MAX_LINES = 3
UTILITY_SHARE = 0.7
MEMBER_MAX_LINES = 3
REEXPORT_FILES = frozenset(
    {"__init__.py", "index.ts", "index.tsx", "index.js", "index.jsx", "index.mjs"}
)
REEXPORT_PREFIXES = ("from ", "import ", "export ", "__all__")
CALLABLE_KINDS = frozenset({"function", "method"})
NAMED_KINDS = frozenset({"function", "method", "class"})
TEST_DIRECTORIES = frozenset({"test", "tests"})
TEST_FILES = frozenset({"conftest.py"})
BATCH = 500


def boilerplate_member(name: str, kind: str, lines: int) -> bool:
    """A dunder comparison/repr method or a short getter/setter."""
    if kind not in CALLABLE_KINDS:
        return False
    short = name.rsplit(".", 1)[-1]
    return short in BOILERPLATE_DUNDERS or (
        short.startswith(ACCESSOR_PREFIXES) and lines <= ACCESSOR_MAX_LINES
    )


def is_boilerplate(item: PackItem) -> bool:
    return boilerplate_member(
        item.cand.qualified_name or "", item.cand.kind or "", len(item.text.splitlines())
    )


def is_test_item(item: PackItem) -> bool:
    """A named symbol living in a test or fixture file."""
    if item.cand.kind not in NAMED_KINDS or item.cand.qualified_name is None:
        return False
    path = PurePosixPath(item.cand.path)
    return (
        bool(TEST_DIRECTORIES & set(path.parts[:-1]))
        or path.name in TEST_FILES
        or path.name.startswith("test_")
        or path.name.endswith("_test.py")
    )


def summary_text(item: PackItem) -> str | None:
    """Signature plus cached purpose, or None when no current annotation exists."""
    if item.annotation is None:
        return None
    head = item.signature or item.cand.qualified_name
    purpose = item.annotation.purpose
    return purpose if head is None else f"{head}\n{purpose}"


def can_fold_boilerplate(item: PackItem, allow_critical: bool) -> bool:
    return (
        item.rep == "source"
        and item.signature is not None
        and len(item.signature) < len(item.text)
        and is_boilerplate(item)
    )


def to_signature(item: PackItem) -> PackItem | None:
    return item.with_signature_rep()


def can_sign(item: PackItem, allow_critical: bool) -> bool:
    return (
        item.rep in ("source", "summary")
        and item.signature is not None
        and len(item.signature) < len(item.text)
    )


def can_drop(item: PackItem, allow_critical: bool) -> bool:
    return allow_critical or not item.cand.critical


def to_ref(item: PackItem) -> PackItem | None:
    return None


def folded_with(item: PackItem, text: str | None, rep: Rep) -> PackItem | None:
    return None if text is None else item.with_text_rep(rep, text)


def can_fold_reexports(item: PackItem, allow_critical: bool) -> bool:
    text = item.folds.reexports
    return item.rep == "source" and text is not None and len(text) < len(item.text)


def to_reexports(item: PackItem) -> PackItem | None:
    return folded_with(item, item.folds.reexports, "signature")


def can_key_paths(item: PackItem, allow_critical: bool) -> bool:
    text = item.folds.key_paths
    return item.rep == "source" and text is not None and len(text) < len(item.text)


def to_key_paths(item: PackItem) -> PackItem | None:
    return folded_with(item, item.folds.key_paths, "schema")


def can_list_members(item: PackItem, allow_critical: bool) -> bool:
    text = item.folds.members
    return item.rep == "source" and text is not None and len(text) < len(item.text)


def to_members(item: PackItem) -> PackItem | None:
    return folded_with(item, item.folds.members, "signature")


def can_name_only(item: PackItem, allow_critical: bool) -> bool:
    name = item.cand.qualified_name
    return (
        item.rep == "source"
        and name is not None
        and len(name) < len(item.text)
        and is_test_item(item)
    )


def to_names_only(item: PackItem) -> PackItem | None:
    return item.with_text_rep("signature", item.cand.qualified_name or item.text)


def can_summarize(item: PackItem, allow_critical: bool) -> bool:
    text = summary_text(item)
    return item.rep == "source" and text is not None and len(text) < len(item.text)


def to_summary(item: PackItem) -> PackItem | None:
    text = summary_text(item)
    return None if text is None else item.with_text_rep("summary", text)


def attach_annotations(
    store: Store, config: AnnotateConfig, items: Sequence[PackItem]
) -> list[PackItem]:
    """Attach the cached annotation matching each item's current chunk hash, model and prompt version."""
    wanted = {item.cand.chunk_id for item in items}
    found: dict[int, tuple[int, int, CachedAnnotation]] = {}
    ids = sorted(wanted)
    for offset in range(0, len(ids), BATCH):
        batch = ids[offset : offset + BATCH]
        marks = ", ".join("?" for _ in batch)
        rows = store.query(
            "SELECT a.chunk_id, c.start_byte, c.end_byte, a.purpose FROM annotations a"
            " JOIN chunks c ON c.id = a.chunk_id AND c.content_hash = a.content_hash"
            f" WHERE a.model_id = ? AND a.prompt_version = ? AND a.chunk_id IN ({marks})",
            [config.model, config.prompt_version, *batch],
        )
        for row in rows:
            chunk_id, start, end, purpose = row[0], row[1], row[2], row[3]
            if (
                isinstance(chunk_id, int)
                and isinstance(start, int)
                and isinstance(end, int)
                and isinstance(purpose, str)
                and purpose.strip()
            ):
                found[chunk_id] = (
                    start,
                    end,
                    CachedAnnotation(purpose.strip(), config.model, config.prompt_version),
                )
    attached: list[PackItem] = []
    for item in items:
        hit = found.get(item.cand.chunk_id)
        exact = hit is not None and hit[:2] == (item.cand.start_byte, item.cand.end_byte)
        attached.append(replace(item, annotation=hit[2]) if hit is not None and exact else item)
    return attached


def key_path_lines(nodes: Sequence[SchemaNodeRecord], start: int, end: int) -> str | None:
    """Key paths with kind, child count and uniformity for the nodes inside a span; never values."""
    lines: list[str] = []
    for node in nodes:
        if node.start_byte < start or node.end_byte > end or not node.key_path:
            continue
        parts = [node.key_path, node.kind]
        if node.children_count:
            parts.append(f"{node.children_count} children")
        if node.uniformity:
            parts.append(node.uniformity)
        lines.append(" ".join(parts))
    return "\n".join(lines) if lines else None


def class_members(
    symbols: Sequence[SymbolRecord], name: str, start: int, end: int
) -> tuple[SymbolRecord, ...]:
    return tuple(
        symbol
        for symbol in symbols
        if symbol.parent == name
        and symbol.kind in CALLABLE_KINDS
        and start <= symbol.start_byte
        and symbol.end_byte <= end
    )


def member_list(item: PackItem, symbols: Sequence[SymbolRecord]) -> str | None:
    """Class header plus member signatures when the class is a utility class."""
    cand = item.cand
    name = cand.qualified_name
    if cand.kind != "class" or name is None:
        return None
    members = class_members(symbols, name, cand.start_byte, cand.end_byte)
    if not members:
        return None
    boilerplate = sum(
        boilerplate_member(m.qualified_name, m.kind, m.end_line - m.start_line + 1) for m in members
    )
    small = all(m.end_line - m.start_line + 1 <= MEMBER_MAX_LINES for m in members)
    if boilerplate / len(members) < UTILITY_SHARE and not small:
        return None
    header = item.signature or item.text.split("\n", 1)[0].rstrip()
    return "\n".join([header, *(f"  {m.signature or m.name}" for m in members)])


def reexport_list(item: PackItem, data_imports: Sequence[str]) -> str | None:
    """The module names a pure re-export chunk of an index/init file pulls in."""
    if item.cand.qualified_name is not None:
        return None
    if item.cand.path.rsplit("/", 1)[-1] not in REEXPORT_FILES or not data_imports:
        return None
    lines = [line.strip() for line in item.text.splitlines() if line.strip()]
    if not all(line.startswith(REEXPORT_PREFIXES) for line in lines):
        return None
    return "re-exports: " + ", ".join(dict.fromkeys(data_imports))


def attach_folds(store: Store, items: Sequence[PackItem]) -> list[PackItem]:
    """Precompute the schema, member-list and re-export renderings from the index (no model)."""
    cache: dict[str, FileData | None] = {}
    folded: list[PackItem] = []
    for item in items:
        path = item.cand.path
        if path not in cache:
            cache[path] = store.get_data(path)
        data = cache[path]
        if data is None:
            folded.append(item)
            continue
        cand = item.cand
        key_paths = (
            key_path_lines(data.schema_nodes, cand.start_byte, cand.end_byte)
            if cand.profile == "schema"
            else None
        )
        modules = [
            record.module
            for record in data.imports
            if item.start_line <= record.line <= item.end_line
        ]
        folds = Folds(
            key_paths=key_paths,
            members=member_list(item, data.symbols),
            reexports=reexport_list(item, modules),
        )
        folded.append(replace(item, folds=folds))
    return folded


def prepare_lossy(
    store: Store, config: AnnotateConfig, items: Sequence[PackItem]
) -> list[PackItem]:
    """Everything the lossy rungs read, gathered from the index and the annotation cache."""
    return attach_annotations(store, config, attach_folds(store, items))
