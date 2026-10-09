from contextweave.core.config import Config
from contextweave.core.refs import project_key
from contextweave.index.records import FileStamp
from contextweave.index.store import Store
from contextweave.lang.parse import parse_bytes
from contextweave.retrieve.context import Mode, QueryContext


def ingest(store: Store, relative: str, text: str) -> None:
    target = store.root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    stat = target.stat()
    source = text.encode()
    store.upsert_file(
        relative, parse_bytes(relative, source), source, FileStamp(stat.st_mtime_ns, stat.st_size)
    )


def make_ctx(
    store: Store,
    query: str,
    *,
    config: Config | None = None,
    scope: str | None = None,
    mode: Mode = "auto",
) -> QueryContext:
    return QueryContext(
        query=query,
        store=store,
        config=config or Config(),
        project_key=project_key(store.root),
        root=store.root,
        scope=scope,
        mode=mode,
    )
