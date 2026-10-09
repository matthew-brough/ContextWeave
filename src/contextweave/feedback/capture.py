import os
from contextlib import suppress
from pathlib import Path
from typing import Literal

from contextweave.agents.base import AgentConfigError
from contextweave.agents.jsonio import parse_object
from contextweave.index.root import state_dir
from contextweave.index.store import DB_NAME, Store
from contextweave.memory.store import MemoryStore

type Signal = Literal["used", "miss"]

USED: Signal = "used"
MISS: Signal = "miss"
TRACKED_TOOLS = frozenset({"Read", "Edit", "Write"})


def tool_path(tool_name: str, raw_input: str, root: Path) -> str | None:
    """Project-relative posix path a Read/Edit/Write call touched, None otherwise."""
    if tool_name not in TRACKED_TOOLS:
        return None
    try:
        data = parse_object(raw_input, Path("<tool_input>"))
    except AgentConfigError:
        return None
    raw = data.get("file_path")
    if not isinstance(raw, str) or not raw:
        return None
    target = Path(raw) if Path(raw).is_absolute() else root / raw
    try:
        return Path(os.path.normpath(target)).relative_to(os.path.normpath(root)).as_posix()
    except ValueError:
        return None


def _last_pack(index: Store) -> tuple[str, int] | None:
    rows = index.query("SELECT pack_id, ts FROM served_items ORDER BY ts DESC, rowid DESC LIMIT 1")
    if not rows:
        return None
    pack_id, ts = rows[0][0], rows[0][1]
    return (pack_id, ts) if isinstance(pack_id, str) and isinstance(ts, int) else None


def _insert_once(index: Store, pack_id: str, ref: str, signal: Signal, ts: int) -> None:
    seen = index.query(
        "SELECT 1 FROM feedback WHERE pack_id = ? AND ref = ? AND signal = ?",
        (pack_id, ref, signal),
    )
    if not seen:
        index.execute(
            "INSERT INTO feedback(pack_id, ref, signal, ts) VALUES (?, ?, ?, ?)",
            (pack_id, ref, signal, ts),
        )


def record_feedback(
    index: Store, memory: MemoryStore, session_id: str, path: str
) -> tuple[Signal, ...]:
    """Signal for a touched path against the last pack served this turn; nothing without one."""
    turn_start = memory.current_prompt_epoch(session_id)
    pack = _last_pack(index)
    if turn_start is None or pack is None or pack[1] < turn_start:
        return ()
    pack_id, _ = pack
    refs = [
        row[0]
        for row in index.query(
            "SELECT ref FROM served_items WHERE pack_id = ? AND path = ?", (pack_id, path)
        )
        if isinstance(row[0], str)
    ]
    now = memory.now()
    with index.write_transaction():
        if refs:
            for ref in refs:
                _insert_once(index, pack_id, ref, USED, now)
            return (USED,)
        _insert_once(index, pack_id, path, MISS, now)
    return (MISS,)


def capture_tool_feedback(
    root: Path, memory: MemoryStore, session_id: str, tool_name: str, raw_input: str
) -> tuple[Signal, ...]:
    """Fail-open PostToolUse feedback: opens the project db only when it exists."""
    path = tool_path(tool_name, raw_input, root)
    if path is None or not (state_dir(root) / DB_NAME).is_file():
        return ()
    with suppress(Exception), Store.open(root) as index:
        return record_feedback(index, memory, session_id, path)
    return ()
