import json
from pathlib import Path

import pytest

from contextweave.agents.hooks import run_hook
from contextweave.agents.transport import InprocTransport
from contextweave.core.paths import user_state_dir
from contextweave.core.project import ResolvedProject
from contextweave.core.refs import project_key
from contextweave.feedback.capture import capture_tool_feedback, record_feedback, tool_path
from contextweave.feedback.served import FEATURE_NAMES
from contextweave.index.projects import ProjectRegistry
from contextweave.index.store import Store
from contextweave.memory.store import MemoryStore, memory_db_path

SESSION = "s1"
PACK = "pack-1"


def serve(store: Store, paths: list[str], ts: int, pack: str = PACK) -> None:
    columns = ", ".join(("pack_id", "ref", "path", *FEATURE_NAMES, "ts"))
    marks = ", ".join("?" * (len(FEATURE_NAMES) + 4))
    for path in paths:
        store.execute(
            f"INSERT INTO served_items({columns}) VALUES ({marks})",
            (pack, f"cw://{path}", path, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 0.0, ts),
        )


def signals(store: Store) -> list[tuple[str, str]]:
    rows = store.query("SELECT ref, signal FROM feedback ORDER BY rowid")
    return [(str(ref), str(signal)) for ref, signal in rows]


@pytest.fixture
def memory(tmp_path: Path) -> MemoryStore:
    clock = [100.0]
    store = MemoryStore.open(tmp_path / "memory.db", "k", clock=lambda: clock[0])
    store.add_prompt(SESSION, "find alpha")
    return store


def test_tool_path_variants(tmp_path: Path) -> None:
    root = tmp_path
    inside = json.dumps({"file_path": str(root / "src" / "a.py")})
    assert tool_path("Read", inside, root) == "src/a.py"
    assert tool_path("Edit", json.dumps({"file_path": "src/a.py"}), root) == "src/a.py"
    assert tool_path("Write", json.dumps({"file_path": "/etc/passwd"}), root) is None
    assert tool_path("Bash", inside, root) is None
    assert tool_path("Read", "not json", root) is None
    assert tool_path("Read", json.dumps({"other": 1}), root) is None
    assert tool_path("Read", json.dumps([1]), root) is None


def test_path_in_pack_is_used_and_deduplicated(index: Store, memory: MemoryStore) -> None:
    serve(index, ["a.py", "b.py"], ts=100)
    assert record_feedback(index, memory, SESSION, "a.py") == ("used",)
    assert record_feedback(index, memory, SESSION, "a.py") == ("used",)
    assert signals(index) == [("cw://a.py", "used")]


def test_path_outside_pack_is_a_miss(index: Store, memory: MemoryStore) -> None:
    serve(index, ["a.py"], ts=101)
    assert record_feedback(index, memory, SESSION, "c.py") == ("miss",)
    assert signals(index) == [("c.py", "miss")]


def test_no_feedback_without_a_pack_this_turn(index: Store, memory: MemoryStore) -> None:
    assert record_feedback(index, memory, SESSION, "a.py") == ()
    serve(index, ["a.py"], ts=99)
    assert record_feedback(index, memory, SESSION, "a.py") == ()
    assert record_feedback(index, memory, "unknown-session", "a.py") == ()
    assert signals(index) == []


def test_only_the_last_pack_counts(index: Store, memory: MemoryStore) -> None:
    serve(index, ["a.py"], ts=100, pack="old")
    serve(index, ["b.py"], ts=105, pack="new")
    assert record_feedback(index, memory, SESSION, "a.py") == ("miss",)


def test_capture_is_fail_open(tmp_path: Path, memory: MemoryStore) -> None:
    raw = json.dumps({"file_path": str(tmp_path / "a.py")})
    assert capture_tool_feedback(tmp_path, memory, SESSION, "Read", raw) == ()
    state = tmp_path / ".contextweave"
    state.mkdir()
    (state / "index.db").write_bytes(b"garbage")
    assert capture_tool_feedback(tmp_path, memory, SESSION, "Read", raw) == ()


def test_hook_sequence_records_used_then_miss(tmp_path: Path, env: dict[str, str]) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    ProjectRegistry(user_state_dir(env)).register(root)
    key = project_key(root)
    with Store.open(root) as index:
        serve(index, ["a.py"], ts=2**40)
    memory = MemoryStore.open(memory_db_path(key, env), key)
    memory.add_prompt(SESSION, "look at alpha")
    memory.close()
    transport = InprocTransport(env)

    def post(name: str, path: str) -> None:
        body = {
            "session_id": SESSION,
            "cwd": str(root),
            "tool_name": name,
            "tool_input": {"file_path": str(root / path)},
            "tool_response": "ok",
        }
        output, code = run_hook(
            "PostToolUse", json.dumps(body).encode(), env=env, transport=transport
        )
        assert (output, code) == (b"", 0)

    post("Read", "a.py")
    post("Edit", "a.py")
    post("Read", "z.py")
    post("Grep", "z.py")
    with Store.open(root, readonly=True) as index:
        assert signals(index) == [("cw://a.py", "used"), ("z.py", "miss")]
    _ = ResolvedProject
