import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from agents_support import make_root

from contextweave.agents.cli_entry import run_hook
from contextweave.agents.hooks import HookLimits, parse_event
from contextweave.agents.hooks import run_hook as dispatch
from contextweave.agents.transport import HookTransport, InprocTransport, NullTransport
from contextweave.core.budget import bytes35
from contextweave.core.config import Config, LoadedConfig
from contextweave.core.paths import user_state_dir
from contextweave.core.project import ResolvedProject
from contextweave.core.refs import project_key
from contextweave.index.indexer import index_project
from contextweave.index.projects import ProjectRegistry
from contextweave.memory.store import MemoryStore, memory_db_path

SESSION = "sess-1"


def payload(root: Path, **extra: object) -> bytes:
    body: dict[str, object] = {"session_id": SESSION, "cwd": str(root), **extra}
    return json.dumps(body).encode()


@pytest.fixture
def project(root: Path, env: dict[str, str]) -> ResolvedProject:
    ProjectRegistry(user_state_dir(env)).register(root)
    return ResolvedProject(project_key(root), root, "cwd")


@contextmanager
def store_of(env: dict[str, str], key: str) -> Generator[MemoryStore]:
    store = MemoryStore.open(memory_db_path(key, env), key)
    try:
        yield store
    finally:
        store.close()


def count(store: MemoryStore, table: str) -> int:
    row: tuple[int] = store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return row[0]


class Recorder:
    def __init__(self, context: str | None = None) -> None:
        self.calls: list[str] = []
        self.context = context

    def session_start(
        self, project: ResolvedProject, session_id: str, loaded: LoadedConfig
    ) -> str | None:
        self.calls.append("session_start")
        return self.context

    def prompt(self, project: ResolvedProject, session_id: str, text: str) -> None:
        self.calls.append(f"prompt:{text}")

    def tool_event(
        self,
        project: ResolvedProject,
        session_id: str,
        tool_name: str,
        summary: str,
        raw_input: str,
        raw_output: str,
    ) -> None:
        self.calls.append(f"tool:{tool_name}:{summary}")

    def stop(self, project: ResolvedProject, session_id: str) -> None:
        self.calls.append("stop")

    def session_end(self, project: ResolvedProject, session_id: str, reason: str | None) -> None:
        self.calls.append(f"end:{reason}")


class Exploding(Recorder):
    def session_start(
        self, project: ResolvedProject, session_id: str, loaded: LoadedConfig
    ) -> str | None:
        raise RuntimeError("daemon exploded")

    def prompt(self, project: ResolvedProject, session_id: str, text: str) -> None:
        raise RuntimeError("daemon exploded")


class Slow(Recorder):
    def __init__(self) -> None:
        super().__init__("late")
        self.release = threading.Event()

    def session_start(
        self, project: ResolvedProject, session_id: str, loaded: LoadedConfig
    ) -> str | None:
        self.release.wait(5)
        return self.context

    def prompt(self, project: ResolvedProject, session_id: str, text: str) -> None:
        self.release.wait(5)
        self.calls.append("prompt")


def test_transports_satisfy_protocol() -> None:
    transports: list[HookTransport] = [NullTransport(), InprocTransport({}), Recorder()]
    assert len(transports) == 3


def test_event_name_normalisation() -> None:
    assert parse_event("session-start") is parse_event("SessionStart")
    assert parse_event("post_tool_use") is not None
    assert parse_event("bogus") is None


def test_dispatch_each_event(root: Path, project: ResolvedProject, env: dict[str, str]) -> None:
    rec = Recorder("ctx")
    out, code = run_hook("SessionStart", payload(root), env, rec)
    assert code == 0
    assert json.loads(out) == {
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "ctx"}
    }
    assert run_hook("UserPromptSubmit", payload(root, prompt="hi"), env, rec) == (b"", 0)
    tool = payload(root, tool_name="Read", tool_input={"file_path": "a.py"}, tool_response="x")
    assert run_hook("PostToolUse", tool, env, rec) == (b"", 0)
    assert run_hook("Stop", payload(root), env, rec) == (b"", 0)
    assert run_hook("SessionEnd", payload(root, reason="clear"), env, rec) == (b"", 0)
    assert rec.calls == ["session_start", "prompt:hi", "tool:Read:Read: a.py", "stop", "end:clear"]


def test_inproc_records_full_session(
    root: Path, project: ResolvedProject, env: dict[str, str]
) -> None:
    run_hook("SessionStart", payload(root), env, InprocTransport(env))
    run_hook("UserPromptSubmit", payload(root, prompt="why is x"), env, InprocTransport(env))
    tool = payload(
        root, tool_name="Edit", tool_input={"file_path": "a.py"}, tool_response={"ok": 1}
    )
    run_hook("PostToolUse", tool, env, InprocTransport(env))
    run_hook("Stop", payload(root), env, InprocTransport(env))
    run_hook("SessionEnd", payload(root), env, InprocTransport(env))
    with store_of(env, project.key) as store:
        assert count(store, "sessions") == 1
        assert count(store, "prompts") == 1
        row: tuple[str] = store.conn.execute("SELECT status FROM sessions").fetchone()
        assert row[0] == "completed"


@pytest.mark.parametrize(
    "event", ["SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "SessionEnd"]
)
@pytest.mark.parametrize("stdin", [b"", b"{not json", b"[]", b'{"cwd": 3}', b"\xff\xfe"])
def test_malformed_stdin_is_silent(event: str, stdin: bytes, env: dict[str, str]) -> None:
    assert run_hook(event, stdin, env, Recorder("x")) == (b"", 0)


def test_unknown_event_is_silent(root: Path, env: dict[str, str]) -> None:
    assert run_hook("Nonsense", payload(root), env) == (b"", 0)


def test_unregistered_project_is_noop(root: Path, env: dict[str, str]) -> None:
    rec = Recorder("x")
    assert run_hook("SessionStart", payload(root), env, rec) == (b"", 0)
    assert rec.calls == []


@pytest.mark.parametrize("event", ["SessionStart", "UserPromptSubmit"])
def test_transport_exception_swallowed(
    event: str, root: Path, project: ResolvedProject, env: dict[str, str]
) -> None:
    assert run_hook(event, payload(root, prompt="p"), env, Exploding()) == (b"", 0)


def test_session_start_deadline_gives_no_output(
    root: Path, project: ResolvedProject, env: dict[str, str]
) -> None:
    slow = Slow()
    limits = HookLimits(total_s=1.0, session_start_s=0.05)
    assert run_hook_with(limits, "SessionStart", payload(root), env, slow) == (b"", 0)
    slow.release.set()


def test_total_deadline_gives_no_output(
    root: Path, project: ResolvedProject, env: dict[str, str]
) -> None:
    slow = Slow()
    limits = HookLimits(total_s=0.05, session_start_s=0.05)
    assert run_hook_with(limits, "UserPromptSubmit", payload(root, prompt="x"), env, slow) == (
        b"",
        0,
    )
    slow.release.set()


def run_hook_with(
    limits: HookLimits, event: str, stdin: bytes, env: dict[str, str], transport: HookTransport
) -> tuple[bytes, int]:
    return dispatch(event, stdin, env=env, transport=transport, limits=limits)


def test_session_start_injection_is_budgeted_and_scoped(
    tmp_path: Path, root: Path, project: ResolvedProject, env: dict[str, str]
) -> None:
    other = make_root(tmp_path, "other")
    ProjectRegistry(user_state_dir(env)).register(other)
    out, code = run_hook("SessionStart", payload(root), env, InprocTransport(env))
    assert code == 0
    text = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert bytes35(text) <= 400
    assert project_key(other) not in text
    assert "index:" in text and project.key in text


def test_session_start_respects_configured_budget(
    root: Path, project: ResolvedProject, env: dict[str, str]
) -> None:
    (root / ".contextweave.toml").write_text(
        "[hooks]\nsession_start_tokens = 30\n", encoding="utf-8"
    )
    out, _ = run_hook("SessionStart", payload(root), env, InprocTransport(env))
    assert bytes35(json.loads(out)["hookSpecificOutput"]["additionalContext"]) <= 30


def test_session_start_warns_on_config_issue(
    root: Path, project: ResolvedProject, env: dict[str, str]
) -> None:
    (root / ".contextweave.toml").write_text("[hooks]\nbogus = 1\n", encoding="utf-8")
    out, _ = run_hook("SessionStart", payload(root), env, InprocTransport(env))
    text = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "config_invalid" in text and "hooks.bogus" in text


def test_session_start_reports_built_index(
    root: Path, project: ResolvedProject, env: dict[str, str]
) -> None:
    index_project(root, Config())
    out, _ = run_hook("SessionStart", payload(root), env, InprocTransport(env))
    assert "index: 1 files" in json.loads(out)["hookSpecificOutput"]["additionalContext"]


def test_subdirectory_cwd_resolves_to_root(
    root: Path, project: ResolvedProject, env: dict[str, str]
) -> None:
    sub = root / "pkg"
    sub.mkdir()
    rec = Recorder()
    run_hook("Stop", payload(sub), env, rec)
    assert rec.calls == ["stop"]
