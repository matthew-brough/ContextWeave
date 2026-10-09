from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Protocol

from contextweave.agents.session_context import build_session_context
from contextweave.core.config import LoadedConfig
from contextweave.core.project import ResolvedProject
from contextweave.core.types import JSON
from contextweave.daemon.client import DaemonUnavailable, IpcClient, RemoteError
from contextweave.daemon.lifecycle import spawn_detached
from contextweave.index.store import Store
from contextweave.memory.capture import (
    end_session,
    enqueue_turn_summary,
    record_prompt,
    record_tool_event,
    start_session,
)
from contextweave.memory.store import MemoryStore, memory_db_path

HEALTH_UNBUILT = "index: not built yet (run cw init)"
HEALTH_UNREADABLE = "index: unreadable (run cw index --rebuild)"


class HookTransport(Protocol):
    """Where hook events go; the daemon IPC client will implement this."""

    def session_start(
        self, project: ResolvedProject, session_id: str, loaded: LoadedConfig
    ) -> str | None: ...

    def prompt(self, project: ResolvedProject, session_id: str, text: str) -> None: ...

    def tool_event(
        self,
        project: ResolvedProject,
        session_id: str,
        tool_name: str,
        summary: str,
        raw_input: str,
        raw_output: str,
    ) -> None: ...

    def stop(self, project: ResolvedProject, session_id: str) -> None: ...

    def session_end(
        self, project: ResolvedProject, session_id: str, reason: str | None
    ) -> None: ...


class NullTransport:
    """Transport that drops everything; used when no backend is reachable."""

    def session_start(
        self, project: ResolvedProject, session_id: str, loaded: LoadedConfig
    ) -> str | None:
        return None

    def prompt(self, project: ResolvedProject, session_id: str, text: str) -> None:
        return None

    def tool_event(
        self,
        project: ResolvedProject,
        session_id: str,
        tool_name: str,
        summary: str,
        raw_input: str,
        raw_output: str,
    ) -> None:
        return None

    def stop(self, project: ResolvedProject, session_id: str) -> None:
        return None

    def session_end(self, project: ResolvedProject, session_id: str, reason: str | None) -> None:
        return None


def index_health(root: Path) -> str:
    """One-line index status read without writing."""
    if not (root / ".contextweave" / "index.db").is_file():
        return HEALTH_UNBUILT
    try:
        with Store.open(root, readonly=True) as store:
            return f"index: {len(store.files())} files, generation {store.generation}"
    except Exception:
        return HEALTH_UNREADABLE


class InprocTransport:
    """Writes straight to the project's memory.db; warnings supplies live backend warnings."""

    def __init__(
        self, env: Mapping[str, str], warnings: Callable[[], Sequence[str]] | None = None
    ) -> None:
        self._env = env
        self._extra_warnings = warnings

    @contextmanager
    def _store(self, project: ResolvedProject) -> Generator[MemoryStore]:
        store = MemoryStore.open(memory_db_path(project.key, self._env), project.key)
        try:
            yield store
        finally:
            store.close()

    def session_start(
        self, project: ResolvedProject, session_id: str, loaded: LoadedConfig
    ) -> str | None:
        with self._store(project) as store:
            start_session(store, session_id)
            warnings = [
                f"warning: config_invalid key {issue.key} in {issue.file}"
                for issue in loaded.issues
            ]
            if self._extra_warnings is not None:
                warnings.extend(self._extra_warnings())
            return build_session_context(
                title=f"ContextWeave project {project.root.name} ({project.key})",
                health=index_health(project.root),
                warnings=warnings,
                decisions=store.recent("decision", 5),
                code_areas=store.recent("code_area", 5),
                limit_tokens=loaded.config.hooks.session_start_tokens,
            )

    def prompt(self, project: ResolvedProject, session_id: str, text: str) -> None:
        with self._store(project) as store:
            record_prompt(store, session_id, text)

    def tool_event(
        self,
        project: ResolvedProject,
        session_id: str,
        tool_name: str,
        summary: str,
        raw_input: str,
        raw_output: str,
    ) -> None:
        with self._store(project) as store:
            record_tool_event(
                store,
                session_id,
                store.current_prompt_number(session_id),
                tool_name,
                summary=summary,
                raw_input=raw_input,
                raw_output=raw_output,
            )

    def stop(self, project: ResolvedProject, session_id: str) -> None:
        with self._store(project) as store:
            number = store.current_prompt_number(session_id)
            if number > 0:
                enqueue_turn_summary(store, session_id, number)

    def session_end(self, project: ResolvedProject, session_id: str, reason: str | None) -> None:
        with self._store(project) as store:
            end_session(store, session_id, "completed", reason)


HOOK_TIMEOUT_S = 0.2


class IpcTransport:
    """Sends hook events to the daemon; every failure is swallowed so hooks fail open (V17)."""

    def __init__(
        self, client: IpcClient, env: Mapping[str, str], *, timeout_s: float = HOOK_TIMEOUT_S
    ) -> None:
        self._client = client
        self._env = env
        self._timeout_s = timeout_s

    def _send(self, method: str, params: dict[str, JSON], *, wake: bool = False) -> JSON:
        try:
            return self._client.call(method, params, timeout_s=self._timeout_s)
        except DaemonUnavailable:
            if wake:
                self._wake()
        except RemoteError:
            pass
        return None

    def _wake(self) -> None:
        with suppress(OSError):
            spawn_detached(self._env)

    @staticmethod
    def _base(project: ResolvedProject, session_id: str) -> dict[str, JSON]:
        return {
            "key": project.key,
            "root": str(project.root),
            "via": project.via,
            "session_id": session_id,
        }

    def session_start(
        self, project: ResolvedProject, session_id: str, loaded: LoadedConfig
    ) -> str | None:
        result = self._send("hook.session_start", self._base(project, session_id), wake=True)
        return result if isinstance(result, str) else None

    def prompt(self, project: ResolvedProject, session_id: str, text: str) -> None:
        self._send("hook.prompt", {**self._base(project, session_id), "text": text})

    def tool_event(
        self,
        project: ResolvedProject,
        session_id: str,
        tool_name: str,
        summary: str,
        raw_input: str,
        raw_output: str,
    ) -> None:
        self._send(
            "hook.tool_event",
            {
                **self._base(project, session_id),
                "tool_name": tool_name,
                "summary": summary,
                "raw_input": raw_input,
                "raw_output": raw_output,
            },
        )

    def stop(self, project: ResolvedProject, session_id: str) -> None:
        self._send("hook.stop", self._base(project, session_id))

    def session_end(self, project: ResolvedProject, session_id: str, reason: str | None) -> None:
        params = self._base(project, session_id)
        if reason is not None:
            params["reason"] = reason
        self._send("hook.session_end", params)
