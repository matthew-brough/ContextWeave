import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from contextweave.agents.transport import InprocTransport
from contextweave.core.config import Config, load_config
from contextweave.core.dist import package_version
from contextweave.core.paths import user_state_dir
from contextweave.core.project import ResolvedProject, Via, resolve_project
from contextweave.core.registry import Registry, UnknownKeyError
from contextweave.core.types import JSON
from contextweave.daemon.client import RemoteError
from contextweave.daemon.codec import (
    encode_body,
    encode_resources,
    encode_tool_result,
)
from contextweave.daemon.models import ModelRuntime
from contextweave.daemon.projects import ProjectManager
from contextweave.daemon.runner import IndexRunner
from contextweave.daemon.scheduler import ActivityGate, JobKind, Priority, Scheduler, Worker
from contextweave.daemon.skipwarn import skip_rate_warnings
from contextweave.daemon.watcher import FileEvent, PollWatcher
from contextweave.index.projects import ProjectEntry, ProjectRegistry
from contextweave.memory.savings import ledger_sink
from contextweave.models.residency import Profile
from contextweave.server import resources
from contextweave.server.context import PackCache, ToolContext, inproc_opener
from contextweave.server.tools import TOOLS

type Params = Mapping[str, JSON]
type Handler = Callable[[DaemonService, Params], JSON]

METHODS: Registry[str, Handler] = Registry()
WAIT_TIMEOUT_S = 3600.0


def method(name: str) -> Callable[[Handler], Handler]:
    """Register a handler under an IPC method name."""

    def register(handler: Handler) -> Handler:
        return METHODS.register(name, handler)

    return register


def _str(params: Params, name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise RemoteError("bad_params", f"{name} must be a non-empty string")
    return value


def _opt_str(params: Params, name: str) -> str | None:
    value = params.get(name)
    return value if isinstance(value, str) and value else None


def _flag(params: Params, name: str) -> bool:
    return params.get(name) is True


def _strings(params: Params, name: str) -> tuple[str, ...]:
    value = params.get(name)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RemoteError("bad_params", f"{name} must be a list of strings")
    return tuple(item for item in value if isinstance(item, str))


def _via(value: str) -> Via:
    match value:
        case "arg" | "path" | "ref" | "roots" | "cwd":
            return value
        case _:
            raise RemoteError("bad_params", f"unknown via {value}")


@dataclass(slots=True)
class DaemonService:
    """Daemon state and the IPC method table; every handler is synchronous and thread-safe."""

    env: Mapping[str, str]
    config: Config
    registry: ProjectRegistry
    scheduler: Scheduler
    gate: ActivityGate
    runner: IndexRunner
    projects: ProjectManager
    worker: Worker
    base: ToolContext
    hooks: InprocTransport
    models: ModelRuntime | None
    started_at: float = field(default_factory=time.time)
    shutdown: threading.Event = field(default_factory=threading.Event)
    last_activity: float = field(default_factory=time.monotonic)

    @classmethod
    def create(
        cls,
        env: Mapping[str, str],
        config: Config,
        *,
        models: ModelRuntime | None = None,
        poll_interval_s: float = 2.0,
        debounce_s: float = 0.3,
    ) -> DaemonService:
        registry = ProjectRegistry(user_state_dir(env))
        scheduler = Scheduler(debounce_s=debounce_s)
        gate = ActivityGate()

        def make_watcher(
            root: Path, key: str, on_events: Callable[[Sequence[FileEvent]], None]
        ) -> PollWatcher:
            return PollWatcher(
                root, config.index.ignore, poll_interval_s, on_events, config.index.include
            )

        projects = ProjectManager(
            scheduler, make_watcher, idle_s=config.daemon.project_idle_minutes * 60.0
        )

        def root_of(key: str) -> Path | None:
            opened = projects.get(key)
            return opened.root if opened is not None else registry.known_roots().get(key)

        runner = IndexRunner(config, scheduler, root_of, models)
        worker = Worker(
            scheduler, runner.run, lambda job, error: runner.note_error(job.key, error), gate
        )
        base = ToolContext(
            config,
            registry,
            inproc_opener(config, readonly=False),
            packs=PackCache(config.server.pack_cache),
            ledger=ledger_sink(env),
        )
        return cls(
            env,
            config,
            registry,
            scheduler,
            gate,
            runner,
            projects,
            worker,
            base,
            InprocTransport(env, skip_rate_warnings(models)),
            models,
        )

    def start(self) -> None:
        self.worker.start()
        if self.models is not None:
            threading.Thread(target=self._serve_residency, name="cw-residency", daemon=True).start()

    def _serve_residency(self) -> None:
        if self.models is None:
            return
        try:
            self.models.apply(Profile.SERVE)
        except Exception as exc:
            self.runner.note_error("residency", exc)

    def stop(self) -> None:
        self.projects.close_all()
        self.worker.stop()
        if self.models is not None:
            self.models.close()

    def dispatch(self, name: str, params: Params) -> JSON:
        """Run one method; unknown names and handler failures become RemoteError."""
        self.last_activity = time.monotonic()
        try:
            handler = METHODS.get(name)
        except UnknownKeyError:
            raise RemoteError("unknown_method", name) from None
        try:
            return handler(self, params)
        except RemoteError:
            raise
        except Exception as exc:
            raise RemoteError("internal_error", f"{type(exc).__name__}: {exc}") from exc

    def entry(self, params: Params) -> ProjectEntry:
        """Registered project named by params['project'] (key or root path)."""
        ident = _str(params, "project")
        found = self.registry.get(ident)
        if found is None:
            raise RemoteError("not_indexed", f"no registered project {ident}")
        return found

    def open_project(self, entry: ProjectEntry) -> None:
        self.projects.open(entry.key, entry.root)
        self.registry.touch(entry.key)


@method("ping")
def _ping(service: DaemonService, params: Params) -> JSON:
    return {"pid": os.getpid(), "version": package_version()}


@method("status")
def _status(service: DaemonService, params: Params) -> JSON:
    now = time.monotonic()
    activity = service.runner.snapshot()
    projects: list[JSON] = []
    for project in service.projects.snapshot():
        stats = activity.get(project.key)
        projects.append(
            {
                "key": project.key,
                "root": str(project.root),
                "idle_s": round(now - project.last_request, 3),
                "pending": service.scheduler.pending(project.key),
                "indexed_paths": stats.indexed_paths if stats else 0,
                "reconciles": stats.reconciles if stats else 0,
                "rebuilds": stats.rebuilds if stats else 0,
                "rebuilding": stats.rebuilding if stats else False,
                "last_error": stats.last_error if stats else None,
            }
        )
    profile = service.models.slots.residency.current if service.models is not None else None
    slots: list[JSON] = []
    if service.models is not None:
        for health in service.models.slots.status():
            slots.append(
                {
                    "slot": health.slot.value,
                    "active": str(health.active) if health.active is not None else None,
                    "down": [str(device) for device in health.down],
                    "skips_total": health.skips_total,
                    "calls_in_window": health.calls_in_window,
                    "skip_rate": round(health.skip_rate, 4),
                    "warn": health.warn,
                    "slow_rate": round(health.slow_rate, 4),
                    "slow_warn": health.slow_warn,
                    "reason": health.reason,
                }
            )
    return {
        "pid": os.getpid(),
        "version": package_version(),
        "uptime_s": round(time.time() - service.started_at, 3),
        "queue": service.scheduler.pending(),
        "profile": profile.value if profile is not None else None,
        "models": slots,
        "projects": projects,
    }


@method("shutdown")
def _shutdown(service: DaemonService, params: Params) -> JSON:
    service.shutdown.set()
    return {"stopping": True}


@method("tool.call")
def _tool_call(service: DaemonService, params: Params) -> JSON:
    name = _str(params, "name")
    if name not in TOOLS:
        raise RemoteError("unknown_tool", name)
    arguments = params.get("arguments", {})
    if not isinstance(arguments, dict):
        raise RemoteError("bad_params", "arguments must be a map")
    cwd = _opt_str(params, "cwd")
    context = replace(
        service.base,
        cwd=Path(cwd) if cwd is not None else None,
        roots=tuple(Path(root) for root in _strings(params, "roots")),
    )
    resolution = resolve_project(
        arguments, context.roots, context.cwd, service.registry.known_roots()
    )
    spec = TOOLS.get(name)
    with service.gate.query():
        if not isinstance(resolution, ResolvedProject):
            return encode_tool_result(spec.run(context, arguments))
        entry = service.registry.get(resolution.key)
        if entry is not None:
            service.open_project(entry)
        with service.runner.guard(resolution.key).shared():
            return encode_tool_result(spec.run(context, arguments))


@method("resource.list")
def _resource_list(service: DaemonService, params: Params) -> JSON:
    return encode_resources(resources.list_resources(service.base))


@method("resource.read")
def _resource_read(service: DaemonService, params: Params) -> JSON:
    return encode_body(resources.read_resource(service.base, _str(params, "uri")))


@method("project.open")
def _project_open(service: DaemonService, params: Params) -> JSON:
    entry = service.entry(params)
    service.open_project(entry)
    return {"key": entry.key, "root": str(entry.root)}


@method("project.close")
def _project_close(service: DaemonService, params: Params) -> JSON:
    entry = service.entry(params)
    return {"key": entry.key, "closed": service.projects.close(entry.key)}


@method("project.list")
def _project_list(service: DaemonService, params: Params) -> JSON:
    return [
        {"key": project.key, "root": str(project.root)} for project in service.projects.snapshot()
    ]


@method("project.reconcile")
def _project_reconcile(service: DaemonService, params: Params) -> JSON:
    entry = service.entry(params)
    service.open_project(entry)
    if not _flag(params, "wait"):
        service.scheduler.enqueue_project(entry.key, JobKind.RECONCILE)
        return {"queued": True}
    plan = service.runner.reconcile_plan(entry.root)
    if plan.paths:
        service.runner.index_paths(entry.key, entry.root, plan.paths)
    return {"changed": list(plan.changed), "removed": list(plan.removed)}


@method("index.reindex")
def _index_reindex(service: DaemonService, params: Params) -> JSON:
    entry = service.entry(params)
    service.open_project(entry)
    paths = _strings(params, "paths")
    for path in paths:
        service.scheduler.enqueue_path(entry.key, path, Priority.QUERY)
    drained = service.worker.wait_drained(WAIT_TIMEOUT_S) if _flag(params, "wait") else None
    return {"queued": len(paths), "drained": drained}


@method("index.rebuild")
def _index_rebuild(service: DaemonService, params: Params) -> JSON:
    entry = service.entry(params)
    service.open_project(entry)
    service.runner.start_rebuild(entry.key, entry.root, bulk=_flag(params, "bulk"))
    done = service.runner.join_rebuild(entry.key, WAIT_TIMEOUT_S) if _flag(params, "wait") else None
    return {"started": True, "done": done}


def _hook_project(service: DaemonService, params: Params) -> ResolvedProject:
    project = ResolvedProject(
        _str(params, "key"), Path(_str(params, "root")), _via(_opt_str(params, "via") or "cwd")
    )
    entry = service.registry.get(project.key)
    if entry is None:
        raise RemoteError("not_indexed", project.key)
    service.open_project(entry)
    return project


@method("hook.session_start")
def _hook_session_start(service: DaemonService, params: Params) -> JSON:
    project = _hook_project(service, params)
    loaded = load_config(project.root, strict=False, env=service.env)
    return service.hooks.session_start(project, _str(params, "session_id"), loaded)


@method("hook.prompt")
def _hook_prompt(service: DaemonService, params: Params) -> JSON:
    service.hooks.prompt(
        _hook_project(service, params), _str(params, "session_id"), _str(params, "text")
    )
    return None


@method("hook.tool_event")
def _hook_tool_event(service: DaemonService, params: Params) -> JSON:
    service.hooks.tool_event(
        _hook_project(service, params),
        _str(params, "session_id"),
        _str(params, "tool_name"),
        _opt_str(params, "summary") or "",
        _opt_str(params, "raw_input") or "",
        _opt_str(params, "raw_output") or "",
    )
    return None


@method("hook.stop")
def _hook_stop(service: DaemonService, params: Params) -> JSON:
    service.hooks.stop(_hook_project(service, params), _str(params, "session_id"))
    return None


@method("hook.session_end")
def _hook_session_end(service: DaemonService, params: Params) -> JSON:
    service.hooks.session_end(
        _hook_project(service, params), _str(params, "session_id"), _opt_str(params, "reason")
    )
    return None
