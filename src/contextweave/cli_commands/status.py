import json
from dataclasses import dataclass, field

from contextweave.cli_commands.common import EXIT_OK, CommandContext, SubparserRegistry, say
from contextweave.core.config import load_config
from contextweave.core.project import ResolvedProject, resolve_project
from contextweave.core.types import JSON
from contextweave.daemon.lifecycle import daemon_status
from contextweave.index.store import Store
from contextweave.index.vectors import spaces, vec_available
from contextweave.memory.savings import savings_summary
from contextweave.memory.store import MemoryStore, memory_db_path

INDEX_TABLES = ("files", "chunks", "symbols", "imports", "annotations")
MEMORY_TABLES = ("sessions", "decisions", "code_areas")


@dataclass(frozen=True, slots=True)
class IndexStatus:
    built: bool
    generation: int = 0
    counts: dict[str, int] = field(default_factory=dict[str, int])
    spaces: dict[str, str] = field(default_factory=dict[str, str])
    degraded: str | None = None

    def to_json(self) -> dict[str, JSON]:
        counts: dict[str, JSON] = {name: value for name, value in self.counts.items()}
        found: dict[str, JSON] = {name: value for name, value in self.spaces.items()}
        return {
            "built": self.built,
            "generation": self.generation,
            "counts": counts,
            "spaces": found,
            "degraded": self.degraded,
        }


@dataclass(frozen=True, slots=True)
class ProjectStatus:
    key: str
    root: str
    index: IndexStatus
    memory: dict[str, int]
    config_issues: int

    def to_json(self) -> dict[str, JSON]:
        memory: dict[str, JSON] = {name: value for name, value in self.memory.items()}
        return {
            "key": self.key,
            "root": self.root,
            "index": self.index.to_json(),
            "memory": memory,
            "config_issues": self.config_issues,
        }


@dataclass(frozen=True, slots=True)
class StatusReport:
    project: ProjectStatus | None
    no_project_reason: str | None
    daemon: dict[str, JSON] | None

    def to_json(self) -> dict[str, JSON]:
        daemon: JSON = self.daemon if self.daemon is not None else {"running": False}
        return {
            "project": self.project.to_json() if self.project is not None else None,
            "no_project_reason": self.no_project_reason,
            "daemon": daemon,
            "vram": None,
        }


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("status", help="Counts, freshness, daemon and model health")
    parser.add_argument("--json", action="store_true", dest="as_json")


def _index_status(project: ResolvedProject) -> IndexStatus:
    if not (project.root / ".contextweave" / "index.db").is_file():
        return IndexStatus(built=False)
    try:
        with Store.open(project.root, readonly=True) as store:
            counts = {
                table: int(str(store.query(f"SELECT COUNT(*) FROM {table}")[0][0]))
                for table in INDEX_TABLES
            }
            found = (
                {space.name: space.status for space in spaces(store)}
                if vec_available(store)
                else {}
            )
            return IndexStatus(True, store.generation, counts, found)
    except Exception as exc:
        return IndexStatus(built=True, degraded=f"{type(exc).__name__}: {exc}")


def _memory_status(project: ResolvedProject, ctx: CommandContext) -> dict[str, int]:
    path = memory_db_path(project.key, ctx.env)
    if not path.is_file():
        return {}
    store = MemoryStore.open(path, project.key)
    try:
        counts: dict[str, int] = {}
        for table in MEMORY_TABLES:
            row: tuple[int] = store.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE project_key = ?", (project.key,)
            ).fetchone()
            counts[table] = row[0]
        counts["saved_tokens"] = sum(item["saved"] for item in savings_summary(store).values())
        return counts
    finally:
        store.close()


def collect(ctx: CommandContext) -> StatusReport:
    """Everything `cw status` reports."""
    daemon = daemon_status(ctx.env)
    resolution = resolve_project({}, [], ctx.cwd, ctx.registry().known_roots())
    if not isinstance(resolution, ResolvedProject):
        return StatusReport(None, resolution.get("message", resolution["code"]), daemon)
    issues = len(load_config(resolution.root, strict=False, env=ctx.env).issues)
    project = ProjectStatus(
        resolution.key,
        str(resolution.root),
        _index_status(resolution),
        _memory_status(resolution, ctx),
        issues,
    )
    return StatusReport(project, None, daemon)


def _pairs(values: dict[str, int] | dict[str, str]) -> str:
    return ", ".join(f"{name} {value}" for name, value in values.items())


def _project_lines(project: ProjectStatus) -> list[str]:
    lines = [f"project {project.key} {project.root}"]
    index = project.index
    if not index.built:
        lines.append("index: not built (run cw init)")
    else:
        lines.append(f"index: generation {index.generation}, {_pairs(index.counts)}")
        if index.spaces:
            lines.append(f"vector spaces: {_pairs(index.spaces)}")
        if index.degraded:
            lines.append(f"index degraded: {index.degraded}")
    if project.memory:
        lines.append(f"memory: {_pairs(project.memory)}")
    if project.config_issues:
        lines.append(f"config issues: {project.config_issues} (see cw config show)")
    return lines


def _slot_lines(daemon: dict[str, JSON]) -> list[str]:
    models = daemon.get("models")
    lines: list[str] = []
    for slot in models if isinstance(models, list) else []:
        if isinstance(slot, dict):
            flag = " WARN" if slot.get("warn") is True else ""
            slow = " SLOW" if slot.get("slow_warn") is True else ""
            lines.append(
                f"  model {slot.get('slot')}: device {slot.get('active')}, skip rate "
                f"{slot.get('skip_rate')}, slow rate {slot.get('slow_rate')}"
                f" over {slot.get('calls_in_window')} calls{flag}{slow}"
            )
    return lines


def text_lines(report: StatusReport) -> list[str]:
    """Human-readable form of the report."""
    lines = (
        _project_lines(report.project)
        if report.project is not None
        else [f"project: none ({report.no_project_reason})"]
    )
    if report.daemon is None:
        lines.append("daemon: not running")
    else:
        daemon = report.daemon
        lines.append(
            f"daemon: pid {daemon.get('pid')} up {daemon.get('uptime_s')}s queue {daemon.get('queue')}"
        )
        lines.extend(_slot_lines(daemon))
    lines.append("vram: unavailable")
    return lines


def run(as_json: bool, ctx: CommandContext) -> int:
    report = collect(ctx)
    if as_json:
        say(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        for line in text_lines(report):
            say(line)
    return EXIT_OK
