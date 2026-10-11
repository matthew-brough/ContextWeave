import sys
import tomllib
from pathlib import Path

import anyio
import mcp_types as types
import pytest
from agents_support import make_env
from mcp import Client
from mcp.client.session import ClientRequestContext
from mcp.client.stdio import StdioServerParameters
from pydantic import FileUrl, TypeAdapter

from contextweave.agents.base import InstallOpts
from contextweave.agents.codex import CodexAdapter
from contextweave.core.config import Config
from contextweave.core.refs import project_key
from contextweave.core.types import JSON
from contextweave.eval.mcp_adapters import STRUCTURED_ENV
from contextweave.index.indexer import index_project
from contextweave.index.projects import ProjectRegistry

SERVE_INPROC = "from contextweave.cli import main; raise SystemExit(main(['serve', '--inproc']))"
SYMBOLS = {"alpha": "alpha_only", "beta": "beta_only"}


def make_project(base: Path, name: str) -> Path:
    root = base / name
    (root / ".git").mkdir(parents=True)
    (root / "mod.py").write_text(f"def {SYMBOLS[name]}():\n    return 1\n", encoding="utf-8")
    index_project(root, Config())
    return root.resolve()


class Shared:
    """One global Codex registration: the same command, args and env for every project."""

    def __init__(self, base: Path, registered: dict[str, str]) -> None:
        self.env = {**make_env(base), "CW_MODELS__EMBED__PROVIDER": "none"}
        self.registered = registered

    def params(self, cwd: Path) -> StdioServerParameters:
        return StdioServerParameters(
            command=sys.executable,
            args=["-c", SERVE_INPROC],
            env={**self.env, STRUCTURED_ENV: "1"},
            cwd=cwd,
        )


@pytest.fixture
def projects(tmp_path: Path) -> dict[str, Path]:
    return {name: make_project(tmp_path, name) for name in SYMBOLS}


@pytest.fixture
def shared(tmp_path: Path, projects: dict[str, Path]) -> Shared:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = make_env(tmp_path)
    CodexAdapter().install(projects["alpha"], InstallOpts(home, env=env))
    config = tomllib.loads((home / ".codex" / "config.toml").read_text(encoding="utf-8"))
    entry = config["mcp_servers"]["contextweave"]
    assert entry == {"command": "cw", "args": ["serve"]}
    assert not (projects["beta"] / ".mcp.json").exists()
    registry = ProjectRegistry(Path(env["XDG_STATE_HOME"]) / "contextweave")
    for root in projects.values():
        registry.register(root)
    return Shared(tmp_path, {name: project_key(root) for name, root in projects.items()})


def call(
    shared: Shared,
    cwd: Path,
    name: str,
    arguments: dict[str, JSON],
    roots: list[Path] | None = None,
) -> dict[str, JSON]:
    result: list[dict[str, JSON]] = []

    async def list_roots(context: ClientRequestContext) -> types.ListRootsResult:
        return types.ListRootsResult(
            roots=[types.Root(uri=FileUrl(p.as_uri())) for p in roots or []]
        )

    async def main() -> None:
        async with Client(
            shared.params(cwd),
            list_roots_callback=list_roots if roots is not None else None,
            mode="legacy",
        ) as client:
            outcome = await client.call_tool("get_symbol", {"name": name, **arguments})
            assert not outcome.is_error
            result.append(TypeAdapter(dict[str, JSON]).validate_python(outcome.structured_content))

    anyio.run(main)
    return result[0]


def echoed(payload: dict[str, JSON]) -> tuple[str, str]:
    project = payload["project"]
    assert isinstance(project, dict)
    return str(project["key"]), str(project["via"])


def test_cwd_selects_project_per_spawn(shared: Shared, projects: dict[str, Path]) -> None:
    for name, symbol in SYMBOLS.items():
        payload = call(shared, projects[name], symbol, {})
        assert echoed(payload) == (shared.registered[name], "cwd")
        assert payload["found"] is True


def test_cwd_inside_subdirectory_discovers_root(shared: Shared, projects: dict[str, Path]) -> None:
    nested = projects["beta"] / "pkg" / "deep"
    nested.mkdir(parents=True)
    assert echoed(call(shared, nested, SYMBOLS["beta"], {})) == (shared.registered["beta"], "cwd")


def test_project_arg_overrides_cwd(shared: Shared, projects: dict[str, Path]) -> None:
    payload = call(shared, projects["alpha"], SYMBOLS["beta"], {"project": str(projects["beta"])})
    assert echoed(payload) == (shared.registered["beta"], "arg")
    assert payload["found"] is True
    by_key = call(
        shared, projects["alpha"], SYMBOLS["beta"], {"project": shared.registered["beta"]}
    )
    assert echoed(by_key)[0] == shared.registered["beta"]


def test_single_root_beats_cwd(shared: Shared, projects: dict[str, Path]) -> None:
    payload = call(shared, projects["alpha"], SYMBOLS["beta"], {}, roots=[projects["beta"]])
    assert echoed(payload) == (shared.registered["beta"], "roots")


def test_symbol_of_other_project_is_not_found(shared: Shared, projects: dict[str, Path]) -> None:
    payload = call(shared, projects["alpha"], SYMBOLS["beta"], {})
    assert echoed(payload)[0] == shared.registered["alpha"]
    assert payload["found"] is False


def test_multiple_roots_without_arg_is_no_project(
    shared: Shared, projects: dict[str, Path]
) -> None:
    payload = call(shared, projects["alpha"], "x", {}, roots=list(projects.values()))
    errors = payload["errors"]
    assert isinstance(errors, list) and isinstance(errors[0], dict)
    assert errors[0]["code"] == "no_project"


def test_unregistered_cwd_is_not_indexed_never_a_neighbour(shared: Shared, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    payload = call(shared, elsewhere, SYMBOLS["alpha"], {})
    errors = payload["errors"]
    assert isinstance(errors, list) and isinstance(errors[0], dict)
    assert errors[0]["code"] == "not_indexed"
    assert payload["found"] is False
