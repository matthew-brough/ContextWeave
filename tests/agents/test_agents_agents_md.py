from pathlib import Path

from agents_support import read_json, write_json

from contextweave.agents.agents_md import AgentsMdAdapter
from contextweave.agents.base import InstallOpts
from contextweave.agents.blocks import find_block

USER_AGENTS = "# Team rules\n\nBe kind.\n"


def test_install_writes_mcp_and_block(root: Path, home: Path) -> None:
    changes = AgentsMdAdapter().install(root, InstallOpts(home))
    assert {change.path.name for change in changes} == {".mcp.json", "AGENTS.md"}
    assert read_json(root / ".mcp.json") == {
        "mcpServers": {"contextweave": {"command": "cw", "args": ["serve"]}}
    }
    assert find_block((root / "AGENTS.md").read_text(encoding="utf-8")) is not None
    assert not (root / "CLAUDE.md").exists() and not (root / ".claude").exists()


def test_no_hooks() -> None:
    assert AgentsMdAdapter().hook_events() == {}


def test_detect(root: Path) -> None:
    adapter = AgentsMdAdapter()
    assert not adapter.detect(root)
    (root / "AGENTS.md").write_text("x", encoding="utf-8")
    assert adapter.detect(root)


def test_reinstall_is_idempotent(root: Path, home: Path) -> None:
    adapter = AgentsMdAdapter()
    adapter.install(root, InstallOpts(home))
    first = {p.name: p.read_bytes() for p in (root / ".mcp.json", root / "AGENTS.md")}
    assert adapter.install(root, InstallOpts(home)) == []
    assert first == {p.name: p.read_bytes() for p in (root / ".mcp.json", root / "AGENTS.md")}


def test_round_trip_leaves_user_content_byte_identical(root: Path, home: Path) -> None:
    (root / "AGENTS.md").write_text(USER_AGENTS, encoding="utf-8")
    write_json(root / ".mcp.json", {"mcpServers": {"mine": {"command": "m"}}})
    before = {p.name: p.read_bytes() for p in (root / ".mcp.json", root / "AGENTS.md")}
    adapter = AgentsMdAdapter()
    adapter.install(root, InstallOpts(home))
    assert (root / "AGENTS.md").read_text(encoding="utf-8").startswith(USER_AGENTS)
    adapter.uninstall(root, InstallOpts(home))
    assert before == {p.name: p.read_bytes() for p in (root / ".mcp.json", root / "AGENTS.md")}


def test_uninstall_deletes_files_it_created(root: Path, home: Path) -> None:
    adapter = AgentsMdAdapter()
    adapter.install(root, InstallOpts(home))
    adapter.uninstall(root, InstallOpts(home))
    assert not (root / "AGENTS.md").exists() and not (root / ".mcp.json").exists()


def test_dry_run_writes_nothing(root: Path, home: Path) -> None:
    changes = AgentsMdAdapter().install(root, InstallOpts(home, dry_run=True))
    assert len(changes) == 2
    assert not (root / "AGENTS.md").exists() and not (root / ".mcp.json").exists()
