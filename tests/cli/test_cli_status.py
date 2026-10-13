import json
from pathlib import Path

import pytest
from cli_support import cw

from contextweave.core.types import JSON


def test_status_text(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, initialised, "status")
    assert result.code == 0
    assert "index: generation" in result.out
    assert "files 1" in result.out
    assert "daemon: not running" in result.out


def test_status_json(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    data = json.loads(cw(capsys, env, initialised, "status", "--json").out)
    assert data["project"]["index"]["built"] is True
    assert data["project"]["index"]["counts"]["files"] == 1
    assert data["daemon"] == {"running": False}


def test_status_outside_project(
    tmp_path: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, tmp_path, "status")
    assert result.code == 0
    assert "project: none" in result.out


def test_status_unbuilt_index(
    root: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    from contextweave.core.paths import user_state_dir
    from contextweave.index.projects import ProjectRegistry

    ProjectRegistry(user_state_dir(env)).register(root)
    assert "not built" in cw(capsys, env, root, "status").out


def test_status_text_shows_why_a_slot_has_no_device() -> None:
    from contextweave.cli_commands.status import StatusReport, text_lines

    daemon: dict[str, JSON] = {
        "pid": 1,
        "uptime_s": 1.0,
        "queue": 0,
        "models": [
            {
                "slot": "embed",
                "active": None,
                "skip_rate": 1.0,
                "slow_rate": 0.0,
                "calls_in_window": 3,
                "reason": "no device ready, last start failed: set models.embed.gpu.launch",
            },
            {"slot": "annotate", "active": "cpu", "reason": None},
        ],
    }
    lines = text_lines(StatusReport(None, "none", daemon))
    hint = lines.index("    no device ready, last start failed: set models.embed.gpu.launch")
    assert lines[hint - 1].startswith("  model embed: device None")
    assert sum(line.startswith("    ") for line in lines) == 1
