from pathlib import Path

import pytest
from cli_support import cw


def test_search_prints_pack_and_timings(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, initialised, "search", "alpha")
    assert result.code == 0, result.err
    assert "app.py" in result.out
    assert "stages:" in result.out
    assert "ms" in result.out


def test_search_explain_adds_plan(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, initialised, "search", "alpha", "--explain")
    assert "plan: intent" in result.out
    assert "coverage:" in result.out


def test_search_requires_project(
    tmp_path: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, tmp_path, "search", "alpha")
    assert result.code == 1
    assert "cw search" in result.err
