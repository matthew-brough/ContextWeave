from pathlib import Path

import pytest
from cli_support import cw


def test_config_show_and_origin(
    root: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    (root / ".contextweave.toml").write_text("[budget]\nsearch_max_tokens = 1234\n")
    plain = cw(capsys, env, root, "config", "show")
    assert plain.code == 0
    assert "budget.search_max_tokens = 1234" in plain.out
    assert "#" not in plain.out
    origin = cw(capsys, env, root, "config", "show", "--origin")
    assert "budget.search_max_tokens = 1234  # project" in origin.out
    assert 'models.embed.provider = "none"  # env' in origin.out


def test_config_show_reports_issues(
    root: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    (root / ".contextweave.toml").write_text("bogus = 1\n")
    result = cw(capsys, env, root, "config", "show")
    assert result.code == 0
    assert "bogus" in result.err


def test_languages_lists_registry(
    root: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, root, "languages")
    assert result.code == 0
    assert "python" in result.out
    assert "markdown" in result.out
