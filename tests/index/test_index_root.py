from pathlib import Path

from contextweave.index.root import discover_root, ensure_self_ignore


def test_config_file_wins_over_git(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    inner = tmp_path / "pkg"
    (inner / "src").mkdir(parents=True)
    (inner / ".contextweave.toml").write_text("", encoding="utf-8")
    assert discover_root(inner / "src") == inner.resolve()


def test_state_dir_marks_root(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / ".contextweave").mkdir()
    assert discover_root(tmp_path / "a" / "b") == tmp_path.resolve()


def test_outer_config_beats_inner_git(tmp_path: Path) -> None:
    (tmp_path / ".contextweave.toml").write_text("", encoding="utf-8")
    inner = tmp_path / "repo"
    (inner / ".git").mkdir(parents=True)
    assert discover_root(inner) == tmp_path.resolve()


def test_git_dir(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "x" / "y").mkdir(parents=True)
    assert discover_root(tmp_path / "x" / "y") == tmp_path.resolve()


def test_git_file_is_worktree_root(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    worktree = tmp_path / "wt"
    (worktree / "sub").mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: ../.git/worktrees/wt\n", encoding="utf-8")
    assert discover_root(worktree / "sub") == worktree.resolve()


def test_fallback_is_start(tmp_path: Path) -> None:
    start = tmp_path / "plain"
    start.mkdir()
    assert discover_root(start) == start.resolve()


def test_file_start_uses_parent(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    target = tmp_path / "a.py"
    target.write_text("", encoding="utf-8")
    assert discover_root(target) == tmp_path.resolve()


def test_self_ignore_leaves_user_gitignore_alone(tmp_path: Path) -> None:
    user = tmp_path / ".gitignore"
    user.write_text("node_modules\n", encoding="utf-8")
    directory = ensure_self_ignore(tmp_path)
    assert (directory / ".gitignore").read_text(encoding="utf-8") == "*\n"
    assert user.read_text(encoding="utf-8") == "node_modules\n"


def test_self_ignore_creates_no_user_gitignore_and_is_idempotent(tmp_path: Path) -> None:
    ensure_self_ignore(tmp_path)
    ensure_self_ignore(tmp_path)
    assert not (tmp_path / ".gitignore").exists()
    assert (tmp_path / ".contextweave" / ".gitignore").read_text(encoding="utf-8") == "*\n"
