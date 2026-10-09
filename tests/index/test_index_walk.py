from pathlib import Path

from contextweave.index.walk import walk_project


def write(root: Path, relative: str, text: str = "x = 1\n") -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def paths(root: Path, ignore: tuple[str, ...] = ()) -> list[str]:
    return [item.path for item in walk_project(root, ignore)]


def test_sorted_and_spec_mapping(tmp_path: Path) -> None:
    write(tmp_path, "b.py")
    write(tmp_path, "a/z.md", "# t\n")
    write(tmp_path, "notes.xyz", "plain\n")
    walked = walk_project(tmp_path)
    assert [item.path for item in walked] == ["a/z.md", "b.py", "notes.xyz"]
    assert [item.spec.lang if item.spec else None for item in walked] == [
        "markdown",
        "python",
        None,
    ]


def test_always_skips_git_and_state(tmp_path: Path) -> None:
    write(tmp_path, "keep.py")
    write(tmp_path, ".git/config", "x")
    write(tmp_path, ".contextweave/notes.txt", "x")
    write(tmp_path, "sub/.git", "gitdir: elsewhere\n")
    assert paths(tmp_path) == ["keep.py"]


def test_gitignore_negation_and_dir_only(tmp_path: Path) -> None:
    write(tmp_path, ".gitignore", "*.log\n!keep.log\nbuild/\n")
    write(tmp_path, "a.log")
    write(tmp_path, "keep.log")
    write(tmp_path, "build/out.py")
    write(tmp_path, "src/build", "a file named build\n")
    assert paths(tmp_path) == [".gitignore", "keep.log", "src/build"]


def test_gitignore_anchored(tmp_path: Path) -> None:
    write(tmp_path, ".gitignore", "/top.txt\n")
    write(tmp_path, "top.txt")
    write(tmp_path, "sub/top.txt")
    assert paths(tmp_path) == [".gitignore", "sub/top.txt"]


def test_nested_gitignore_is_scoped(tmp_path: Path) -> None:
    write(tmp_path, "pkg/.gitignore", "gen.py\n/only_here.py\n")
    write(tmp_path, "pkg/gen.py")
    write(tmp_path, "pkg/deep/gen.py")
    write(tmp_path, "pkg/only_here.py")
    write(tmp_path, "pkg/deep/only_here.py")
    write(tmp_path, "other/gen.py")
    assert paths(tmp_path) == ["other/gen.py", "pkg/.gitignore", "pkg/deep/only_here.py"]


def test_nested_negation_overrides_parent(tmp_path: Path) -> None:
    write(tmp_path, ".gitignore", "*.gen.py\n")
    write(tmp_path, "pkg/.gitignore", "!keep.gen.py\n")
    write(tmp_path, "pkg/keep.gen.py")
    write(tmp_path, "pkg/drop.gen.py")
    assert paths(tmp_path) == [".gitignore", "pkg/.gitignore", "pkg/keep.gen.py"]


def test_ignored_dir_is_not_reentered(tmp_path: Path) -> None:
    write(tmp_path, ".gitignore", "vendor/\n!vendor/keep.py\n")
    write(tmp_path, "vendor/keep.py")
    assert paths(tmp_path) == [".gitignore"]


def test_config_ignore_globs(tmp_path: Path) -> None:
    write(tmp_path, "a.py")
    write(tmp_path, "gen/b.py")
    write(tmp_path, "c.snap")
    assert paths(tmp_path, ("gen/", "*.snap")) == ["a.py"]


def test_binary_skipped_by_nul_in_first_8k(tmp_path: Path) -> None:
    write(tmp_path, "text.py")
    (tmp_path / "blob.bin").write_bytes(b"abc\0def")
    (tmp_path / "late.txt").write_bytes(b"a" * 9000 + b"\0")
    assert paths(tmp_path) == ["late.txt", "text.py"]


def test_no_size_cap(tmp_path: Path) -> None:
    (tmp_path / "big.json").write_bytes(b'{"a": "' + b"x" * 3_000_000 + b'"}')
    walked = walk_project(tmp_path)
    assert [item.path for item in walked] == ["big.json"]
    assert walked[0].size > 2_000_000


def test_dir_symlink_not_followed_and_outside_file_skipped(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    write(root, "in.py")
    write(outside, "secret.py")
    write(root, "real/inner.py")
    (root / "linkdir").symlink_to(outside)
    (root / "alias").symlink_to(root / "real")
    (root / "linkfile.py").symlink_to(outside / "secret.py")
    (root / "samefile.py").symlink_to(root / "in.py")
    assert paths(root) == ["in.py", "real/inner.py", "samefile.py"]
