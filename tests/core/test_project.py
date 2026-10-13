from pathlib import Path

import pytest

from contextweave.core.errors import ErrorEntry
from contextweave.core.project import ResolvedProject, resolve_project
from contextweave.core.refs import ByteAnchor, Ref, encode_ref, project_key
from contextweave.core.types import JSON


def make_root(base: Path, name: str) -> Path:
    root = base / name
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    return root.resolve()


@pytest.fixture
def alpha(tmp_path: Path) -> Path:
    return make_root(tmp_path, "alpha")


@pytest.fixture
def beta(tmp_path: Path) -> Path:
    return make_root(tmp_path, "beta")


def known_of(*roots: Path) -> dict[str, Path]:
    return {project_key(root): root for root in roots}


def ok(result: ResolvedProject | ErrorEntry) -> ResolvedProject:
    assert isinstance(result, ResolvedProject)
    return result


def err(result: ResolvedProject | ErrorEntry) -> ErrorEntry:
    assert not isinstance(result, ResolvedProject)
    return result


def ref_for(root: Path) -> str:
    return encode_ref(Ref(project_key(root), "a.py", ByteAnchor(0, 1), "0" * 12))


def test_arg_by_root_path(alpha: Path, beta: Path) -> None:
    result = ok(resolve_project({"project": str(alpha)}, [beta], beta, known_of(alpha, beta)))
    assert (result.root, result.via, result.key) == (alpha, "arg", project_key(alpha))


def test_arg_by_subdir_discovers_root(alpha: Path) -> None:
    result = ok(resolve_project({"project": str(alpha / "src")}, [], None, known_of(alpha)))
    assert result.root == alpha


def test_arg_by_key(alpha: Path) -> None:
    result = ok(resolve_project({"project": project_key(alpha)}, [], None, known_of(alpha)))
    assert result.via == "arg"
    assert result.echo() == {"key": project_key(alpha), "root": str(alpha), "via": "arg"}


def test_arg_unknown_key_is_no_project(alpha: Path) -> None:
    error = err(resolve_project({"project": "abcdef012345"}, [alpha], alpha, known_of(alpha)))
    assert error["code"] == "no_project"


def test_arg_missing_path_does_not_fall_through(alpha: Path, tmp_path: Path) -> None:
    error = err(
        resolve_project({"project": str(tmp_path / "nope")}, [alpha], alpha, known_of(alpha))
    )
    assert error["code"] == "no_project"


def test_arg_relative_resolved_against_cwd(alpha: Path, tmp_path: Path) -> None:
    result = ok(resolve_project({"project": "alpha"}, [], tmp_path, known_of(alpha)))
    assert result.root == alpha


def test_arg_beats_path_and_roots(alpha: Path, beta: Path) -> None:
    args: dict[str, JSON] = {"project": str(alpha), "path": str(beta / "src")}
    assert ok(resolve_project(args, [beta], beta, known_of(alpha, beta))).root == alpha


def test_absolute_path_arg(alpha: Path, beta: Path) -> None:
    args: dict[str, JSON] = {"path": str(beta / "src" / "x.py")}
    result = ok(resolve_project(args, [alpha], alpha, known_of(alpha, beta)))
    assert (result.root, result.via) == (beta, "path")


def test_relative_path_arg_skipped(alpha: Path) -> None:
    result = ok(resolve_project({"path": "src/x.py"}, [], alpha, known_of(alpha)))
    assert result.via == "cwd"


def test_ref_key_looked_up_in_registry(alpha: Path, beta: Path) -> None:
    for name in ("ref", "target"):
        result = ok(resolve_project({name: ref_for(beta)}, [alpha], alpha, known_of(alpha, beta)))
        assert (result.root, result.via) == (beta, "ref")


def test_ref_unknown_key(alpha: Path, beta: Path) -> None:
    error = err(resolve_project({"ref": ref_for(beta)}, [], alpha, known_of(alpha)))
    assert error["code"] == "no_project"


def test_target_not_a_ref_is_ignored(alpha: Path) -> None:
    result = ok(resolve_project({"target": "some_symbol"}, [alpha], None, known_of(alpha)))
    assert result.via == "roots"


def test_path_beats_ref(alpha: Path, beta: Path) -> None:
    args: dict[str, JSON] = {"path": str(alpha / "src"), "ref": ref_for(beta)}
    assert ok(resolve_project(args, [], None, known_of(alpha, beta))).via == "path"


def test_single_root(alpha: Path, beta: Path) -> None:
    result = ok(resolve_project({}, [alpha], beta, known_of(alpha, beta)))
    assert (result.root, result.via) == (alpha, "roots")


def test_ambiguous_roots_no_pick_and_no_cwd_fallback(alpha: Path, beta: Path) -> None:
    error = err(resolve_project({}, [alpha, beta], alpha, known_of(alpha, beta)))
    assert error["code"] == "no_project"
    assert error.get("detail") == {"known": sorted([str(alpha), str(beta)])}


def test_cwd_fallback(alpha: Path) -> None:
    result = ok(resolve_project({}, [], alpha / "src", known_of(alpha)))
    assert (result.root, result.via) == (alpha, "cwd")


def test_nothing_gives_no_project_with_known(alpha: Path, beta: Path) -> None:
    error = err(resolve_project({}, [], None, known_of(alpha, beta)))
    assert error["code"] == "no_project"
    assert error.get("detail") == {"known": sorted([str(alpha), str(beta)])}


def test_unregistered_root_not_indexed(alpha: Path) -> None:
    error = err(resolve_project({}, [], alpha, {}))
    assert error["code"] == "not_indexed"
    detail = error.get("detail")
    assert detail is not None
    assert detail["hint"] == "cw init"


def test_root_registered_under_other_spelling(alpha: Path) -> None:
    resolved = ok(resolve_project({}, [], alpha, {"0123456789ab": alpha}))
    assert (resolved.key, resolved.root) == ("0123456789ab", alpha)


def test_garbage_args_never_raise(alpha: Path) -> None:
    args: dict[str, JSON] = {"project": 5, "path": [1], "ref": "cw://zz", "target": None}
    result = resolve_project(args, [], alpha, known_of(alpha))
    assert ok(result).via == "cwd"
