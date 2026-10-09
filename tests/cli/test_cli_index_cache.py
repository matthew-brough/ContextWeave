from pathlib import Path

import pytest
from cli_support import cw

from contextweave.core.paths import derived_db_path
from contextweave.index.derived import DerivedAnnotation, DerivedCache


def test_index_refreshes(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    cw(capsys, env, initialised, "index")
    (initialised / "new.py").write_text("def delta():\n    return 4\n", encoding="utf-8")
    result = cw(capsys, env, initialised, "index")
    assert result.code == 0, result.err
    assert "1 added" in result.out


def test_index_stage_structure(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    cw(capsys, env, initialised, "index")
    result = cw(capsys, env, initialised, "index", "--stage", "structure")
    assert result.code == 0
    assert "0 added" in result.out
    assert "embedded" not in result.out


def test_index_stage_embed(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, initialised, "index", "--stage", "embed")
    assert result.code == 0, result.err
    assert "embedded" in result.out


def test_index_stage_annotate(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, initialised, "index", "--stage", "annotate")
    assert result.code == 0, result.err
    assert "annotated" in result.out


def test_index_rebuild_swaps_shadow(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    (initialised / "app.py").write_text("def only():\n    return 0\n", encoding="utf-8")
    result = cw(capsys, env, initialised, "index", "--rebuild")
    assert result.code == 0, result.err
    assert not (initialised / ".contextweave" / "index.next.db").exists()
    assert "only" in cw(capsys, env, initialised, "search", "only").out


def test_index_needs_project(
    tmp_path: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cw(capsys, env, tmp_path, "index").code == 1


def test_cache_gc_prunes_unreferenced(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    cache = DerivedCache(derived_db_path(env))
    cache.put_annotation("orphan", "model", "1", DerivedAnnotation("purpose", ("t",)))
    cache.close()
    result = cw(capsys, env, initialised, "cache", "gc")
    assert result.code == 0, result.err
    assert "removed 1 annotations" in result.out


def test_cache_gc_bad_duration(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cw(capsys, env, initialised, "cache", "gc", "--older-than", "soon").code == 2
    assert cw(capsys, env, initialised, "cache", "gc", "--older-than", "30d").code == 0
