import shlex
from pathlib import Path

import pytest

from contextweave.core.config import EmbedConfig, ModelsConfig
from contextweave.models.errors import ModelTransportError
from contextweave.models.providers import Slot
from contextweave.models.weights import (
    WeightRef,
    WeightsRefError,
    ensure_weights,
    launch_expander,
    weights_root,
)

FILE_REF = "org/repo-GGUF/model-Q8_0.gguf"


def test_parse_file_and_repo_refs() -> None:
    assert WeightRef.parse(FILE_REF) == WeightRef("org/repo-GGUF", "model-Q8_0.gguf")
    assert WeightRef.parse("org/repo") == WeightRef("org/repo", None)
    with pytest.raises(WeightsRefError):
        WeightRef.parse("repo-only")


def test_weights_root_defaults_to_cache(tmp_path: Path) -> None:
    env = {"HOME": str(tmp_path), "XDG_CACHE_HOME": str(tmp_path / "cache")}
    assert weights_root(ModelsConfig(), env) == tmp_path / "cache" / "contextweave" / "models"
    custom = ModelsConfig(weights_dir=str(tmp_path / "w"))
    assert weights_root(custom, env) == tmp_path / "w"


def test_expander_substitutes_local_path(tmp_path: Path) -> None:
    (tmp_path / "model-Q8_0.gguf").write_bytes(b"gguf")
    expand = launch_expander((FILE_REF,), tmp_path)
    expanded = shlex.split(expand("llama-server -m {weights}"))
    assert expanded == ["llama-server", "-m", str(tmp_path / "model-Q8_0.gguf")]


def test_expander_refuses_missing_weights(tmp_path: Path) -> None:
    expand = launch_expander((FILE_REF,), tmp_path)
    with pytest.raises(ModelTransportError, match="cw models fetch"):
        expand("llama-server -m {weights}")


def test_expander_without_weights_passes_command_through(tmp_path: Path) -> None:
    assert launch_expander((), tmp_path)("llama-server -m /m.gguf") == "llama-server -m /m.gguf"
    with pytest.raises(ModelTransportError, match="no weights"):
        launch_expander((), tmp_path)("llama-server -m {weights}")


def test_ensure_weights_skips_present_files(tmp_path: Path) -> None:
    (tmp_path / "model-Q8_0.gguf").write_bytes(b"gguf")
    models = ModelsConfig(weights_dir=str(tmp_path), embed=EmbedConfig(weights=(FILE_REF,)))
    fetched: list[str] = []
    results = ensure_weights(models, {}, lambda slot, ref: fetched.append(ref))
    assert fetched == []
    assert [(r.slot, r.fetched, r.error) for r in results] == [(Slot.EMBED, False, None)]


def test_ensure_weights_reports_bad_ref(tmp_path: Path) -> None:
    models = ModelsConfig(weights_dir=str(tmp_path), embed=EmbedConfig(weights=("nope",)))
    [result] = ensure_weights(models, {})
    assert result.error is not None and "org/repo" in result.error
