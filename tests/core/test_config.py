from pathlib import Path

import pytest

from contextweave.core.config import (
    Config,
    ConfigError,
    Layer,
    config_hash,
    flatten,
    flatten_config,
    leaf_keys,
    load_config,
    unflatten,
)


@pytest.fixture
def xdg(tmp_path: Path) -> dict[str, str]:
    return {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def user_file(xdg: dict[str, str]) -> Path:
    return Path(xdg["XDG_CONFIG_HOME"]) / "contextweave" / "config.toml"


def test_defaults_match_spec(tmp_path: Path, xdg: dict[str, str]) -> None:
    config = load_config(tmp_path, env=xdg, strict=True).config
    assert (config.chunk.max_tokens, config.chunk.min_tokens) == (800, 40)
    assert config.fuse.keep == 100
    assert (config.gate.window, config.gate.expand_window) == (24, 8)
    assert (config.gate.card_tokens, config.gate.critical) == (128, 0.8)
    assert (config.expand.min_score, config.expand.max_depth) == (0.6, 2)
    assert config.coverage.max_iterations == 2
    assert config.budget.default_max_tokens == 4000
    assert config.budget.search_max_tokens == 6000
    assert config.budget.omitted_max == 50
    assert config.index.vector_quant == "none"
    assert config.index.ignore == ()
    assert "tests/**" in config.index.test_globs
    assert config.score.gate_answers_query == 0.40
    assert config.score.gate_norm == 0.90
    assert config.score.final_redundancy == -0.20
    assert all(flatten_config(config)[f"pipeline.{s}"] for s in config.pipeline.__slots__)
    assert (config.daemon.idle_minutes, config.daemon.project_idle_minutes) == (30, 30)
    assert config.server.pack_cache == 20
    assert config.hooks.session_start_tokens == 400
    assert (config.models.warn_ms, config.models.deadline_ms) == (1500, 15000)
    assert (config.models.annotate.warn_ms, config.models.annotate.deadline_ms) == (30000, 600000)
    assert config.models.embed.batch_deadline_ms == 300000
    assert (config.models.skip_warn, config.models.slow_warn) == (0.05, 0.25)
    assert config.models.embed.dim == 1536
    assert config.models.classify.devices == ("gpu",)
    assert config.models.classify.model == "kev-latest"
    assert config.models.classify.build_id == "kev-4b-int8-emb8"
    assert config.models.classify.parallel == 8
    assert (config.models.annotate.parallel, config.models.annotate.max_output_tokens) == (4, 512)
    assert config.models.classify.gpu.endpoint == "http://127.0.0.1:8765"
    assert config.models.classify.cpu.endpoint == "http://127.0.0.1:8766"
    assert "TriggerClientEvent" in config.edges.events.emitters
    assert "RegisterNetEvent" in config.edges.events.registrars


def test_default_origins(tmp_path: Path, xdg: dict[str, str]) -> None:
    loaded = load_config(tmp_path, env=xdg, strict=True)
    assert set(loaded.origins) == set(leaf_keys())
    assert all(origin.layer is Layer.DEFAULT for origin in loaded.origins.values())


def test_layer_precedence_and_origins(tmp_path: Path, xdg: dict[str, str]) -> None:
    write(user_file(xdg), "[fuse]\nkeep = 1\n[chunk]\nmax_tokens = 1\nmin_tokens = 1\n")
    project = write(tmp_path / ".contextweave.toml", "[fuse]\nkeep = 2\n[chunk]\nmax_tokens = 2\n")
    local = write(tmp_path / ".contextweave" / "local.toml", "[fuse]\nkeep = 3\n")
    env = {**xdg, "CW_FUSE__KEEP": "4", "CW_GATE__WINDOW": "7"}
    loaded = load_config(tmp_path, env=env, cli={"gate.window": 8}, strict=True)
    assert loaded.config.fuse.keep == 4
    assert loaded.config.gate.window == 8
    assert loaded.config.chunk.max_tokens == 2
    assert loaded.config.chunk.min_tokens == 1
    assert loaded.origins["fuse.keep"].layer is Layer.ENV
    assert loaded.origins["gate.window"].layer is Layer.CLI
    assert loaded.origins["chunk.max_tokens"].layer is Layer.PROJECT
    assert loaded.origins["chunk.max_tokens"].source == str(project)
    assert loaded.origins["chunk.min_tokens"].layer is Layer.USER
    assert loaded.origins["chunk.min_tokens"].source == str(user_file(xdg))
    only_local = load_config(tmp_path, env=xdg, strict=True)
    assert only_local.config.fuse.keep == 3
    assert only_local.origins["fuse.keep"].source == str(local)


def test_no_root_skips_project_layers(tmp_path: Path, xdg: dict[str, str]) -> None:
    write(tmp_path / ".contextweave.toml", "[fuse]\nkeep = 2\n")
    assert load_config(None, env=xdg, strict=True).config.fuse.keep == 100


def test_xdg_fallback_to_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    write(tmp_path / ".config" / "contextweave" / "config.toml", "[fuse]\nkeep = 9\n")
    assert load_config(None, env={}, strict=True).config.fuse.keep == 9


def test_env_parsing(tmp_path: Path, xdg: dict[str, str]) -> None:
    env = {
        **xdg,
        "CW_MODELS__EMBED__DIM": "512",
        "CW_MODELS__SKIP_WARN": "0.1",
        "CW_MODELS__ANNOTATE__THINK": "true",
        "CW_MODELS__EMBED__MODEL": "123",
        "CW_MODELS__CLASSIFY__DEVICES": '["cpu"]',
        "CW_INDEX__VECTOR_QUANT": "int8",
        "CW_UNRELATED": "x",
        "CW_PROJECT_DIR": "/x",
    }
    config = load_config(tmp_path, env=env, strict=True).config
    assert config.models.embed.dim == 512
    assert config.models.skip_warn == 0.1
    assert config.models.annotate.think is True
    assert config.models.embed.model == "123"
    assert config.models.classify.devices == ("cpu",)
    assert config.index.vector_quant == "int8"


def test_cli_string_values_are_parsed(tmp_path: Path, xdg: dict[str, str]) -> None:
    loaded = load_config(tmp_path, env=xdg, cli={"fuse.keep": "12"}, strict=True)
    assert loaded.config.fuse.keep == 12


def test_list_merge_and_replace(tmp_path: Path, xdg: dict[str, str]) -> None:
    write(user_file(xdg), '[index]\nignore = ["a", "b"]\ntest_globs = ["x"]\n')
    write(
        tmp_path / ".contextweave.toml",
        '[index]\nignore = ["b", "c"]\ntest_globs = ["y"]\n[edges.events]\nregistrars = ["Mine"]\n',
    )
    env = {**xdg, "CW_INDEX__IGNORE": '["d"]'}
    config = load_config(tmp_path, env=env, strict=True).config
    assert config.index.ignore == ("a", "b", "c", "d")
    assert config.index.test_globs == ("y",)
    assert config.edges.events.registrars[-1] == "Mine"
    assert config.edges.events.registrars[0] == "RegisterNetEvent"
    assert config.edges.events.emitters[0] == "TriggerEvent"


def test_per_device_tables(tmp_path: Path, xdg: dict[str, str]) -> None:
    write(
        tmp_path / ".contextweave.toml",
        '[models.classify.gpu]\nlaunch = "run-kev"\n',
    )
    gpu = load_config(tmp_path, env=xdg, strict=True).config.models.classify.gpu
    assert gpu.launch == "run-kev"
    assert gpu.endpoint == "http://127.0.0.1:8765"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[fuse]\nkeep = 'many'\n", "fuse.keep"),
        ("[fuse]\nkeep = true\n", "fuse.keep"),
        ("[fuse]\nbogus = 1\n", "fuse.bogus"),
        ("bogus = 1\n", "bogus"),
        ("fuse = 3\n", "fuse"),
        ("[index]\nignore = [1]\n", "index.ignore"),
    ],
)
def test_strict_error_names_key_and_file(
    tmp_path: Path, xdg: dict[str, str], text: str, key: str
) -> None:
    path = write(tmp_path / ".contextweave.toml", text)
    with pytest.raises(ConfigError) as info:
        load_config(tmp_path, env=xdg, strict=True)
    assert info.value.issue.key == key
    assert info.value.issue.file == str(path)
    assert key in str(info.value)
    assert str(path) in str(info.value)


def test_strict_env_and_cli_errors(tmp_path: Path, xdg: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as env_info:
        load_config(tmp_path, env={**xdg, "CW_FUSE__KEEP": "abc"}, strict=True)
    assert env_info.value.issue.key == "fuse.keep"
    with pytest.raises(ConfigError) as cli_info:
        load_config(tmp_path, env=xdg, cli={"nope.x": 1}, strict=True)
    assert cli_info.value.issue.key == "nope.x"


def test_strict_malformed_toml(tmp_path: Path, xdg: dict[str, str]) -> None:
    path = write(tmp_path / ".contextweave.toml", "[fuse\n")
    with pytest.raises(ConfigError) as info:
        load_config(tmp_path, env=xdg, strict=True)
    assert info.value.issue.file == str(path)


def test_lenient_drops_bad_value_and_reports(tmp_path: Path, xdg: dict[str, str]) -> None:
    write(user_file(xdg), "[fuse]\nkeep = 5\n")
    path = write(
        tmp_path / ".contextweave.toml",
        "[fuse]\nkeep = 'many'\nbogus = 1\n[chunk]\nmax_tokens = 900\n",
    )
    loaded = load_config(tmp_path, env=xdg, strict=False)
    assert loaded.config.fuse.keep == 5
    assert loaded.origins["fuse.keep"].layer is Layer.USER
    assert loaded.config.chunk.max_tokens == 900
    assert {(i.code, i.key, i.file) for i in loaded.issues} == {
        ("config_invalid", "fuse.keep", str(path)),
        ("config_invalid", "fuse.bogus", str(path)),
    }


def test_lenient_never_raises_on_broken_file(tmp_path: Path, xdg: dict[str, str]) -> None:
    write(tmp_path / ".contextweave.toml", "[fuse\n")
    loaded = load_config(tmp_path, env=xdg, strict=False)
    assert loaded.config == Config()
    assert len(loaded.issues) == 1


def test_lenient_bad_env_keeps_lower_layer(tmp_path: Path, xdg: dict[str, str]) -> None:
    write(tmp_path / ".contextweave.toml", "[fuse]\nkeep = 5\n")
    loaded = load_config(tmp_path, env={**xdg, "CW_FUSE__KEEP": "abc"}, strict=False)
    assert loaded.config.fuse.keep == 5
    assert loaded.issues[0].key == "fuse.keep"


def test_config_hash_stable_and_sensitive(tmp_path: Path, xdg: dict[str, str]) -> None:
    first = load_config(tmp_path, env=xdg, strict=True).config
    second = load_config(tmp_path, env=xdg, strict=True).config
    assert config_hash(first) == config_hash(second)
    assert len(config_hash(first)) == 64
    changed = load_config(tmp_path, env={**xdg, "CW_FUSE__KEEP": "101"}, strict=True).config
    assert config_hash(changed) != config_hash(first)


def test_flatten_roundtrip() -> None:
    nested = {"a": {"b": 1, "c": {"d": "x"}}, "e": 2}
    flat = flatten(nested)
    assert flat == {"a.b": 1, "a.c.d": "x", "e": 2}
    assert unflatten(flat) == nested
