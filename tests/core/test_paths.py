from pathlib import Path

from contextweave.core.paths import (
    derived_db_path,
    projects_dir,
    user_cache_dir,
    user_config_dir,
    user_runtime_dir,
    user_state_dir,
)


def test_xdg_values_used() -> None:
    env = {
        "HOME": "/h",
        "XDG_STATE_HOME": "/s",
        "XDG_CONFIG_HOME": "/c",
        "XDG_CACHE_HOME": "/k",
        "XDG_RUNTIME_DIR": "/r",
    }
    assert user_state_dir(env) == Path("/s/contextweave")
    assert user_config_dir(env) == Path("/c/contextweave")
    assert user_cache_dir(env) == Path("/k/contextweave")
    assert user_runtime_dir(env) == Path("/r/contextweave")
    assert projects_dir(env) == Path("/s/contextweave/projects")
    assert derived_db_path(env) == Path("/k/contextweave/derived.db")


def test_fallbacks_from_home() -> None:
    env = {"HOME": "/h"}
    assert user_state_dir(env) == Path("/h/.local/state/contextweave")
    assert user_config_dir(env) == Path("/h/.config/contextweave")
    assert user_cache_dir(env) == Path("/h/.cache/contextweave")


def test_runtime_unset_falls_back_under_state() -> None:
    env = {"HOME": "/h", "XDG_STATE_HOME": "/s"}
    assert user_runtime_dir(env) == Path("/s/contextweave/run")


def test_relative_and_empty_values_ignored() -> None:
    env = {"HOME": "/h", "XDG_STATE_HOME": "rel", "XDG_RUNTIME_DIR": ""}
    assert user_state_dir(env) == Path("/h/.local/state/contextweave")
    assert user_runtime_dir(env) == Path("/h/.local/state/contextweave/run")
