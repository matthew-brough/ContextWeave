from pathlib import Path

from contextweave.core.paths import (
    derived_db_path,
    projects_dir,
    user_cache_dir,
    user_config_dir,
    user_runtime_dir,
    user_state_dir,
)

ANCHOR = Path(__file__).anchor


def at(posix: str) -> Path:
    return Path(ANCHOR, posix.lstrip("/"))


def test_xdg_values_used() -> None:
    env = {
        "HOME": str(at("/h")),
        "XDG_STATE_HOME": str(at("/s")),
        "XDG_CONFIG_HOME": str(at("/c")),
        "XDG_CACHE_HOME": str(at("/k")),
        "XDG_RUNTIME_DIR": str(at("/r")),
    }
    assert user_state_dir(env) == at("/s/contextweave")
    assert user_config_dir(env) == at("/c/contextweave")
    assert user_cache_dir(env) == at("/k/contextweave")
    assert user_runtime_dir(env) == at("/r/contextweave")
    assert projects_dir(env) == at("/s/contextweave/projects")
    assert derived_db_path(env) == at("/k/contextweave/derived.db")


def test_fallbacks_from_home() -> None:
    env = {"HOME": str(at("/h"))}
    assert user_state_dir(env) == at("/h/.local/state/contextweave")
    assert user_config_dir(env) == at("/h/.config/contextweave")
    assert user_cache_dir(env) == at("/h/.cache/contextweave")


def test_runtime_unset_falls_back_under_state() -> None:
    env = {"HOME": str(at("/h")), "XDG_STATE_HOME": str(at("/s"))}
    assert user_runtime_dir(env) == at("/s/contextweave/run")


def test_relative_and_empty_values_ignored() -> None:
    env = {"HOME": str(at("/h")), "XDG_STATE_HOME": "rel", "XDG_RUNTIME_DIR": ""}
    assert user_state_dir(env) == at("/h/.local/state/contextweave")
    assert user_runtime_dir(env) == at("/h/.local/state/contextweave/run")
