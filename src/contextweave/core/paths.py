from collections.abc import Mapping
from pathlib import Path

APP_DIR = "contextweave"
RUN_SUBDIR = "run"


def _home(env: Mapping[str, str]) -> Path:
    home = env.get("HOME")
    return Path(home) if home else Path.home()


def _xdg(env: Mapping[str, str], variable: str, fallback: Path) -> Path:
    value = env.get(variable)
    if value and Path(value).is_absolute():
        return Path(value)
    return fallback


def state_home(env: Mapping[str, str]) -> Path:
    """Return $XDG_STATE_HOME or ~/.local/state."""
    return _xdg(env, "XDG_STATE_HOME", _home(env) / ".local" / "state")


def user_state_dir(env: Mapping[str, str]) -> Path:
    """Return the per-user state directory."""
    return state_home(env) / APP_DIR


def user_config_dir(env: Mapping[str, str]) -> Path:
    """Return the per-user config directory."""
    return _xdg(env, "XDG_CONFIG_HOME", _home(env) / ".config") / APP_DIR


def user_cache_dir(env: Mapping[str, str]) -> Path:
    """Return the per-user cache directory."""
    return _xdg(env, "XDG_CACHE_HOME", _home(env) / ".cache") / APP_DIR


def user_runtime_dir(env: Mapping[str, str]) -> Path:
    """Return the runtime directory, falling back to <state>/run when XDG_RUNTIME_DIR is unset."""
    runtime = env.get("XDG_RUNTIME_DIR")
    if runtime and Path(runtime).is_absolute():
        return Path(runtime) / APP_DIR
    return user_state_dir(env) / RUN_SUBDIR


MEMORY_DB_NAME = "memory.db"


def projects_dir(env: Mapping[str, str]) -> Path:
    """Return the registry directory holding one folder per project key."""
    return user_state_dir(env) / "projects"


def derived_db_path(env: Mapping[str, str]) -> Path:
    """Return the path of the shared derived cache database."""
    return user_cache_dir(env) / "derived.db"
