import json
from pathlib import Path

from contextweave.core.types import JSON


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> JSON:
    decoded: JSON = json.loads(path.read_text(encoding="utf-8"))
    return decoded


def make_root(base: Path, name: str = "proj") -> Path:
    root = base / name
    (root / ".git").mkdir(parents=True)
    (root / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    return root.resolve()


def make_env(base: Path) -> dict[str, str]:
    return {
        "HOME": str(base / "home"),
        "XDG_STATE_HOME": str(base / "state"),
        "XDG_CONFIG_HOME": str(base / "config"),
        "XDG_CACHE_HOME": str(base / "cache"),
    }
