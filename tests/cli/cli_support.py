from dataclasses import dataclass
from pathlib import Path

import pytest

from contextweave.cli import main

SOURCE = "def alpha():\n    return 1\n\n\ndef beta():\n    return alpha() + 1\n"


@dataclass(frozen=True, slots=True)
class Run:
    code: int
    out: str
    err: str


def make_env(base: Path) -> dict[str, str]:
    return {
        "HOME": str(base / "home"),
        "XDG_STATE_HOME": str(base / "state"),
        "XDG_CONFIG_HOME": str(base / "config"),
        "XDG_CACHE_HOME": str(base / "cache"),
        "XDG_RUNTIME_DIR": str(base / "run"),
        "CW_MODELS__EMBED__PROVIDER": "none",
        "CW_MODELS__ANNOTATE__PROVIDER": "none",
        "CW_MODELS__CLASSIFY__PROVIDER": "none",
    }


def make_project(base: Path, name: str = "proj") -> Path:
    root = base / name
    (root / ".git").mkdir(parents=True)
    (root / "app.py").write_text(SOURCE, encoding="utf-8")
    (base / "home").mkdir(exist_ok=True)
    return root.resolve()


def cw(capsys: pytest.CaptureFixture[str], env: dict[str, str], cwd: Path, *argv: str) -> Run:
    capsys.readouterr()
    code = main(list(argv), env=env, cwd=cwd)
    captured = capsys.readouterr()
    return Run(code, captured.out, captured.err)
