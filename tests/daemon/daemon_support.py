import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from contextweave.daemon.client import IpcClient

SOURCES = {
    "alpha.py": "def alpha():\n    return 1\n",
    "beta.py": "def beta():\n    return 2\n",
    "gamma.py": "def gamma():\n    return 3\n",
}


def eventually(check: Callable[[], bool], timeout_s: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


@dataclass(frozen=True, slots=True)
class Project:
    root: Path
    key: str


@dataclass(frozen=True, slots=True)
class RunningDaemon:
    env: dict[str, str]
    client: IpcClient
