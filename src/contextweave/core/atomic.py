import os
import secrets
import sys
import time
from pathlib import Path

REPLACE_ATTEMPTS = 10 if sys.platform == "win32" else 1
REPLACE_BACKOFF_S = 0.05
NEW_FILE_MODE = 0o666
BINARY_FLAG = os.O_BINARY if sys.platform == "win32" else 0


def write_atomic(path: Path, data: bytes) -> None:
    """Replace the file in one rename, through any symlink and keeping its mode, so readers
    and an interrupted write never see it half-written."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | BINARY_FLAG
    try:
        with os.fdopen(os.open(temp, flags, NEW_FILE_MODE), "wb") as handle:
            handle.write(data)
        if target.exists():
            temp.chmod(target.stat().st_mode)
        _replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _replace(source: Path, target: Path) -> None:
    """Rename over target; on Windows retry while another process briefly holds it open."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_BACKOFF_S * (attempt + 1))
