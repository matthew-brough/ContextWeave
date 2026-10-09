import io
import re
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path

from contextweave.eval.suite import Suite

type Checkout = Callable[[Suite, Path], Path]


class CheckoutError(Exception):
    """Raised when a suite's pinned commit cannot be materialised."""


def _git(*args: str, cwd: Path | None = None) -> bytes:
    try:
        completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise CheckoutError(f"git {args[0]} failed: {detail.strip()}") from exc
    return completed.stdout


def _slug(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", repo).strip("_")[-60:]


def materialise(suite: Suite, cache: Path) -> Path:
    """Export suite.commit of suite.repo into <cache>/<slug>/<commit> and return it."""
    source = Path(suite.repo)
    repo = str(source.resolve()) if source.exists() else suite.repo
    target = cache / _slug(repo) / suite.commit
    if target.is_dir() and any(target.iterdir()):
        return target
    mirror = cache / _slug(repo) / "mirror"
    if not mirror.is_dir():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--no-checkout", repo, str(mirror))
    try:
        archive = _git("archive", "--format=tar", suite.commit, cwd=mirror)
    except CheckoutError:
        _git("fetch", "--quiet", "origin", cwd=mirror)
        archive = _git("archive", "--format=tar", suite.commit, cwd=mirror)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(target, filter="data")
    return target
