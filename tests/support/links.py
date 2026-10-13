import tempfile
from pathlib import Path

import pytest


def _can_symlink() -> bool:
    """Windows needs Developer Mode or admin to create symlinks."""
    with tempfile.TemporaryDirectory() as scratch:
        try:
            (Path(scratch) / "link").symlink_to(scratch, target_is_directory=True)
        except OSError:
            return False
    return True


needs_symlinks = pytest.mark.skipif(not _can_symlink(), reason="symlinks need privilege here")
