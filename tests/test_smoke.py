import subprocess
import sys

from contextweave.cli import main


def test_cli_runs() -> None:
    assert main([]) == 0
    out = subprocess.run(
        [sys.executable, "-m", "contextweave.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip()
