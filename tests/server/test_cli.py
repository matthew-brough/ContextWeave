import pytest

from contextweave.cli import main


def test_serve_without_inproc_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["serve"]) == 2
    assert "use --inproc" in capsys.readouterr().err
