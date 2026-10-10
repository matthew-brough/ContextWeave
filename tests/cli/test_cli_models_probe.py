import json
from pathlib import Path

import pytest

from contextweave.cli import main
from contextweave.cli_commands.models_cmd import CandidateSpecError, parse_candidate
from contextweave.models.probe.context import ClassifyCandidate


def test_parse_candidate() -> None:
    parsed = parse_candidate("family=kev,rung=3,build=int8,gpu=http://a:1,cpu=http://b:2", "m")
    assert parsed == ClassifyCandidate("kev", 3, "int8", "m", "http://a:1", "http://b:2")
    with pytest.raises(CandidateSpecError):
        parse_candidate("rung=3", "m")
    with pytest.raises(CandidateSpecError):
        parse_candidate("family=kev,rung=x", "m")
    with pytest.raises(CandidateSpecError):
        parse_candidate("family=kev,rung=1,bogus=1", "m")


def cli_env(base: Path) -> dict[str, str]:
    return {
        "HOME": str(base),
        "XDG_STATE_HOME": str(base / "state"),
        "XDG_CONFIG_HOME": str(base / "config"),
        "XDG_CACHE_HOME": str(base / "cache"),
        "XDG_RUNTIME_DIR": str(base / "run"),
        "CW_MODELS__EMBED__PROVIDER": "none",
        "CW_MODELS__ANNOTATE__PROVIDER": "none",
        "CW_MODELS__CLASSIFY__PROVIDER": "none",
    }


def test_cli_probe_with_nothing_configured_exits_nonzero_and_writes_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = tmp_path / "probe.md"
    code = main(
        ["models", "probe", "--json", "--record", str(record)],
        env=cli_env(tmp_path),
        cwd=tmp_path,
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["probed"] is False
    assert record.read_text(encoding="utf-8").startswith("### T10 model probe")


def test_cli_probe_bad_candidate_is_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["models", "probe", "--candidate", "nope"], env=cli_env(tmp_path), cwd=tmp_path)
    assert code == 2
    assert "candidate" in capsys.readouterr().err


def test_cli_probe_table_for_unreachable_llama_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = cli_env(tmp_path) | {
        "CW_MODELS__EMBED__PROVIDER": "llama_server",
        "CW_MODELS__EMBED__DEVICES": '["cpu"]',
        "CW_MODELS__EMBED__CPU__ENDPOINT": "http://127.0.0.1:9",
    }
    code = main(["models", "probe", "--slot", "embed"], env=env, cwd=tmp_path)
    out = capsys.readouterr().out
    assert code == 1
    assert "embed.served" in out
    assert "unavailable" in out
