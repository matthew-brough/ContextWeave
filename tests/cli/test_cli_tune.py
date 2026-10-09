import random
import tomllib
from pathlib import Path

import pytest
from cli_support import cw

from contextweave.core.config import ScoreConfig, load_config
from contextweave.feedback.served import FEATURE_NAMES
from contextweave.index.store import Store

PACKS = 60
PER_PACK = 5


def seed_feedback(project_root: Path, packs: int = PACKS) -> None:
    """Write served items and used signals: the last two items of each pack are used."""
    columns = ", ".join(("pack_id", "ref", "path", *FEATURE_NAMES, "ts"))
    marks = ", ".join("?" * (len(FEATURE_NAMES) + 4))
    rng = random.Random(3)
    with Store.open(project_root) as store, store.write_transaction():
        for pack in range(packs):
            for slot in range(PER_PACK):
                ref = f"cw://x/p{pack}/{slot}"
                features = (
                    0.0,
                    rng.random(),
                    1.0,
                    slot / (PER_PACK - 1),
                    0.0,
                    rng.random(),
                    rng.random(),
                )
                store.execute(
                    f"INSERT INTO served_items({columns}) VALUES ({marks})",
                    (f"pack{pack}", ref, f"f{slot}.py", *features, 1000 + pack),
                )
                if slot >= PER_PACK - 2:
                    store.execute(
                        "INSERT INTO feedback(pack_id, ref, signal, ts) VALUES (?, ?, 'used', 1)",
                        (f"pack{pack}", ref),
                    )


def test_tune_refuses_without_feedback(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, initialised, "tune")
    assert result.code == 1
    assert "0 labelled items" in result.err
    assert "need at least 50" in result.err
    assert not (initialised / ".contextweave" / "local.toml").exists()


def test_tune_dry_run_writes_nothing(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    seed_feedback(initialised)
    result = cw(capsys, env, initialised, "tune", "--dry-run")
    assert result.code == 0, result.err
    assert "score.gate_lexical_match" in result.out
    assert not (initialised / ".contextweave" / "local.toml").exists()


def test_tune_writes_score_table_and_preserves_other_keys(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    seed_feedback(initialised)
    local = initialised / ".contextweave" / "local.toml"
    local.write_text(
        "[budget]\nsearch_max_tokens = 1234\n\n[score]\ngate_norm = 0.9\n", encoding="utf-8"
    )
    result = cw(capsys, env, initialised, "tune")
    assert result.code == 0, result.err
    data = tomllib.loads(local.read_text(encoding="utf-8"))
    assert data["budget"] == {"search_max_tokens": 1234}
    assert data["score"]["gate_norm"] == 0.9
    assert data["score"]["gate_lexical_match"] > ScoreConfig().gate_lexical_match
    loaded = load_config(initialised, strict=True, env=env)
    assert loaded.config.score.gate_lexical_match == data["score"]["gate_lexical_match"]
    assert loaded.config.budget.search_max_tokens == 1234


def test_tune_refuses_invalid_local_toml(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    seed_feedback(initialised)
    local = initialised / ".contextweave" / "local.toml"
    local.write_text("not = [valid", encoding="utf-8")
    result = cw(capsys, env, initialised, "tune")
    assert result.code == 1
    assert local.read_text(encoding="utf-8") == "not = [valid"
