from dataclasses import replace
from pathlib import Path

from contextweave.core.config import ClassifyConfig, Config, ModelsConfig
from contextweave.eval.mcp_adapters import FULL_PROFILE, PROFILE_ENV
from contextweave.models.calibration import (
    CALIBRATION_FILE_ENV,
    Calibration,
    CalibrationKey,
    save_calibration,
)
from contextweave.server.context import eval_models


def test_nomodel_profile_builds_no_slots() -> None:
    assert eval_models(Config(), {PROFILE_ENV: "nomodel"}) is None
    assert eval_models(Config(), {}) is None


def test_full_profile_builds_slots_with_fitted_temperatures(tmp_path: Path) -> None:
    classify = ClassifyConfig(provider="none", model="m", build_id="b")
    config = replace(Config(), models=ModelsConfig(classify=classify))
    path = tmp_path / "cal.json"
    save_calibration(
        path, Calibration().with_entry(CalibrationKey("m", "b", 1), {"relevance": 2.0})
    )
    slots = eval_models(config, {PROFILE_ENV: FULL_PROFILE, CALIBRATION_FILE_ENV: str(path)})
    assert slots is not None
    assert slots.temperatures == {"relevance": 2.0}
