import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from contextweave.core.registry import Registry
from contextweave.core.types import JSON
from contextweave.eval.adapters import AdapterContext, AdapterFactory, ScoreSample, Served
from contextweave.eval.cli_entry import (
    EvalDeps,
    EvalOptions,
    add_eval_parser,
    options_from_args,
    run_eval,
)
from contextweave.eval.mcp_adapters import (
    FULL_PROFILE,
    PROFILE_ENV,
    count_skips,
    extract_samples,
)
from contextweave.eval.runner import calibration_of, render_calibration, render_table, run_suite
from contextweave.eval.suite import parse_suite
from contextweave.eval.variants import (
    BUILD_ENV,
    DEVICES_ENV,
    MODEL_ENV,
    PROVIDER_ENV,
    VariantOptions,
    default_variants,
)
from contextweave.models.calibration import CalibrationKey, load_calibration

SUITE_TEXT = (
    'repo = "."\ncommit = "x"\n[[case]]\nquery = "q1"\nexpect = ["a"]\n'
    '[[case]]\nquery = "q2"\nexpect = ["b"]\n'
)
SUITE = parse_suite("fake", SUITE_TEXT)
seen: list[AdapterContext] = []


class SampleAdapter:
    def __init__(self, context: AdapterContext) -> None:
        seen.append(context)
        self._skip = context.env.get(PROVIDER_ENV) == "none"

    def serve_all(self, queries: Sequence[str]) -> tuple[Served, ...]:
        samples = (
            ()
            if self._skip
            else (ScoreSample("relevance", 0.9, 1.0), ScoreSample("relevance", 0.9, 0.0))
        )
        return tuple(
            Served(("a", "b"), 10, 1.0, 1, samples=samples, skips=1 if self._skip else 0)
            for _ in queries
        )


def registry() -> Registry[str, AdapterFactory]:
    adapters: Registry[str, AdapterFactory] = Registry()
    adapters.register("cw", SampleAdapter)
    return adapters


def deps(tmp_path: Path) -> EvalDeps:
    suites = tmp_path / "suites"
    suites.mkdir()
    (suites / "fake.toml").write_text(SUITE_TEXT)
    return EvalDeps(
        adapters=registry(),
        suites_dir=suites,
        baselines_dir=tmp_path / "baselines",
        checkout=lambda suite, cache: tmp_path,
        env={
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "CW_CALIBRATION_FILE": str(tmp_path / "cal.json"),
        },
    )


def test_variant_registry_has_kev_clef_skip() -> None:
    assert set(default_variants().names()) == {"kev", "clef", "skip"}


def test_kev_env_carries_device_model_and_build() -> None:
    env = default_variants().get("kev").env(VariantOptions("gpu", "m", "b1"))
    assert env == {
        PROVIDER_ENV: "kev",
        DEVICES_ENV: '["gpu"]',
        MODEL_ENV: "m",
        BUILD_ENV: "b1",
    }


def test_auto_device_keeps_the_chain() -> None:
    env = default_variants().get("clef").env(VariantOptions())
    assert env[PROVIDER_ENV] == "clef"
    assert env[DEVICES_ENV] == '["gpu", "cpu"]'


def test_skip_variant_selects_provider_none() -> None:
    assert default_variants().get("skip").env(VariantOptions("cpu")) == {PROVIDER_ENV: "none"}


def test_parser_wires_variant_options() -> None:
    parser = argparse.ArgumentParser()
    add_eval_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "eval",
            "--variant",
            "kev",
            "--variant",
            "skip",
            "--device",
            "cpu",
            "--model",
            "m",
            "--build-id",
            "b",
            "--write-calibration",
        ]
    )
    options = options_from_args(args)
    assert options.variants == ("kev", "skip")
    assert (options.device, options.model, options.build_id) == ("cpu", "m", "b")
    assert options.write_calibration


def test_run_eval_runs_each_variant_with_its_env_and_full_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen.clear()
    code = run_eval(
        EvalOptions(suite="fake", variants=("kev", "skip"), device="gpu"), deps(tmp_path)
    )
    out = capsys.readouterr().out
    assert code == 0
    variant_runs = seen[1:]
    assert [c.env.get(PROVIDER_ENV) for c in variant_runs] == ["kev", "none"]
    assert all(c.profile == "full" for c in variant_runs)
    assert variant_runs[0].env[DEVICES_ENV] == '["gpu"]'
    assert "cw[kev/gpu]" in out
    assert "cw[skip]" in out
    assert "calibration: cw[kev/gpu]" in out


def test_unknown_variant_is_a_usage_error(tmp_path: Path) -> None:
    assert run_eval(EvalOptions(suite="fake", variants=("nope",)), deps(tmp_path)) == 2


def test_write_calibration_stores_fitted_temperatures(tmp_path: Path) -> None:
    wiring = deps(tmp_path)
    options = EvalOptions(
        suite="fake", variants=("kev",), model="m", build_id="b", write_calibration=True
    )
    assert run_eval(options, wiring) == 0
    loaded = load_calibration(tmp_path / "cal.json")
    assert "relevance" in loaded.temperatures(CalibrationKey("m", "b", 1))


def test_skip_counts_reach_the_table() -> None:
    report = run_suite(
        SUITE, Path(), "cw", registry(), "full", None, {PROVIDER_ENV: "none"}, "skip"
    )
    assert report.metrics.skips == 2
    assert report.calibration == ()
    assert "skips" in render_table([report])
    assert render_calibration([report]) == ""


def test_calibration_report_per_question() -> None:
    rows = calibration_of(
        [ScoreSample("relevance", 0.95, 0.0)] * 10 + [ScoreSample("answers_query", 1.0, 1.0)] * 4
    )
    assert [row.question for row in rows] == ["answers_query", "relevance"]
    assert rows[0].meets_target
    assert not rows[1].meets_target


def pack(
    devices: str = "gpu",
    degraded: Sequence[Mapping[str, JSON]] = (),
    wanted: Sequence[str] = ("A",),
) -> JSON:
    return {
        "plan": {"devices": {"classify": devices}},
        "degraded": [dict(entry) for entry in degraded],
        "coverage": {"score": 0.8, "sufficient": True},
        "items": [
            {
                "qualified_name": name,
                "scores": {"answers_query": 0.7, "relevance": 0.6, "structural_value": 0.5},
            }
            for name in ("A", "B")
        ],
    }


def test_samples_extracted_and_labelled_by_expected_membership() -> None:
    samples = extract_samples(pack(), ["A"])
    relevance = [s for s in samples if s.question == "relevance"]
    assert [(s.prob, s.label) for s in relevance] == [(0.6, 1.0), (0.6, 0.0)]
    assert [s.label for s in samples if s.question == "sufficient"] == [1.0]


def test_no_samples_when_slot_is_down_or_a_stage_skipped() -> None:
    assert extract_samples(pack(devices="none"), ["A"]) == ()
    skipped = [{"stage": "gate", "reason": "x"}]
    assert extract_samples(pack(degraded=skipped), ["A"]) == ()
    assert count_skips(pack(degraded=skipped)) == 1
    assert count_skips(pack(degraded=[{"stage": "retrieve", "reason": "vec"}])) == 0


def test_profile_constants_agree_with_the_server() -> None:
    assert PROFILE_ENV == "CW_EVAL_PROFILE"
    assert FULL_PROFILE == "full"
