import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from contextweave.core.config import ClassifyConfig
from contextweave.core.paths import user_cache_dir
from contextweave.core.registry import Registry, UnknownKeyError
from contextweave.eval.adapters import AdapterFactory
from contextweave.eval.baselines import (
    BASELINES_DIR,
    PROFILES,
    BaselineError,
    baseline_from,
    check_regression,
    read_baseline,
    write_baseline,
)
from contextweave.eval.catalog import default_adapters
from contextweave.eval.checkout import Checkout, CheckoutError, materialise
from contextweave.eval.runner import (
    RunReport,
    render_calibration,
    render_delta,
    render_table,
    run_suite,
)
from contextweave.eval.suite import SUITES_DIR, SuiteError, load_suite
from contextweave.eval.variants import (
    DEVICE_CHOICES,
    VARIANT_PROFILE,
    Variant,
    VariantOptions,
    default_variants,
)
from contextweave.models.calibration import (
    CalibrationKey,
    calibration_path,
    load_calibration,
    save_calibration,
)

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2
DEFAULT_SUITE = "self"
DEFAULT_ADAPTER = "cw"


@dataclass(frozen=True, slots=True)
class EvalOptions:
    """Typed arguments of `cw eval`."""

    suite: str = DEFAULT_SUITE
    adapters: tuple[str, ...] = (DEFAULT_ADAPTER,)
    ablate: str | None = None
    profile: str = "nomodel"
    check: bool = False
    write_baseline: bool = False
    variants: tuple[str, ...] = ()
    device: str = "auto"
    model: str | None = None
    build_id: str | None = None
    write_calibration: bool = False


@dataclass(frozen=True, slots=True)
class EvalDeps:
    """Injectable collaborators for tests."""

    adapters: Registry[str, AdapterFactory] = field(default_factory=default_adapters)
    suites_dir: Path = SUITES_DIR
    baselines_dir: Path = BASELINES_DIR
    checkout: Checkout | None = None
    variants: Registry[str, Variant] = field(default_factory=default_variants)
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))


class SubparserRegistry(Protocol):
    """The add_parser surface of an argparse subparsers action."""

    def add_parser(self, name: str, *, help: str) -> argparse.ArgumentParser: ...


def add_eval_parser(subparsers: SubparserRegistry) -> None:
    """Register `eval` on the top-level subparsers."""
    parser = subparsers.add_parser("eval", help="run an evaluation suite")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--adapter", action="append", dest="adapters", metavar="NAME")
    parser.add_argument("--ablate", metavar="STAGE")
    parser.add_argument("--profile", choices=PROFILES, default="nomodel")
    parser.add_argument("--check", action="store_true", help="fail on regression vs baseline")
    parser.add_argument("--write-baseline", action="store_true", dest="write_baseline")
    parser.add_argument("--variant", action="append", dest="variants", metavar="NAME")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--model", metavar="ID")
    parser.add_argument("--build-id", dest="build_id", metavar="ID")
    parser.add_argument("--write-calibration", action="store_true", dest="write_calibration")


def options_from_args(args: argparse.Namespace) -> EvalOptions:
    """Build options from a parsed `eval` namespace."""
    adapters: list[str] | None = args.adapters
    variants: list[str] | None = args.variants
    return EvalOptions(
        suite=args.suite,
        adapters=tuple(adapters) if adapters else (DEFAULT_ADAPTER,),
        ablate=args.ablate,
        profile=args.profile,
        check=args.check,
        write_baseline=args.write_baseline,
        variants=tuple(variants) if variants else (),
        device=args.device,
        model=args.model,
        build_id=args.build_id,
        write_calibration=args.write_calibration,
    )


def _say(message: str) -> None:
    print(message)


def run_eval(options: EvalOptions, deps: EvalDeps | None = None) -> int:
    """Run the suite on each adapter, print tables, apply baseline actions; returns an exit code."""
    wiring = deps if deps is not None else EvalDeps()
    try:
        suite = load_suite(options.suite, wiring.suites_dir)
        for name in options.adapters:
            wiring.adapters.get(name)
        chosen = [wiring.variants.get(name) for name in options.variants]
        checkout: Checkout = wiring.checkout or materialise
        root = (
            Path(suite.repo) if suite.live else checkout(suite, user_cache_dir(wiring.env) / "eval")
        )
    except UnknownKeyError as exc:
        _say(f"unknown adapter: {exc}")
        return EXIT_USAGE
    except (SuiteError, CheckoutError) as exc:
        _say(str(exc))
        return EXIT_USAGE

    reports: list[RunReport] = []
    for name in options.adapters:
        reports.append(
            run_suite(suite, root, name, wiring.adapters, options.profile, None, wiring.env)
        )
        if options.ablate is not None:
            reports.append(
                run_suite(
                    suite, root, name, wiring.adapters, options.profile, options.ablate, wiring.env
                )
            )
    knobs = VariantOptions(options.device, options.model, options.build_id)
    for variant in chosen:
        reports.append(
            run_suite(
                suite,
                root,
                DEFAULT_ADAPTER,
                wiring.adapters,
                VARIANT_PROFILE,
                None,
                {**wiring.env, **variant.env(knobs)},
                variant.label(knobs),
            )
        )
    _say(render_table(reports))
    calibration = render_calibration(reports)
    if calibration:
        _say("")
        _say(calibration)
    if options.write_calibration:
        _write_calibration(options, wiring, reports)
    for index, report in enumerate(reports):
        if report.ablate is not None:
            _say("")
            _say(render_delta(reports[index - 1].metrics, report.metrics, report.ablate))
    return _baseline_actions(options, wiring, reports)


def _write_calibration(options: EvalOptions, deps: EvalDeps, reports: Sequence[RunReport]) -> None:
    fitted = [report for report in reports if report.variant is not None and report.calibration]
    if len(fitted) != 1:
        _say("calibration not written: need exactly one variant run with samples")
        return
    defaults = ClassifyConfig()
    key = CalibrationKey(
        options.model or defaults.model,
        options.build_id if options.build_id is not None else defaults.build_id,
        defaults.question_set_version,
    )
    path = calibration_path(deps.env)
    temperatures = {row.question: row.temperature for row in fitted[0].calibration}
    save_calibration(path, load_calibration(path).with_entry(key, temperatures))
    _say(f"wrote {path}")


def _baseline_actions(options: EvalOptions, deps: EvalDeps, reports: Sequence[RunReport]) -> int:
    subject = next((r for r in reports if r.adapter == DEFAULT_ADAPTER and r.ablate is None), None)
    if subject is None or not (options.check or options.write_baseline):
        return EXIT_OK
    if options.write_baseline:
        path = write_baseline(
            baseline_from(options.suite, options.profile, subject.metrics), deps.baselines_dir
        )
        _say(f"wrote {path}")
    if options.check:
        try:
            baseline = read_baseline(options.suite, options.profile, deps.baselines_dir)
        except BaselineError as exc:
            _say(str(exc))
            return EXIT_USAGE
        failures = check_regression(baseline, subject.metrics)
        for failure in failures:
            _say(f"REGRESSION: {failure}")
        if failures:
            return EXIT_REGRESSION
        _say("baseline check passed")
    return EXIT_OK
