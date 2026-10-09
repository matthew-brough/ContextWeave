import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

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
from contextweave.eval.runner import RunReport, render_delta, render_table, run_suite
from contextweave.eval.suite import SUITES_DIR, SuiteError, load_suite

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


@dataclass(frozen=True, slots=True)
class EvalDeps:
    """Injectable collaborators for tests."""

    adapters: Registry[str, AdapterFactory] = field(default_factory=default_adapters)
    suites_dir: Path = SUITES_DIR
    baselines_dir: Path = BASELINES_DIR
    checkout: Checkout | None = None
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


def options_from_args(args: argparse.Namespace) -> EvalOptions:
    """Build options from a parsed `eval` namespace."""
    adapters: list[str] | None = args.adapters
    return EvalOptions(
        suite=args.suite,
        adapters=tuple(adapters) if adapters else (DEFAULT_ADAPTER,),
        ablate=args.ablate,
        profile=args.profile,
        check=args.check,
        write_baseline=args.write_baseline,
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
        checkout: Checkout = wiring.checkout or materialise
        root = checkout(suite, user_cache_dir(wiring.env) / "eval")
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
    _say(render_table(reports))
    for index, report in enumerate(reports):
        if report.ablate is not None:
            _say("")
            _say(render_delta(reports[index - 1].metrics, report.metrics, report.ablate))
    return _baseline_actions(options, wiring, reports)


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
