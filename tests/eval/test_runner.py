import argparse
from collections.abc import Sequence
from pathlib import Path

import pytest

from contextweave.core.registry import Registry
from contextweave.eval.adapters import AdapterContext, AdapterFactory, Served
from contextweave.eval.baselines import Baseline, write_baseline
from contextweave.eval.checkout import CheckoutError
from contextweave.eval.cli_entry import (
    EvalDeps,
    EvalOptions,
    add_eval_parser,
    options_from_args,
    run_eval,
)
from contextweave.eval.runner import render_delta, render_table, run_suite
from contextweave.eval.suite import Suite, parse_suite

SUITE = parse_suite(
    "fake",
    'repo = "."\ncommit = "x"\n[[case]]\nquery = "q1"\nexpect = ["a"]\n[[case]]\nquery = "q2"\nexpect = ["b"]\n',
)


class FakeAdapter:
    def __init__(self, context: AdapterContext) -> None:
        self._ablate = context.ablate

    def serve_all(self, queries: Sequence[str]) -> tuple[Served, ...]:
        found = ("a", "b") if self._ablate is None else ("a",)
        return tuple(Served(found, 100, 2.0, 1) for _ in queries)


def _registry() -> Registry[str, AdapterFactory]:
    registry: Registry[str, AdapterFactory] = Registry()
    registry.register("cw", FakeAdapter)
    return registry


def _deps(tmp_path: Path) -> EvalDeps:
    suites = tmp_path / "suites"
    suites.mkdir()
    (suites / "fake.toml").write_text(
        'repo = "."\ncommit = "x"\n[[case]]\nquery = "q1"\nexpect = ["a"]\n[[case]]\nquery = "q2"\nexpect = ["b"]\n'
    )
    return EvalDeps(
        adapters=_registry(),
        suites_dir=suites,
        baselines_dir=tmp_path / "baselines",
        checkout=lambda suite, cache: tmp_path,
        env={"XDG_CACHE_HOME": str(tmp_path / "cache")},
    )


def test_run_suite_and_ablation() -> None:
    full = run_suite(SUITE, Path(), "cw", _registry())
    ablated = run_suite(SUITE, Path(), "cw", _registry(), ablate="lexical")
    assert full.metrics.recall == 1.0
    assert ablated.metrics.recall == 0.5
    delta = render_delta(full.metrics, ablated.metrics, "lexical")
    assert "recall      -50.0 pts" in delta
    table = render_table([full, ablated])
    assert "cw-lexical" in table


def test_run_eval_ablate_prints_delta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run_eval(EvalOptions(suite="fake", ablate="lexical"), _deps(tmp_path))
    out = capsys.readouterr().out
    assert code == 0
    assert "ablation delta: lexical" in out


def test_write_then_check_baseline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deps = _deps(tmp_path)
    assert run_eval(EvalOptions(suite="fake", write_baseline=True), deps) == 0
    assert (tmp_path / "baselines" / "fake.nomodel.json").exists()
    assert run_eval(EvalOptions(suite="fake", check=True), deps) == 0
    assert "baseline check passed" in capsys.readouterr().out


def test_check_fails_on_synthetic_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deps = _deps(tmp_path)
    write_baseline(Baseline("fake", "nomodel", 1.0, 1.0, 50.0), deps.baselines_dir)
    assert run_eval(EvalOptions(suite="fake", check=True), deps) == 1
    assert "REGRESSION: tokens" in capsys.readouterr().out


def test_check_without_baseline_is_usage_error(tmp_path: Path) -> None:
    assert run_eval(EvalOptions(suite="fake", check=True), _deps(tmp_path)) == 2


def test_unknown_adapter_and_suite(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    assert run_eval(EvalOptions(suite="fake", adapters=("nope",)), deps) == 2
    assert run_eval(EvalOptions(suite="absent"), deps) == 2


def test_parser_wiring() -> None:
    parser = argparse.ArgumentParser()
    add_eval_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "eval",
            "--suite",
            "self",
            "--adapter",
            "read",
            "--adapter",
            "cw",
            "--ablate",
            "vector",
            "--check",
        ]
    )
    options = options_from_args(args)
    assert options == EvalOptions("self", ("read", "cw"), "vector", "nomodel", True, False)
    assert options_from_args(parser.parse_args(["eval"])).adapters == ("cw",)


def test_live_suite_serves_repo_without_checkout(tmp_path: Path) -> None:
    seen: list[AdapterContext] = []

    def factory(context: AdapterContext) -> FakeAdapter:
        seen.append(context)
        return FakeAdapter(context)

    def no_checkout(suite: Suite, cache: Path) -> Path:
        raise CheckoutError("live suites must not check out")

    registry: Registry[str, AdapterFactory] = Registry()
    registry.register("cw", factory)
    suite_file = tmp_path / "live.toml"
    suite_file.write_text(
        f'repo = "{tmp_path.as_posix()}"\ncommit = "live"\n[[case]]\nquery = "q1"\nexpect = ["a"]\n'
    )
    deps = EvalDeps(adapters=registry, checkout=no_checkout, env={})
    assert run_eval(EvalOptions(suite=str(suite_file)), deps) == 0
    assert [(context.root, context.live) for context in seen] == [(tmp_path, True)]
