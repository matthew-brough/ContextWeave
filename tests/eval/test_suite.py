import shutil
import subprocess
from pathlib import Path

import pytest

from contextweave.eval.checkout import CheckoutError, materialise
from contextweave.eval.suite import SuiteError, load_suite, parse_suite
from contextweave.eval.symbols import build_symbol_map

GOOD = """
repo = "."
commit = "abc"

[[case]]
query = "q1"
expect = ["a", "b"]
must_not = ["c"]

[[case]]
query = "q2"
expect = ["d"]
"""


def test_parse_suite() -> None:
    suite = parse_suite("demo", GOOD)
    assert (suite.name, suite.repo, suite.commit) == ("demo", ".", "abc")
    assert suite.cases[0].expect == ("a", "b")
    assert suite.cases[0].must_not == ("c",)
    assert suite.cases[1].must_not == ()


@pytest.mark.parametrize(
    "text",
    [
        "not toml [",
        'repo = "."\ncommit = "x"\n',
        'commit = "x"\n[[case]]\nquery = "q"\nexpect = ["a"]\n',
        'repo = "."\ncommit = "x"\n[[case]]\nquery = "q"\nexpect = []\n',
        'repo = "."\ncommit = "x"\n[[case]]\nquery = "q"\nexpect = [1]\n',
        '[[case]]\nquery = "q"\nexpect = ["a"]\n',
        'repo = "."\ncommit = "x"\n[[case]]\nquery = "q"\nexpect = ["a"]\n[[case]]\nquery = "q"\nexpect = ["b"]\n',
    ],
)
def test_parse_suite_rejects(text: str) -> None:
    with pytest.raises(SuiteError):
        parse_suite("bad", text)


def test_unknown_suite(tmp_path: Path) -> None:
    with pytest.raises(SuiteError):
        load_suite("missing", tmp_path)


def test_symbol_map(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "class A:\n    def f(self) -> None: ...\n\ndef g() -> None: ...\n"
    )
    symbols = build_symbol_map(tmp_path)
    assert symbols.by_path["m.py"] == ("A", "A.f", "g")
    assert symbols.paths_of("A.f") == ("m.py",)
    assert symbols.paths_of("nope") == ()


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True, text=True)
    return done.stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git missing")
def test_materialise_pins_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "m.py").write_text("def one() -> None: ...\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "one")
    pinned = _git(repo, "rev-parse", "HEAD")
    (repo / "m.py").write_text("def two() -> None: ...\n")
    _git(repo, "commit", "-qam", "two")
    suite = parse_suite(
        "t", f'repo = "{repo}"\ncommit = "{pinned}"\n[[case]]\nquery = "q"\nexpect = ["one"]\n'
    )
    root = materialise(suite, tmp_path / "cache")
    assert (root / "m.py").read_text() == "def one() -> None: ...\n"
    assert materialise(suite, tmp_path / "cache") == root
    bad = parse_suite(
        "t", f'repo = "{repo}"\ncommit = "deadbeef"\n[[case]]\nquery = "q"\nexpect = ["one"]\n'
    )
    with pytest.raises(CheckoutError):
        materialise(bad, tmp_path / "cache")


def test_self_suite_is_well_formed() -> None:
    suite = load_suite("self")
    assert len(suite.cases) >= 12
    assert len(suite.commit) == 40
