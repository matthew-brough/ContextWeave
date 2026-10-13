from collections.abc import Callable
from importlib.resources import files

import pytest

from contextweave.lang import registry
from contextweave.lang.extract_doc import extract_doc
from contextweave.lang.extract_schema import extract_schema
from contextweave.lang.extract_symbols import extract_symbols
from contextweave.lang.parse import ParsedFile, parse_bytes
from contextweave.lang.registry import (
    BY_EXT,
    BY_FILENAME,
    BY_LANG,
    GROUPS,
    LANGS,
    PROFILES,
    load_language,
    query_text,
    spec_for_path,
)
from contextweave.lang.ts import compile_query, load_grammar


def test_twenty_seven_rows_over_six_groups_and_four_profiles() -> None:
    assert len(LANGS) == 27
    assert {s.group for s in LANGS} == set(GROUPS)
    assert {s.profile for s in LANGS} == set(PROFILES)
    assert len(BY_LANG) == 27


def test_group_profile_table() -> None:
    expected = {
        ("core", "symbols"): {"python", "javascript", "typescript", "tsx", "go", "lua", "csharp"},
        ("scripting", "defs"): {"ruby", "perl", "r", "bash", "zsh"},
        ("web", "outline"): {"html"},
        ("web", "defs"): {"css", "scss"},
        ("data", "schema"): {"json", "json5", "yaml", "toml", "xml", "csv"},
        ("data", "defs"): {"sql", "graphql", "proto"},
        ("devops", "defs"): {"terraform", "dockerfile"},
        ("docs", "outline"): {"markdown"},
    }
    actual: dict[tuple[str, str], set[str]] = {}
    for spec in LANGS:
        actual.setdefault((spec.group, spec.profile), set()).add(spec.lang)
    assert actual == expected


def test_no_duplicate_extensions() -> None:
    all_exts = [ext for spec in LANGS for ext in spec.exts]
    assert len(all_exts) == len(set(all_exts)) == len(BY_EXT)


@pytest.mark.parametrize(
    "name", ["Dockerfile", "Containerfile", "build/Dockerfile", "a/b/Containerfile"]
)
def test_exact_filenames_route_to_dockerfile(name: str) -> None:
    spec = spec_for_path(name)
    assert spec is not None
    assert spec.lang == "dockerfile"
    assert set(BY_FILENAME) == {"Dockerfile", "Containerfile"}


@pytest.mark.parametrize(
    ("path", "lang"),
    [
        ("mod.py", "python"),
        ("stub.pyi", "python"),
        ("app.mjs", "javascript"),
        ("app.tsx", "tsx"),
        ("app.ts", "typescript"),
        ("ClientMain.cs", "csharp"),
        ("script.csx", "csharp"),
        ("x.R", "r"),
        ("x.r", "r"),
        ("x.json5", "json5"),
        ("x.dockerfile", "dockerfile"),
    ],
)
def test_extension_routing(path: str, lang: str) -> None:
    spec = spec_for_path(path)
    assert spec is not None
    assert spec.lang == lang


@pytest.mark.parametrize("path", ["notes.xyz", "noextension", "Makefile", "rules.mk", "dockerfile"])
def test_unsupported_path_is_none(path: str) -> None:
    assert spec_for_path(path) is None


def test_make_is_excluded() -> None:
    assert "make" not in BY_LANG
    assert load_language("make") is None


def test_unknown_language_loads_as_none() -> None:
    assert load_language("klingon") is None


def test_every_registered_grammar_loads() -> None:
    for spec in LANGS:
        language = load_language(spec.lang)
        assert language is not None, spec.lang
        assert language.abi_version >= 14, spec.lang


def test_loader_falls_back_from_missing_dedicated_package_to_pack() -> None:
    assert load_grammar("no_such_package.language") is None
    assert load_grammar("pack:csharp") is not None
    assert load_grammar("pack:no_such_language") is None
    assert registry.load_language("python") is not None


def test_symbols_and_defs_have_compiling_queries() -> None:
    for spec in LANGS:
        text = query_text(spec)
        if spec.profile in ("symbols", "defs"):
            language = load_language(spec.lang)
            assert text is not None, spec.lang
            assert language is not None
            assert compile_query(language, text) is not None, spec.lang
        else:
            assert text is None, spec.lang


def test_query_files_ship_as_package_data() -> None:
    queries = files("contextweave.lang").joinpath("queries")
    shipped = sorted(
        f"{d.name}/{f.name}"
        for d in queries.iterdir()
        if d.is_dir()
        for f in d.iterdir()
        if f.name.endswith(".scm")
    )
    expected = sorted(
        [
            f"{'core' if s.profile == 'symbols' else 'defs'}/{s.lang}.scm"
            for s in LANGS
            if s.profile in ("symbols", "defs")
        ]
        + [f"edges/{s.lang}.scm" for s in LANGS if s.profile == "symbols"]
    )
    assert shipped == expected


def test_unsupported_and_no_grammar_are_errors_not_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsupported = parse_bytes("notes.xyz", b"")
    assert unsupported.parsed is None
    assert [e.code for e in unsupported.errors] == ["unsupported_lang"]

    def no_grammar(lang: str) -> None:
        return None

    monkeypatch.setattr("contextweave.lang.parse.load_language", no_grammar)
    missing = parse_bytes("a.py", b"x = 1\n")
    assert missing.parsed is None
    assert [e.code for e in missing.errors] == ["no_grammar"]


def test_every_language_has_a_fixture_yielding_items(
    all_fixture_names: list[str], parse_fixture: Callable[[str], ParsedFile]
) -> None:
    yielded: set[str] = set()
    for name in all_fixture_names:
        parsed = parse_fixture(name)
        if parsed.spec.profile == "schema":
            count = len(extract_schema(parsed).nodes)
        elif parsed.spec.profile == "outline":
            count = len(extract_doc(parsed).nodes)
        else:
            count = len(extract_symbols(parsed).symbols)
        if count:
            yielded.add(parsed.spec.lang)
    assert yielded == set(BY_LANG)
