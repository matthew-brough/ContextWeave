from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

from tree_sitter import Language

from contextweave.lang.ts import PACK_PREFIX, load_grammar

type Group = Literal["core", "web", "scripting", "data", "devops", "docs"]
type Profile = Literal["symbols", "defs", "schema", "outline"]

GROUPS: tuple[Group, ...] = ("core", "web", "scripting", "data", "devops", "docs")
PROFILES: tuple[Profile, ...] = ("symbols", "defs", "schema", "outline")

QUERY_DIRS: dict[Profile, str] = {"symbols": "core", "defs": "defs"}


@dataclass(frozen=True, slots=True)
class LangSpec:
    """One registry row."""

    lang: str
    exts: tuple[str, ...]
    group: Group
    profile: Profile
    source: str
    filenames: tuple[str, ...] = ()


LANGS: tuple[LangSpec, ...] = (
    LangSpec("python", (".py", ".pyi"), "core", "symbols", "tree_sitter_python.language"),
    LangSpec(
        "javascript",
        (".js", ".mjs", ".cjs", ".jsx"),
        "core",
        "symbols",
        "tree_sitter_javascript.language",
    ),
    LangSpec(
        "typescript",
        (".ts", ".mts", ".cts"),
        "core",
        "symbols",
        "tree_sitter_typescript.language_typescript",
    ),
    LangSpec("tsx", (".tsx",), "core", "symbols", "tree_sitter_typescript.language_tsx"),
    LangSpec("go", (".go",), "core", "symbols", "tree_sitter_go.language"),
    LangSpec("lua", (".lua",), "core", "symbols", "tree_sitter_lua.language"),
    LangSpec("csharp", (".cs", ".csx"), "core", "symbols", "pack:csharp"),
    LangSpec("ruby", (".rb", ".rake", ".gemspec"), "scripting", "defs", "pack:ruby"),
    LangSpec("perl", (".pl", ".pm", ".t"), "scripting", "defs", "pack:perl"),
    LangSpec("r", (".R", ".r"), "scripting", "defs", "pack:r"),
    LangSpec("bash", (".sh", ".bash"), "scripting", "defs", "pack:bash"),
    LangSpec("zsh", (".zsh", ".zshrc"), "scripting", "defs", "pack:zsh"),
    LangSpec("html", (".html", ".htm"), "web", "outline", "pack:html"),
    LangSpec("css", (".css",), "web", "defs", "pack:css"),
    LangSpec("scss", (".scss",), "web", "defs", "pack:scss"),
    LangSpec("json", (".json",), "data", "schema", "pack:json"),
    LangSpec("json5", (".json5",), "data", "schema", "pack:json5"),
    LangSpec("yaml", (".yaml", ".yml"), "data", "schema", "pack:yaml"),
    LangSpec("toml", (".toml",), "data", "schema", "pack:toml"),
    LangSpec("xml", (".xml", ".xsd", ".svg"), "data", "schema", "pack:xml"),
    LangSpec("csv", (".csv",), "data", "schema", "pack:csv"),
    LangSpec("sql", (".sql",), "data", "defs", "pack:sql"),
    LangSpec("graphql", (".graphql", ".gql"), "data", "defs", "pack:graphql"),
    LangSpec("proto", (".proto",), "data", "defs", "pack:proto"),
    LangSpec("terraform", (".tf", ".tfvars", ".hcl"), "devops", "defs", "pack:terraform"),
    LangSpec(
        "dockerfile",
        (".dockerfile",),
        "devops",
        "defs",
        "pack:dockerfile",
        filenames=("Dockerfile", "Containerfile"),
    ),
    LangSpec("markdown", (".md", ".markdown"), "docs", "outline", "pack:markdown"),
)

BY_LANG: dict[str, LangSpec] = {spec.lang: spec for spec in LANGS}
BY_EXT: dict[str, LangSpec] = {ext: spec for spec in LANGS for ext in spec.exts}
BY_FILENAME: dict[str, LangSpec] = {name: spec for spec in LANGS for name in spec.filenames}


def spec_for_path(path: str | Path) -> LangSpec | None:
    """Registry row for a path; exact filename wins over extension, None when unsupported."""
    candidate = Path(path)
    return BY_FILENAME.get(candidate.name) or BY_EXT.get(candidate.suffix)


@cache
def load_language(lang: str) -> Language | None:
    """Grammar for a registry language: dedicated package, then language pack, else None."""
    spec = BY_LANG.get(lang)
    if spec is None:
        return None
    loaded = load_grammar(spec.source)
    if loaded is None and not spec.source.startswith(PACK_PREFIX):
        loaded = load_grammar(f"{PACK_PREFIX}{spec.lang}")
    return loaded


def query_text(spec: LangSpec) -> str | None:
    """Contents of the language's .scm file, None for table-driven profiles."""
    directory = QUERY_DIRS.get(spec.profile)
    return None if directory is None else _read_query(directory, spec.lang)


@cache
def _read_query(directory: str, lang: str) -> str | None:
    resource = files("contextweave.lang").joinpath("queries", directory, f"{lang}.scm")
    return resource.read_text(encoding="utf-8") if resource.is_file() else None
