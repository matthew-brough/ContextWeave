import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter

from contextweave.core.config import config_hash, flatten_config, load_config
from contextweave.core.errors import OmittedEntry
from contextweave.core.refs import SCHEME, RefError, parse_ref
from contextweave.core.types import JSON
from contextweave.lang.registry import LANGS, load_language
from contextweave.server.context import ToolContext
from contextweave.server.tools import TOOLS

JSON_MIME = "application/json"
STATIC_NAMES = ("status", "config", "languages")
PACK_TEMPLATE = SCHEME + "{key}/pack/{pack_id}/omitted"
REF_TEMPLATE = SCHEME + "{key}/{+path}#{anchor}@{hash}"

_OMITTED_LIST: TypeAdapter[list[OmittedEntry]] = TypeAdapter(list[OmittedEntry])


@dataclass(frozen=True, slots=True)
class ResourceInfo:
    """A listable resource."""

    uri: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class TemplateInfo:
    """A resource template."""

    uri_template: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class ResourceBody:
    """The text of one resource."""

    uri: str
    mime_type: str
    text: str


def list_resources(ctx: ToolContext) -> list[ResourceInfo]:
    """Status, config and languages for every registered project."""
    found: list[ResourceInfo] = []
    for key in ctx.registry.known_roots():
        found.extend(
            ResourceInfo(f"{SCHEME}{key}/{name}", f"{key} {name}", f"Project {key} {name}")
            for name in STATIC_NAMES
        )
    return found


def list_templates() -> list[TemplateInfo]:
    return [
        TemplateInfo(
            PACK_TEMPLATE, "omitted pack", "Entries that overflowed a response's omitted[]"
        ),
        TemplateInfo(REF_TEMPLATE, "item ref", "Current bytes of a ref returned by a tool"),
    ]


def _scalar(value: object) -> JSON:
    match value:
        case bool() | int() | float() | str():
            return value
        case tuple():
            members = cast(tuple[str, ...], value)  # config tuple leaves are always tuple[str, ...]
            return list(members)
        case _:
            return str(value)


def _dump(value: JSON) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _status(ctx: ToolContext, root: Path, key: str) -> JSON:
    with ctx.open_store(root) as store:
        files = store.files()
        by_profile: dict[str, JSON] = {}
        by_lang: dict[str, JSON] = {}
        for record in files:
            by_profile[record.profile] = _count(by_profile, record.profile)
            by_lang[record.lang or "unknown"] = _count(by_lang, record.lang or "unknown")
        return {
            "key": key,
            "root": str(root),
            "readonly": store.readonly,
            "generation": store.generation,
            "files": len(files),
            "by_profile": by_profile,
            "by_lang": by_lang,
        }


def _count(counts: dict[str, JSON], name: str) -> int:
    current = counts.get(name, 0)
    return (current if isinstance(current, int) else 0) + 1


def _config(root: Path) -> JSON:
    loaded = load_config(root, strict=False)
    values: dict[str, JSON] = {
        name: _scalar(value) for name, value in flatten_config(loaded.config).items()
    }
    origins: dict[str, JSON] = {name: str(origin) for name, origin in loaded.origins.items()}
    issues: list[JSON] = [f"{issue.key}: {issue.message}" for issue in loaded.issues]
    return {
        "hash": config_hash(loaded.config),
        "values": values,
        "origins": origins,
        "issues": issues,
    }


def _languages() -> JSON:
    rows: list[JSON] = [
        {
            "lang": spec.lang,
            "group": spec.group,
            "profile": spec.profile,
            "exts": list(spec.exts),
            "grammar": load_language(spec.lang) is not None,
        }
        for spec in LANGS
    ]
    return rows


def read_resource(ctx: ToolContext, uri: str) -> ResourceBody | None:
    """Body of a cw:// resource, or None when the uri names nothing readable."""
    if not uri.startswith(SCHEME):
        return None
    if "#" in uri:
        return _read_ref(ctx, uri)
    key, _, rest = uri[len(SCHEME) :].partition("/")
    root = ctx.registry.known_roots().get(key)
    if root is None:
        return None
    parts = rest.split("/")
    match parts:
        case ["status"]:
            return ResourceBody(uri, JSON_MIME, _dump(_status(ctx, root, key)))
        case ["config"]:
            return ResourceBody(uri, JSON_MIME, _dump(_config(root)))
        case ["languages"]:
            return ResourceBody(uri, JSON_MIME, _dump(_languages()))
        case ["pack", pack_id, "omitted"]:
            entries = ctx.packs.get(key, pack_id)
            if entries is None:
                return None
            return ResourceBody(uri, JSON_MIME, _OMITTED_LIST.dump_json(list(entries)).decode())
        case _:
            return None


def _read_ref(ctx: ToolContext, uri: str) -> ResourceBody | None:
    try:
        parse_ref(uri)
    except RefError:
        return None
    result = TOOLS.get("expand").run(ctx, {"ref": uri})
    return ResourceBody(uri, JSON_MIME, _dump(result.structured))
