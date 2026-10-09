import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

SCHEME = "cw://"
_HEX12 = re.compile(r"[0-9a-f]{12}")
_BYTE_ANCHOR = re.compile(r"b(\d+)-(\d+)")


class RefError(ValueError):
    """Raised when a ref cannot be built or parsed."""


@dataclass(frozen=True, slots=True)
class SymbolAnchor:
    qualified_name: str


@dataclass(frozen=True, slots=True)
class PartAnchor:
    qualified_name: str
    part: int


@dataclass(frozen=True, slots=True)
class ByteAnchor:
    start: int
    end: int


type Anchor = SymbolAnchor | PartAnchor | ByteAnchor


@dataclass(frozen=True, slots=True)
class Ref:
    """Pointer to a chunk: project key, posix path, anchor, 12-hex content hash."""

    key: str
    path: str
    anchor: Anchor
    content_hash: str

    def __post_init__(self) -> None:
        if not _HEX12.fullmatch(self.key):
            raise RefError(f"bad project key: {self.key!r}")
        if not _HEX12.fullmatch(self.content_hash):
            raise RefError(f"bad content hash: {self.content_hash!r}")
        if not self.path:
            raise RefError("empty path")
        match self.anchor:
            case SymbolAnchor(name) | PartAnchor(name, _):
                if not name:
                    raise RefError("empty qualified name")
            case ByteAnchor(start, end):
                if start < 0 or end < start:
                    raise RefError(f"bad byte range: {start}-{end}")
        if isinstance(self.anchor, PartAnchor) and self.anchor.part < 1:
            raise RefError(f"bad part number: {self.anchor.part}")


def project_key(root: Path) -> str:
    """Return sha256(realpath(root))[:12]."""
    return hashlib.sha256(os.path.realpath(root).encode()).hexdigest()[:12]


def pack_omitted_uri(key: str, pack_id: str) -> str:
    """Return the resource URI listing a pack's overflowed omitted entries."""
    return f"{SCHEME}{key}/pack/{pack_id}/omitted"


def _encode_name(name: str) -> str:
    encoded = quote(name, safe="").replace("~", "%7E")
    if _BYTE_ANCHOR.fullmatch(encoded):
        return "%62" + encoded[1:]
    return encoded


def _encode_anchor(anchor: Anchor) -> str:
    match anchor:
        case SymbolAnchor(name):
            return _encode_name(name)
        case PartAnchor(name, part):
            return f"{_encode_name(name)}~{part}"
        case ByteAnchor(start, end):
            return f"b{start}-{end}"


def encode_ref(ref: Ref) -> str:
    """Render a ref as cw://<key>/<path>#<anchor>@<hash12>."""
    path = quote(ref.path, safe="/")
    return f"{SCHEME}{ref.key}/{path}#{_encode_anchor(ref.anchor)}@{ref.content_hash}"


def _parse_anchor(raw: str) -> Anchor:
    if byte := _BYTE_ANCHOR.fullmatch(raw):
        return ByteAnchor(int(byte[1]), int(byte[2]))
    name, sep, part = raw.rpartition("~")
    if sep:
        if not part.isascii() or not part.isdecimal():
            raise RefError(f"bad part number: {part!r}")
        return PartAnchor(unquote(name, errors="strict"), int(part))
    return SymbolAnchor(unquote(raw, errors="strict"))


def parse_ref(text: str) -> Ref:
    """Parse a ref string; raise RefError on any malformed input."""
    if not text.startswith(SCHEME):
        raise RefError(f"missing {SCHEME} scheme: {text!r}")
    key, slash, rest = text[len(SCHEME) :].partition("/")
    if not slash:
        raise RefError(f"missing path: {text!r}")
    body, at, content_hash = rest.rpartition("@")
    if not at:
        raise RefError(f"missing @hash: {text!r}")
    raw_path, hash_sep, raw_anchor = body.partition("#")
    if not hash_sep:
        raise RefError(f"missing #anchor: {text!r}")
    try:
        return Ref(key, unquote(raw_path, errors="strict"), _parse_anchor(raw_anchor), content_hash)
    except UnicodeDecodeError as exc:
        raise RefError(f"bad percent-encoding: {text!r}") from exc
