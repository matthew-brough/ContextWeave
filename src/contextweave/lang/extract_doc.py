import re
from dataclasses import dataclass
from typing import Final

from tree_sitter import Node

from contextweave.lang.model import Import, LangError
from contextweave.lang.parse import ParsedFile
from contextweave.lang.registry import Group, Profile

LANDMARKS = frozenset({"main", "nav", "header", "footer", "section", "article", "aside", "form"})
HEADING_TAGS = {f"h{n}": n for n in range(1, 7)}

HTML_LINK_ATTRS = {
    "script": ("src", "script"),
    "link": ("href", "stylesheet"),
    "img": ("src", "image"),
    "a": ("href", "link"),
    "iframe": ("src", "frame"),
}

_MD_LINK: Final = re.compile(r"(!?)\[[^\]]*\]\(\s*<?([^)\s>]+)")
_SLUG_STRIP: Final = re.compile(r"[^a-z0-9\s-]")
_SLUG_SPACE: Final = re.compile(r"[\s-]+")


@dataclass(frozen=True, slots=True)
class DocNode:
    """One document structure (heading, code block, table, landmark, element); never a symbol."""

    title: str
    slug: str
    level: int
    kind: str
    lang: str
    group: Group
    profile: Profile
    path: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    info: str | None
    parent: str | None


@dataclass(frozen=True, slots=True)
class DocExtraction:
    """Outline nodes, link targets and per-heading ledes of one document."""

    nodes: tuple[DocNode, ...]
    links: tuple[Import, ...]
    ledes: dict[str, str]
    errors: tuple[LangError, ...]


@dataclass(slots=True)
class _Sink:
    nodes: list[DocNode]
    links: list[Import]
    ledes: dict[str, str]


class _Slugger:
    """GitHub-style anchors, deduplicated with a numeric suffix."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def __call__(self, title: str) -> str:
        base = _SLUG_SPACE.sub("-", _SLUG_STRIP.sub("", title.strip().lower())).strip("-")
        base = base or "section"
        count = self._seen.get(base, 0)
        self._seen[base] = count + 1
        return base if count == 0 else f"{base}-{count}"


def extract_doc(parsed: ParsedFile) -> DocExtraction:
    """Walk a document into its outline; never raises."""
    sink = _Sink([], [], {})
    path = str(parsed.path)
    lang = parsed.spec.lang
    try:
        if lang == "markdown":
            _walk_markdown(parsed, parsed.tree.root_node, None, 0, _Slugger(), sink)
        elif lang == "html":
            _walk_html(parsed, parsed.tree.root_node, None, 0, _Slugger(), sink)
        else:
            error = LangError("unsupported_lang", path, f"{lang} has no outline walker")
            return DocExtraction((), (), {}, (error,))
    except Exception as exc:
        error = LangError("outline_failed", path, f"{type(exc).__name__}: {exc}")
        return DocExtraction(tuple(sink.nodes), tuple(sink.links), sink.ledes, (error,))
    return DocExtraction(tuple(sink.nodes), tuple(sink.links), sink.ledes, ())


def _walk_markdown(
    parsed: ParsedFile,
    node: Node,
    parent: str | None,
    level: int,
    slug: _Slugger,
    sink: _Sink,
) -> None:
    current_parent, current_level, current_slug = parent, level, None

    for child in node.named_children:
        if child.type in {"atx_heading", "setext_heading"}:
            title = _heading_title(parsed, child)
            heading_level = _heading_level(child) or level + 1
            current_slug = slug(title)
            sink.nodes.append(
                _doc_node(
                    parsed, child, title, current_slug, heading_level, "heading", None, parent
                )
            )
            current_parent, current_level = current_slug, heading_level

        elif child.type == "section":
            _walk_markdown(parsed, child, current_parent, current_level, slug, sink)

        elif child.type == "fenced_code_block":
            info = _first_text(parsed, child, "info_string")
            title = info or "code"
            sink.nodes.append(
                _doc_node(
                    parsed,
                    child,
                    title,
                    slug(f"code-{title}"),
                    current_level,
                    "code_block",
                    info,
                    current_parent,
                )
            )

        elif child.type in {"pipe_table", "table"}:
            title = _first_text(parsed, child, "pipe_table_header") or "table"
            title = " ".join(title.split())
            sink.nodes.append(
                _doc_node(
                    parsed,
                    child,
                    title,
                    slug("table"),
                    current_level,
                    "table",
                    None,
                    current_parent,
                )
            )

        elif child.type == "paragraph":
            text = " ".join(parsed.text(child).split())
            if current_slug is not None and current_slug not in sink.ledes and text:
                sink.ledes[current_slug] = text
            _collect_markdown_links(parsed, child, sink)

        else:
            _collect_markdown_links(parsed, child, sink)


def _collect_markdown_links(parsed: ParsedFile, node: Node, sink: _Sink) -> None:
    text = parsed.text(node)
    line = node.start_point[0] + 1
    for match in _MD_LINK.finditer(text):
        start = node.start_byte + len(text[: match.start(2)].encode("utf-8"))
        sink.links.append(
            Import(
                module=match.group(2),
                names=(),
                alias=None,
                kind="image" if match.group(1) else "link",
                lang=parsed.spec.lang,
                group=parsed.spec.group,
                profile=parsed.spec.profile,
                start_byte=start,
                end_byte=start + len(match.group(2).encode("utf-8")),
                line=line,
            )
        )


def _heading_level(node: Node) -> int | None:
    for child in node.children:
        if child.type.startswith("atx_h") and child.type.endswith("_marker"):
            return int(child.type[5])
        if child.type == "setext_h1_underline":
            return 1
        if child.type == "setext_h2_underline":
            return 2
    return None


def _heading_title(parsed: ParsedFile, node: Node) -> str:
    for child in node.named_children:
        if child.type in {"inline", "heading_content", "paragraph"}:
            return " ".join(parsed.text(child).split())
    text = parsed.text(node).splitlines()[0]
    return text.strip("#= \t") or "section"


def _first_text(parsed: ParsedFile, node: Node, kind: str) -> str | None:
    for child in node.named_children:
        if child.type == kind:
            return parsed.text(child).strip()
    return None


def _walk_html(
    parsed: ParsedFile,
    node: Node,
    parent: str | None,
    level: int,
    slug: _Slugger,
    sink: _Sink,
) -> None:
    for child in node.named_children:
        if not child.type.endswith("element"):
            _walk_html(parsed, child, parent, level, slug, sink)
            continue

        tag = _tag_name(parsed, child)
        attrs = _attributes(parsed, child)
        _collect_html_links(parsed, tag, attrs, child, sink)

        emitted = _html_node(parsed, child, tag, attrs, level, slug, parent)
        if emitted is None:
            _walk_html(parsed, child, parent, level, slug, sink)
            continue

        sink.nodes.append(emitted)
        if emitted.kind == "heading":
            lede = _following_text(parsed, node, child)
            if lede:
                sink.ledes.setdefault(emitted.slug, lede)
            _walk_html(parsed, child, parent, emitted.level, slug, sink)
        else:
            _walk_html(parsed, child, emitted.slug, emitted.level, slug, sink)


def _html_node(
    parsed: ParsedFile,
    element: Node,
    tag: str | None,
    attrs: dict[str, str],
    level: int,
    slug: _Slugger,
    parent: str | None,
) -> DocNode | None:
    if tag in HEADING_TAGS:
        title = _element_text(parsed, element) or tag
        return _doc_node(
            parsed, element, title, slug(title), HEADING_TAGS[tag], "heading", tag, parent
        )
    if tag in LANDMARKS:
        title = attrs.get("aria-label") or attrs.get("id") or tag
        return _doc_node(
            parsed, element, title, slug(f"{tag}-{title}"), level + 1, "landmark", tag, parent
        )
    if "id" in attrs:
        title = f"{tag}#{attrs['id']}"
        return _doc_node(parsed, element, title, slug(title), level + 1, "element", tag, parent)
    return None


def _collect_html_links(
    parsed: ParsedFile, tag: str | None, attrs: dict[str, str], element: Node, sink: _Sink
) -> None:
    entry = HTML_LINK_ATTRS.get(tag or "")
    if entry is None:
        return
    attribute, kind = entry
    target = attrs.get(attribute)
    if target:
        sink.links.append(
            Import(
                module=target,
                names=(),
                alias=None,
                kind=kind,
                lang=parsed.spec.lang,
                group=parsed.spec.group,
                profile=parsed.spec.profile,
                start_byte=element.start_byte,
                end_byte=element.end_byte,
                line=element.start_point[0] + 1,
            )
        )


def _start_tag(element: Node) -> Node | None:
    return next(
        (c for c in element.named_children if c.type in {"start_tag", "self_closing_tag"}), None
    )


def _tag_name(parsed: ParsedFile, element: Node) -> str | None:
    tag = _start_tag(element)
    if tag is None:
        return None
    name = next((c for c in tag.named_children if c.type == "tag_name"), None)
    return parsed.text(name).lower() if name is not None else None


def _attributes(parsed: ParsedFile, element: Node) -> dict[str, str]:
    tag = _start_tag(element)
    if tag is None:
        return {}
    found: dict[str, str] = {}
    for attribute in tag.named_children:
        if attribute.type != "attribute":
            continue
        name = next((c for c in attribute.named_children if c.type == "attribute_name"), None)
        value = next(
            (
                c
                for c in attribute.named_children
                if c.type in {"quoted_attribute_value", "attribute_value"}
            ),
            None,
        )
        if name is None:
            continue
        text = ""
        if value is not None:
            inner = next((c for c in value.named_children if c.type == "attribute_value"), value)
            text = parsed.text(inner).strip("\"'")
        found[parsed.text(name).lower()] = text
    return found


def _following_text(parsed: ParsedFile, container: Node, after: Node) -> str:
    siblings = list(container.named_children)
    try:
        start = siblings.index(after) + 1
    except ValueError:
        return ""
    for sibling in siblings[start:]:
        if _tag_name(parsed, sibling) in HEADING_TAGS:
            break
        text = _element_text(parsed, sibling)
        if text:
            return text
    return ""


def _element_text(parsed: ParsedFile, element: Node) -> str:
    parts = [parsed.text(c) for c in element.named_children if c.type == "text"]
    return " ".join(" ".join(parts).split())


def _doc_node(
    parsed: ParsedFile,
    node: Node,
    title: str,
    slug: str,
    level: int,
    kind: str,
    info: str | None,
    parent: str | None,
) -> DocNode:
    return DocNode(
        title=title,
        slug=slug,
        level=level,
        kind=kind,
        lang=parsed.spec.lang,
        group=parsed.spec.group,
        profile=parsed.spec.profile,
        path=str(parsed.path),
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        info=info,
        parent=parent,
    )
