from pathlib import Path

import pytest
from compact_support import pack_item, sources_for

from contextweave.compact.imports import COLLAPSED_PREFIX, collapse_imports
from contextweave.core.refs import ByteAnchor
from contextweave.retrieve.candidates import Candidate

KEY = "0" * 12
HASH = "1" * 12


def block(path: str, start: int, end: int, profile: str = "symbols") -> Candidate:
    return Candidate(
        chunk_id=1,
        path=path,
        start_byte=start,
        end_byte=end,
        anchor=ByteAnchor(start, end),
        profile="symbols",
        group="core",
    )


def ref_for(path: str) -> str:
    return f"cw://{KEY}/{path}#b0-1@{HASH}"


COUNT = 12

CASES = {
    "python": (
        "a/one.py",
        "a/two.py",
        "".join(f"import module_{n}\n" for n in range(COUNT)),
        "def f():\n    return 1\n",
    ),
    "typescript": (
        "a/one.ts",
        "a/two.ts",
        "".join(f"import name{n} from 'module_{n}';\n" for n in range(COUNT)),
        "export function f() { return 1; }\n",
    ),
    "go": (
        "a/one.go",
        "a/two.go",
        "".join(f'import "module_{n}"\n' for n in range(COUNT)),
        "func F() int { return 1 }\n",
    ),
}


@pytest.mark.parametrize("lang", CASES)
def test_identical_import_block_collapses_to_ref_of_first(tmp_path: Path, lang: str) -> None:
    first, second, imports, tail = CASES[lang]
    header = "package a\n\n" if lang == "go" else ""
    sources = sources_for(
        tmp_path, {first: header + imports + tail, second: header + imports + "\n" + tail}
    )
    start = len(header)
    end = start + len(imports)
    one = pack_item(block(first, start, end), imports, ref_for(first))
    two = pack_item(block(second, start, end), imports, ref_for(second))
    out = collapse_imports(sources, (one, two))
    assert out[0] == one
    assert out[1].rep == "collapsed"
    assert out[1].text == COLLAPSED_PREFIX + one.ref
    assert out[1].ref == two.ref
    assert len(out[1].text) < len(two.text)


def test_item_with_non_import_code_is_kept(tmp_path: Path) -> None:
    text = "import os\n\n\ndef f():\n    return 1\n"
    sources = sources_for(tmp_path, {"a.py": text, "b.py": text})
    one = pack_item(block("a.py", 0, len(text)), text, ref_for("a.py"))
    two = pack_item(block("b.py", 0, len(text)), text, ref_for("b.py"))
    assert collapse_imports(sources, (one, two)) == (one, two)


def test_new_statement_prevents_collapse(tmp_path: Path) -> None:
    a, b = "import os\nimport sys\n", "import os\nimport re\n"
    sources = sources_for(tmp_path, {"a.py": a, "b.py": b})
    one = pack_item(block("a.py", 0, len(a)), a, ref_for("a.py"))
    two = pack_item(block("b.py", 0, len(b)), b, ref_for("b.py"))
    assert collapse_imports(sources, (one, two)) == (one, two)


def test_statements_may_be_covered_by_several_earlier_items(tmp_path: Path) -> None:
    first = "".join(f"import module_{n}\n" for n in range(COUNT))
    second = "".join(f"import other_{n}\n" for n in range(COUNT))
    third = "".join(f"import module_{n}\nimport other_{n}\n" for n in range(COUNT // 2))
    sources = sources_for(tmp_path, {"a.py": first, "b.py": second, "c.py": third})
    items = tuple(
        pack_item(block(path, 0, len(text)), text, ref_for(path))
        for path, text in (("a.py", first), ("b.py", second), ("c.py", third))
    )
    out = collapse_imports(sources, items)
    assert [item.rep for item in out] == ["source", "source", "collapsed"]
    assert out[2].text == f"{COLLAPSED_PREFIX}{items[0].ref} {items[1].ref}"


def test_relative_imports_do_not_collapse_across_directories(tmp_path: Path) -> None:
    text = "from .x import y\n"
    sources = sources_for(tmp_path, {"a/m.py": text, "b/m.py": text})
    one = pack_item(block("a/m.py", 0, len(text)), text, ref_for("a/m.py"))
    two = pack_item(block("b/m.py", 0, len(text)), text, ref_for("b/m.py"))
    assert collapse_imports(sources, (one, two)) == (one, two)


def test_signature_items_are_never_collapsed(tmp_path: Path) -> None:
    text = "import os\n"
    sources = sources_for(tmp_path, {"a.py": text, "b.py": text})
    one = pack_item(block("a.py", 0, len(text)), text, ref_for("a.py"))
    two = pack_item(block("b.py", 0, len(text)), text, ref_for("b.py"))
    sig = two.with_signature_rep()
    assert collapse_imports(sources, (one, sig)) == (one, sig)
