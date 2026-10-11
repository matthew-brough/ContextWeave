import re
from pathlib import Path

import contextweave

POINT_ATTRIBUTE = re.compile(r"_point\.(row|column)\b")


def test_no_point_attribute_access() -> None:
    """tree-sitter 0.26 Point.row/.column corrupt refcounts on Python 3.14; index instead."""
    root = Path(contextweave.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in root.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if POINT_ATTRIBUTE.search(line)
    ]
    assert offenders == []
