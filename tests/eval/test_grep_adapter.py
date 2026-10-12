import shutil
from pathlib import Path

import pytest

from contextweave.eval.adapters import AdapterContext, Served
from contextweave.eval.grep_adapter import GrepAdapter, is_code, parse_rg, rank_files
from contextweave.eval.keywords import keywords
from contextweave.eval.mcp_adapters import merge_served


def test_keywords_drop_stopwords_and_duplicates() -> None:
    assert keywords("How is fuel consumption handled? Fuel!") == ("fuel", "consumption")
    assert keywords("how is it") == ()


def test_merge_served_unions_items_and_sums_costs() -> None:
    merged = merge_served([Served(("a", "b"), 10, 1.0), Served(("b", "c"), 5, 2.0, error="x")])
    assert (merged.items, merged.tokens, merged.latency_ms, merged.error) == (
        ("a", "b", "c"),
        15,
        3.0,
        "x",
    )


def test_rank_prefers_files_matching_more_keywords() -> None:
    output = "a.py\x001:fuel\na.py\x002:fuel\nb.py\x003:fuel consumption\nb.py\x00x:bad\n"
    hits = parse_rg(output, ("fuel", "consumption"))
    assert rank_files(hits) == ["b.py", "a.py"]
    assert hits["b.py"].lines == [(3, "fuel consumption")]


@pytest.mark.skipif(shutil.which("rg") is None, reason="needs ripgrep")
def test_grep_adapter_serves_symbols_in_sed_window(tmp_path: Path) -> None:
    filler = "".join(f"def pad{n}() -> None: ...\n" for n in range(100))
    (tmp_path / "fuel.py").write_text(
        "def burn_fuel(rate: float) -> float:\n    return rate * 2\n" + filler
    )
    (tmp_path / "other.py").write_text("def unrelated() -> None: ...\n")
    served = GrepAdapter(AdapterContext(tmp_path)).serve_all(["how is fuel burned"])[0]
    assert served.error is None
    assert "burn_fuel" in served.items
    assert "pad99" not in served.items
    assert "unrelated" not in served.items
    assert served.tokens > 0


def test_is_code_skips_docs_and_data() -> None:
    assert is_code("./a/client.lua")
    assert not is_code("./docs/SPEC.md")
    assert not is_code("./data.json")
