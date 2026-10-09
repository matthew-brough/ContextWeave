from pathlib import Path

from fake_annotate import FakeAnnotate, annotate_config, config_with, slots_with
from fake_embed import FakeEmbed, embed_config

from contextweave.daemon.models import ModelRuntime
from contextweave.index.derived import DerivedCache
from contextweave.index.indexer import index_project
from contextweave.index.store import Store
from contextweave.index.vectors import ANNOT, spaces


def test_derive_changed_annotates_before_embedding(tmp_path: Path) -> None:
    embed = FakeEmbed(embed_config())
    annotate = FakeAnnotate()
    config = config_with(embed, annotate_config())
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mod.py").write_text("def parse_token(raw):\n    return raw\n", encoding="utf-8")
    index_project(root, config)
    runtime = ModelRuntime(
        slots_with(embed, annotate, config),
        DerivedCache(tmp_path / "derived.db"),
        config,
        None,
    )
    with Store.open(root) as store:
        runtime.derive_changed(store)
        assert annotate.names
        assert [space.status for space in spaces(store) if space.kind == ANNOT] == ["live"]
    annotate.prompts.clear()
    with Store.open(root) as store:
        runtime.derive_changed(store)
    assert annotate.prompts == []
    runtime.close()
