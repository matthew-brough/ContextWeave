import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path

import pytest
from fake_embed import FakeEmbed, config_for, embed_config, hashed_vector, slots_for

from contextweave.core.config import Config
from contextweave.core.types import JSON
from contextweave.index.chunk_tokens import embed_tokenizer, model_slug, tokenizer_file
from contextweave.index.derived import DerivedCache
from contextweave.index.embed import (
    OVERSIZE,
    code_inputs,
    embed_fitting,
    embed_model_id,
    embed_project,
    oversize,
    split_spans,
    truncate_to_tokens,
)
from contextweave.index.indexer import index_project
from contextweave.index.store import Store
from contextweave.index.vectors import CODE, VectorInput, VectorPart, find_space, spaces
from contextweave.models.errors import ModelContextError
from contextweave.models.providers import Device, EmbedKind
from contextweave.models.slots import ModelSlots, default_registries

TWO_FUNCTIONS = (
    "def parse_token(raw):\n    return raw.strip()\n\n\ndef render_page(items):\n    return items\n"
)

OTHER = "def compute_total(prices):\n    return sum(prices)\n"


@pytest.fixture
def cache(tmp_path: Path) -> DerivedCache:
    return DerivedCache(tmp_path / "cache" / "derived.db")


def project(tmp_path: Path, name: str = "proj", text: str = TWO_FUNCTIONS) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "mod.py").write_text(text, encoding="utf-8")
    (root / "other.py").write_text(OTHER, encoding="utf-8")
    return root


def index(root: Path, fake: FakeEmbed, config: Config, cache: DerivedCache) -> int:
    report = index_project(root, config, slots=slots_for(fake, config), cache=cache)
    assert report.embed is not None
    return report.embed.embedded


def test_first_pass_embeds_every_chunk_and_goes_live(tmp_path: Path, cache: DerivedCache) -> None:
    fake = FakeEmbed(embed_config())
    config = config_for(fake.config)
    root = project(tmp_path)
    assert index(root, fake, config, cache) > 0
    with Store.open(root, readonly=True) as store:
        (space,) = spaces(store)
        assert space.kind == CODE and space.status == "live"
        assert store.query(f"SELECT count(*) FROM {space.name}") == store.query(
            "SELECT count(*) FROM chunks"
        )


def test_rerun_without_changes_makes_no_provider_call(tmp_path: Path, cache: DerivedCache) -> None:
    fake = FakeEmbed(embed_config())
    config = config_for(fake.config)
    root = project(tmp_path)
    index(root, fake, config, cache)
    calls = len(fake.calls)
    assert index(root, fake, config, cache) == 0
    assert len(fake.calls) == calls


def test_only_changed_chunk_is_embedded(tmp_path: Path, cache: DerivedCache) -> None:
    fake = FakeEmbed(embed_config())
    config = config_for(fake.config)
    root = project(tmp_path)
    index(root, fake, config, cache)
    fake.calls.clear()
    edited = TWO_FUNCTIONS.replace("raw.strip()", "raw.lstrip().rstrip()")
    (root / "mod.py").write_text(edited, encoding="utf-8")
    assert index(root, fake, config, cache) == 1
    assert len(fake.texts) == 1
    assert "lstrip" in fake.texts[0]
    with Store.open(root, readonly=True) as store:
        (space,) = spaces(store)
        assert (
            store.query(f"SELECT count(*) FROM vectors_meta WHERE space_name = '{space.name}'")[0]
            == store.query("SELECT count(*) FROM chunks")[0]
        )


def test_cache_hit_across_projects_makes_no_provider_call(
    tmp_path: Path, cache: DerivedCache
) -> None:
    fake = FakeEmbed(embed_config())
    config = config_for(fake.config)
    index(project(tmp_path, "one"), fake, config, cache)
    fake.calls.clear()
    assert index(project(tmp_path, "two"), fake, config, cache) == 0
    assert fake.calls == []


def test_model_switch_builds_new_space_then_evicts_old(tmp_path: Path, cache: DerivedCache) -> None:
    first = FakeEmbed(embed_config("fake-a"))
    root = project(tmp_path)
    index(root, first, config_for(first.config), cache)
    second = FakeEmbed(embed_config("fake-b"))
    config = config_for(second.config)
    with Store.open(root) as store:
        report = embed_project(store, slots_for(second, config), cache, config, max_chunks=1)
        assert report.pending > 0 and report.live == ()
        statuses = {space.model_id: space.status for space in spaces(store)}
        assert statuses[embed_model_id(first.config)] == "live"
        assert statuses[embed_model_id(second.config)] == "building"
        done = embed_project(store, slots_for(second, config), cache, config)
        assert done.pending == 0
        (only,) = spaces(store)
        assert only.model_id == embed_model_id(second.config) and only.status == "live"


def test_dim_switch_creates_new_space(tmp_path: Path, cache: DerivedCache) -> None:
    small = FakeEmbed(embed_config(dim=64))
    root = project(tmp_path)
    index(root, small, config_for(small.config), cache)
    wide = FakeEmbed(embed_config(dim=96))
    index(root, wide, config_for(wide.config), cache)
    with Store.open(root, readonly=True) as store:
        (only,) = spaces(store)
        assert only.dim == 96
        assert find_space(store, CODE, embed_model_id(small.config), 64) is None


def test_doc_prefix_change_is_a_new_model_identity() -> None:
    plain = embed_config()
    assert embed_model_id(plain) != embed_model_id(replace(plain, doc_prefix="Doc:\n"))


def test_down_slot_degrades_and_resumes(tmp_path: Path, cache: DerivedCache) -> None:
    fake = FakeEmbed(embed_config(), down=True)
    config = config_for(fake.config)
    root = project(tmp_path)
    report = index_project(root, config, slots=slots_for(fake, config), cache=cache).embed
    assert report is not None
    assert report.degraded and report.pending > 0 and report.live == ()
    fake.down = False
    assert index(root, fake, replace(config), cache) > 0
    with Store.open(root, readonly=True) as store:
        assert [space.status for space in spaces(store)] == ["live"]


def test_input_is_truncated_to_the_budget_but_keyed_by_full_content(
    tmp_path: Path, cache: DerivedCache
) -> None:
    fake = FakeEmbed(embed_config())
    config = config_for(fake.config)
    body = "\n".join(f"    value_{n} = {n}" for n in range(400))
    root = project(tmp_path, text=f"def big():\n{body}\n")
    with Store.open(root) as store, store.writer_lock():
        store.refresh("mod.py")
        report = embed_project(store, slots_for(fake, config), cache, config, max_input_tokens=50)
        sent = max(fake.texts, key=len)
        assert report.pending == 0
        full = store.query("SELECT body FROM fts_code")[0][0]
        assert isinstance(full, str) and len(sent) < len(full)


class WordCount:
    def count(self, text: str) -> int:
        return len(text.split())


def test_truncate_to_tokens_keeps_longest_prefix_within_limit() -> None:
    assert truncate_to_tokens("a b c d e", WordCount(), 3).split() == ["a", "b", "c"]
    assert truncate_to_tokens("a b", WordCount(), 3) == "a b"


def test_split_spans_covers_text_in_capped_spans() -> None:
    text = "a b c d e f g"
    spans = split_spans(text, WordCount(), 3)
    assert "".join(text[start:end] for start, end in spans) == text
    assert all(WordCount().count(text[start:end]) <= 3 for start, end in spans)
    assert len(split_spans(text, WordCount(), 1, max_parts=2)) == 2
    assert split_spans("", WordCount(), 3) == [(0, 0)]


def test_oversize_rekeys_only_inputs_over_budget() -> None:
    short, long = VectorInput(1, "h1", "a b"), VectorInput(2, "h2", "a b c d e")
    assert [item.key for item in oversize([short, long], WordCount(), 3)] == ["h1", "h2" + OVERSIZE]


def test_long_chunk_stores_a_vector_per_part(tmp_path: Path, cache: DerivedCache) -> None:
    fake = FakeEmbed(embed_config())
    config = config_for(fake.config)
    root = project(tmp_path)
    assert index(root, fake, config, cache) > 0
    with Store.open(root) as store:
        fake.calls.clear()
        report = embed_project(store, slots_for(fake, config), cache, config, max_input_tokens=12)
        assert report.pending == 0
        chunks = len(store.query("SELECT id FROM chunks"))
        assert len(fake.texts) > chunks
        full = "".join(str(row[0]) for row in store.query("SELECT body FROM fts_code"))
        assert all(line in "".join(fake.texts) for line in full.splitlines() if line.strip())
        (space,) = spaces(store)
        ((parts,),) = store.query("SELECT count(*) FROM vector_parts")
        assert store.query(f"SELECT count(*) FROM {space.name}")[0] == (parts,)
        assert int(str(parts)) > chunks
        inputs = {item.chunk_id: item.text for item in code_inputs(store)}
        for chunk_id, text in inputs.items():
            rows = store.query(
                "SELECT start_char, end_char FROM vector_parts WHERE chunk_id = ? ORDER BY part",
                (chunk_id,),
            )
            assert "".join(text[int(str(a)) : int(str(b))] for a, b in rows) == text
        fake.calls.clear()
        embed_project(store, slots_for(fake, config), cache, config, max_input_tokens=12)
        assert fake.texts == []


def test_cached_parts_refill_a_fresh_store(tmp_path: Path, cache: DerivedCache) -> None:
    fake = FakeEmbed(embed_config())
    config = config_for(fake.config)
    first, second = project(tmp_path, "one"), project(tmp_path, "two")
    for root in (first, second):
        index(root, fake, config, cache)
        with Store.open(root) as store:
            embed_project(store, slots_for(fake, config), cache, config, max_input_tokens=12)
    with Store.open(first) as one, Store.open(second) as two:
        query = "SELECT count(*) FROM vector_parts"
        assert one.query(query) == two.query(query)
        assert one.query(query)[0] != (0,)


def test_local_tokenizer_json_is_used_when_present(tmp_path: Path) -> None:
    model = "Local-Tok/test-1"
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    definition: dict[str, JSON] = {
        "version": "1.0",
        "added_tokens": [],
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "alpha": 1, "beta": 2},
            "unk_token": "[UNK]",
        },
    }
    target = tokenizer_file(model, env)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(definition), encoding="utf-8")
    assert model_slug(model) in str(target)
    counted = embed_tokenizer(model, env)
    assert counted.count("alpha beta alpha") == 3


def test_missing_tokenizer_json_falls_back_to_estimator(tmp_path: Path) -> None:
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    fallback = embed_tokenizer("absent-model", env)
    assert fallback.count("x" * 70) == 20


def test_annotated_chunks_get_an_annot_space(tmp_path: Path, cache: DerivedCache) -> None:
    fake = FakeEmbed(embed_config())
    config = config_for(fake.config)
    root = project(tmp_path)
    with Store.open(root) as store, store.writer_lock():
        store.refresh("mod.py")
        store.execute(
            "INSERT INTO annotations (chunk_id, content_hash, model_id, prompt_version, purpose,"
            " tags) SELECT id, content_hash, ?, ?, 'strips whitespace from a token', '[]'"
            " FROM chunks LIMIT 1",
            (config.models.annotate.model, config.models.annotate.prompt_version),
        )
        report = embed_project(store, slots_for(fake, config), cache, config)
        assert report.pending == 0
        assert sorted(space.kind for space in spaces(store)) == ["annot", "code"]
        assert all(space.status == "live" for space in spaces(store))


class ContextLimitedEmbed:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls: list[list[str]] = []

    @property
    def devices(self) -> tuple[Device, ...]:
        return (Device.CPU,)

    def set_device(self, device: Device) -> None:
        return None

    def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> list[list[float]]:
        self.calls.append(list(texts))
        longest = max(len(text) for text in texts)
        if longest > self.limit:
            raise ModelContextError("overflow", needed=longest, available=self.limit)
        return [[float(len(text))] for text in texts]


def test_embed_fitting_splits_batch_and_recuts_the_long_item() -> None:
    client = ContextLimitedEmbed(100)
    short, long, tiny = embed_fitting(client, ["short", "x" * 400, "tiny"], EmbedKind.DOC)
    assert short == [VectorPart(0, 5, [5.0])] and tiny == [VectorPart(0, 4, [4.0])]
    assert long[0].start == 0 and long[-1].end == 400
    assert all(a.end == b.start for a, b in pairwise(long))
    assert all(part.end - part.start <= 100 for part in long)
    assert all(part.vector == [float(part.end - part.start)] for part in long)


@dataclass(slots=True)
class OverlapEmbed:
    """Fake embed that sleeps per call and records the peak number of concurrent calls."""

    inner: FakeEmbed
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: int = 0
    peak: int = 0

    @property
    def devices(self) -> tuple[Device, ...]:
        return self.inner.devices

    def set_device(self, device: Device) -> None:
        self.inner.set_device(device)

    def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> list[list[float]]:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
            return self.inner.embed(texts, kind=kind)


def test_batches_overlap_and_land_on_their_chunks(tmp_path: Path, cache: DerivedCache) -> None:
    down = FakeEmbed(embed_config(), down=True)
    config = config_for(down.config)
    root = project(tmp_path)
    for n in range(12):
        (root / f"m{n}.py").write_text(f"def f{n}(x):\n    return x + {n}\n", encoding="utf-8")
    index_project(root, config, slots=slots_for(down, config), cache=cache)
    fake = OverlapEmbed(FakeEmbed(embed_config()))
    registries = default_registries()
    registries.embed.register("fake", lambda ctx: fake)
    slots = ModelSlots.from_config(
        config.models, registries=registries, clock=lambda: 0.0, submit=lambda job: job()
    )
    with Store.open(root) as store:
        report = embed_project(store, slots, cache, config, batch_size=1)
        assert report.pending == 0 and report.degraded == ()
        assert fake.peak > 1
        (space,) = spaces(store)
        inputs = {item.chunk_id: item for item in code_inputs(store)}
        assert report.embedded == len(inputs)
        for chunk_id, item in inputs.items():
            stored = cache.get_parts([item.key], CODE, embed_model_id(config.models.embed), 128)
            ((_, _, vector),) = stored[item.key]
            assert vector == pytest.approx(hashed_vector(item.text, 128), abs=1e-6), chunk_id
        assert store.query(f"SELECT count(*) FROM {space.name}")[0] == (len(inputs),)
