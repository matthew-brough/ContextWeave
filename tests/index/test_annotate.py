import json
from dataclasses import replace
from pathlib import Path

import pytest
from fake_annotate import FakeAnnotate, annotate_config, config_with, slots_with
from fake_embed import FakeEmbed, embed_config

from contextweave.core.config import Config
from contextweave.index.annotate import (
    MAX_CONSECUTIVE_FAULTS,
    QUEUE_DEPTH,
    AnnotateProgress,
    annotate_lines,
    annotate_project,
)
from contextweave.index.derived import DerivedCache
from contextweave.index.indexer import index_project
from contextweave.index.store import Store
from contextweave.index.vectors import ANNOT, spaces
from contextweave.models.prompt_store import PROMPTS_DIR
from contextweave.models.slots import ModelSlots

SOURCE = (
    "def parse_token(raw):\n    return raw.strip()\n\n\ndef render_page(items):\n    return items\n"
)
OTHER = "def compute_total(prices):\n    return sum(prices)\n"


class Rig:
    def __init__(self, tmp_path: Path) -> None:
        self.embed = FakeEmbed(embed_config())
        self.annotate = FakeAnnotate()
        self.config = config_with(self.embed, annotate_config())
        self.cache = DerivedCache(tmp_path / "cache" / "derived.db")
        self.root = tmp_path / "proj"
        self.root.mkdir()
        (self.root / "mod.py").write_text(SOURCE, encoding="utf-8")
        (self.root / "other.py").write_text(OTHER, encoding="utf-8")

    @property
    def slots(self) -> ModelSlots:
        return slots_with(self.embed, self.annotate, self.config)

    def index(self) -> int:
        report = index_project(self.root, self.config, slots=self.slots, cache=self.cache)
        assert report.annotate is not None
        return report.annotate.annotated

    def annotate_slice(
        self, max_chunks: int | None = None, config: Config | None = None
    ) -> tuple[int, int, int]:
        index_project(self.root, self.config)
        with Store.open(self.root) as store:
            report = annotate_project(
                store, self.slots, self.cache, config or self.config, max_chunks=max_chunks
            )
        return report.annotated, report.cached, report.pending


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def stored(rig: Rig) -> list[tuple[str, str, str]]:
    with Store.open(rig.root, readonly=True) as store:
        rows = store.query("SELECT content_hash, purpose, tags FROM annotations ORDER BY chunk_id")
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]


def chunk_count(rig: Rig) -> int:
    with Store.open(rig.root, readonly=True) as store:
        return int(str(store.query("SELECT count(*) FROM chunks")[0][0]))


def test_first_pass_annotates_every_chunk(rig: Rig) -> None:
    annotated = rig.index()
    assert annotated == chunk_count(rig) > 0
    rows = stored(rig)
    assert len(rows) == annotated
    assert any(
        "parse_token" in purpose.replace(" ", "_") or "parse token" in purpose
        for _, purpose, _ in rows
    )
    assert json.loads(rows[0][2])[-1] == "fake"


def test_rerun_makes_no_provider_call(rig: Rig) -> None:
    rig.index()
    calls = len(rig.annotate.prompts)
    assert rig.index() == 0
    assert len(rig.annotate.prompts) == calls


def test_only_changed_chunk_is_reannotated(rig: Rig) -> None:
    rig.index()
    rig.annotate.prompts.clear()
    (rig.root / "mod.py").write_text(SOURCE.replace("raw.strip()", "raw.lstrip()"), "utf-8")
    assert rig.index() == 1
    assert len(rig.annotate.prompts) == 1
    assert "lstrip" in rig.annotate.prompts[0]
    assert len(stored(rig)) == chunk_count(rig)


def test_cache_hit_across_project_makes_no_call(rig: Rig, tmp_path: Path) -> None:
    rig.index()
    clone = Rig(tmp_path / "again")
    clone.cache = rig.cache
    report = index_project(clone.root, clone.config, slots=clone.slots, cache=clone.cache)
    assert report.annotate is not None
    assert report.annotate.annotated == 0
    assert report.annotate.cached == chunk_count(clone)
    assert clone.annotate.prompts == []


def test_prompt_version_bump_reannotates_everything(rig: Rig, tmp_path: Path) -> None:
    rig.index()
    count = chunk_count(rig)
    rig.annotate.prompts.clear()
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    text = (PROMPTS_DIR / "annotate.toml").read_text(encoding="utf-8")
    (prompts / "annotate.toml").write_text(text.replace("prompt_version = 1", "prompt_version = 2"))
    bumped = replace(rig.config.models.annotate, prompt_version=2)
    config = replace(rig.config, models=replace(rig.config.models, annotate=bumped))
    with Store.open(rig.root) as store:
        report = annotate_project(store, rig.slots, rig.cache, config, prompts_dir=prompts)
    assert report.annotated == count
    assert len(rig.annotate.prompts) == count
    with Store.open(rig.root, readonly=True) as store:
        versions = store.query("SELECT DISTINCT prompt_version FROM annotations")
    assert [row[0] for row in versions] == [2]


def test_missing_prompt_version_degrades(rig: Rig) -> None:
    bumped = replace(rig.config.models.annotate, prompt_version=9)
    config = replace(rig.config, models=replace(rig.config.models, annotate=bumped))
    index_project(rig.root, config)
    with Store.open(rig.root) as store:
        report = annotate_project(store, rig.slots, rig.cache, config)
    assert report.annotated == 0 and report.pending > 0
    assert report.degraded[0]["reason"].startswith("annotate_prompt")


def test_model_change_reannotates_everything(rig: Rig) -> None:
    rig.index()
    count = chunk_count(rig)
    rig.annotate.prompts.clear()
    other = replace(rig.config.models.annotate, model="fake-annotate-b")
    rig.config = replace(rig.config, models=replace(rig.config.models, annotate=other))
    assert rig.index() == count
    assert len(rig.annotate.prompts) == count
    assert len(stored(rig)) == count


def test_malformed_output_degrades_without_raising(rig: Rig) -> None:
    rig.annotate.garbage = {"purpose": 7, "tags": "nope"}
    rig.annotate.garbage_for = "parse_token"
    report = index_project(rig.root, rig.config, slots=rig.slots, cache=rig.cache)
    assert report.annotate is not None
    assert report.annotate.pending == 1
    assert report.annotate.annotated == chunk_count(rig) - 1
    assert [entry["stage"] for entry in report.annotate.degraded] == ["annotate"]
    rig.annotate.garbage = None
    assert rig.index() == 1
    assert len(stored(rig)) == chunk_count(rig)


@pytest.mark.parametrize(
    "garbage",
    [
        {"purpose": "ok", "tags": ["Has Space"]},
        {"purpose": "ok", "tags": ["a", "b", "c", "d", "e", "f"]},
        {"purpose": "x" * 300, "tags": []},
        {"purpose": "ok", "tags": [], "extra": 1},
        {"purpose": "  ", "tags": []},
    ],
)
def test_schema_violations_are_not_stored(rig: Rig, garbage: dict[str, object]) -> None:
    rig.annotate.garbage = json.loads(json.dumps(garbage))
    report = index_project(rig.root, rig.config, slots=rig.slots, cache=rig.cache)
    assert report.annotate is not None
    assert report.annotate.annotated == 0
    assert stored(rig) == []
    assert report.annotate.degraded


def test_all_malformed_stops_after_limit(rig: Rig) -> None:
    for index in range(12):
        (rig.root / f"extra_{index}.py").write_text(f"def fn_{index}():\n    return {index}\n")
    rig.annotate.garbage = {"nope": 1}
    report = index_project(rig.root, rig.config, slots=rig.slots, cache=rig.cache)
    assert report.annotate is not None
    queued = rig.config.models.annotate.parallel * QUEUE_DEPTH - 1
    assert MAX_CONSECUTIVE_FAULTS <= len(rig.annotate.prompts) <= MAX_CONSECUTIVE_FAULTS + queued
    assert report.annotate.pending == chunk_count(rig)


def test_model_down_stops_pass_and_reports_pending(rig: Rig) -> None:
    rig.annotate.down = True
    report = index_project(rig.root, rig.config, slots=rig.slots, cache=rig.cache)
    assert report.annotate is not None
    assert report.annotate.annotated == 0
    assert report.annotate.pending == chunk_count(rig)
    assert len(report.annotate.degraded) == 1
    assert report.embed is not None


def test_provider_none_never_raises(rig: Rig) -> None:
    models = replace(
        rig.config.models, annotate=replace(rig.config.models.annotate, provider="none")
    )
    config = replace(rig.config, models=models)
    report = index_project(
        rig.root, config, slots=slots_with(rig.embed, rig.annotate, config), cache=rig.cache
    )
    assert report.annotate is not None
    assert report.annotate.degraded
    assert stored(rig) == []


def test_resumable_in_slices(rig: Rig) -> None:
    total = chunk_count_after_index(rig)
    first = rig.annotate_slice(max_chunks=1)
    assert first == (1, 0, total - 1)
    assert len(stored(rig)) == 1
    second = rig.annotate_slice(max_chunks=1)
    assert second[0] == 1 and second[2] == total - 2
    last = rig.annotate_slice()
    assert last[2] == 0
    assert len(stored(rig)) == total
    assert len(rig.annotate.prompts) == total


def chunk_count_after_index(rig: Rig) -> int:
    index_project(rig.root, rig.config)
    return chunk_count(rig)


def test_interrupted_pass_resumes_from_stored_rows(rig: Rig) -> None:
    total = chunk_count_after_index(rig)
    seen: list[AnnotateProgress] = []

    class Interrupt(Exception):
        pass

    def stop(progress: AnnotateProgress) -> None:
        seen.append(progress)
        if progress.done == 2:
            raise Interrupt

    with Store.open(rig.root) as store, pytest.raises(Interrupt):
        annotate_project(store, rig.slots, rig.cache, rig.config, on_progress=stop)
    assert len(stored(rig)) == 2
    rig.annotate.prompts.clear()
    assert rig.index() == total - 2
    assert len(rig.annotate.prompts) == total - 2


def test_eta_reported_from_measured_rate(rig: Rig) -> None:
    index_project(rig.root, rig.config)
    ticks = iter(float(n) for n in range(1000))
    seen: list[AnnotateProgress] = []
    with Store.open(rig.root) as store:
        report = annotate_project(
            store,
            rig.slots,
            rig.cache,
            rig.config,
            max_chunks=1,
            on_progress=seen.append,
            clock=lambda: next(ticks),
        )
    assert report.rate == pytest.approx(1.0)
    assert report.pending > 0
    assert report.eta_s == pytest.approx(report.pending)
    assert seen[-1].eta_s == pytest.approx(report.pending)
    assert f"eta {report.pending}s" in annotate_lines(report)[0]


def test_annotations_searchable_via_fts(rig: Rig) -> None:
    rig.index()
    with Store.open(rig.root, readonly=True) as store:
        hits = store.query(
            "SELECT rowid FROM fts_code WHERE fts_code MATCH 'annotation:fake'",
        )
        assert len(hits) == chunk_count(rig)
        body = store.query("SELECT annotation FROM fts_code ORDER BY rowid")[0][0]
    assert "Does" in str(body)


def test_reparse_keeps_fts_annotation_from_cache(rig: Rig) -> None:
    rig.index()
    rig.annotate.prompts.clear()
    (rig.root / "mod.py").write_text(SOURCE + "\n# note\n", encoding="utf-8")
    rig.index()
    with Store.open(rig.root, readonly=True) as store:
        empty = store.query("SELECT count(*) FROM fts_code WHERE annotation = ''")[0][0]
    assert empty == 0


def test_annot_space_built_after_annotate_and_embed(rig: Rig) -> None:
    index_project(rig.root, rig.config)
    with Store.open(rig.root, readonly=True) as store:
        assert spaces(store) == ()
    rig.index()
    with Store.open(rig.root, readonly=True) as store:
        annot = [space for space in spaces(store) if space.kind == ANNOT]
        assert len(annot) == 1 and annot[0].status == "live"
        assert store.query(f"SELECT count(*) FROM {annot[0].name}")[0][0] == chunk_count(rig)


class CharTokenizer:
    def count(self, text: str) -> int:
        return len(text)


def test_oversized_chunk_code_is_cut_to_budget(rig: Rig) -> None:
    tail = "TAIL_MARKER"
    body = "\n".join(f"    step_{n} = {n}" for n in range(200))
    (rig.root / "big.py").write_text(f"def huge():\n{body}\n    return '{tail}'\n", "utf-8")
    index_project(rig.root, rig.config)
    with Store.open(rig.root) as store:
        report = annotate_project(
            store, rig.slots, rig.cache, rig.config, tokenizer=CharTokenizer(), max_code_tokens=200
        )
    assert report.pending == 0
    huge = [prompt for prompt in rig.annotate.prompts if "def huge" in prompt]
    assert huge
    assert tail not in huge[0]


def test_context_overflow_is_recut_by_server_count(rig: Rig) -> None:
    tail = "TAIL_MARKER"
    body = "\n".join(f"    step_{n} = {n}" for n in range(100))
    (rig.root / "big.py").write_text(f"def huge():\n{body}\n    return '{tail}'\n", "utf-8")
    rig.annotate.context_chars = 1000
    assert rig.index() == chunk_count(rig)
    huge = [prompt for prompt in rig.annotate.prompts if "def huge" in prompt]
    assert len(huge) == 2
    assert tail in huge[0]
    assert tail not in huge[1]
    assert len(huge[1]) <= 1000


def test_rejected_chunks_are_skipped_and_pass_continues(rig: Rig) -> None:
    rig.annotate.context_chars = 10
    annotated, cached, pending = rig.annotate_slice()
    chunks = chunk_count(rig)
    assert (annotated, cached, pending) == (0, 0, chunks)
    assert len(rig.annotate.prompts) == 2 * chunks
