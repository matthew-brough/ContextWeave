import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

from contextweave.core.config import Config
from contextweave.core.paths import derived_db_path
from contextweave.index.annotate import AnnotateReport, annotate_progress, annotate_project
from contextweave.index.chunk_tokens import ChunkTokenizer, embed_tokenizer
from contextweave.index.derived import DerivedCache
from contextweave.index.embed import EmbedReport, embed_project
from contextweave.index.progress import ProgressSink, discard
from contextweave.index.store import Store
from contextweave.models.residency import Profile
from contextweave.models.slots import ModelSlots


@dataclass(frozen=True, slots=True)
class DeriveReport:
    """What one annotate-then-embed pass did."""

    annotate: AnnotateReport
    embed: EmbedReport


@dataclass(slots=True)
class ModelRuntime:
    """Model clients and derived cache the daemon owns, plus the residency profile switch."""

    slots: ModelSlots
    cache: DerivedCache
    config: Config
    tokenizer: ChunkTokenizer | None
    profile_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls, config: Config, env: Mapping[str, str]) -> ModelRuntime:
        slots = ModelSlots.from_config(config.models, env=env)
        cache = DerivedCache(derived_db_path(env))
        tokenizer = embed_tokenizer(config.models.embed.model, env)
        return cls(slots, cache, config, tokenizer)

    def embed_changed(self, store: Store, on_progress: ProgressSink = discard) -> EmbedReport:
        """Embed chunks lacking a current vector."""
        return embed_project(
            store,
            self.slots,
            self.cache,
            self.config,
            tokenizer=self.tokenizer,
            on_progress=on_progress,
        )

    def annotate_changed(self, store: Store, on_progress: ProgressSink = discard) -> AnnotateReport:
        """Annotate chunks lacking a current annotation, reusing cached ones."""
        return annotate_project(
            store, self.slots, self.cache, self.config, on_progress=annotate_progress(on_progress)
        )

    def derive_changed(self, store: Store) -> DeriveReport:
        """Annotate, then embed, so the annot vector space sees the new annotations."""
        return DeriveReport(self.annotate_changed(store), self.embed_changed(store))

    def apply(self, profile: Profile) -> None:
        self.slots.apply_profile(profile)

    def bulk_derive(
        self, store: Store, *, serve_after: bool = True, on_progress: ProgressSink = discard
    ) -> DeriveReport:
        """Derive with each model alone on the GPU in turn: annotate, then embed."""
        with self.profile_lock:
            try:
                self.slots.apply_profile(Profile.BULK_ANNOTATE)
                annotated = self.annotate_changed(store, on_progress)
                self.slots.apply_profile(Profile.BULK_EMBED)
                return DeriveReport(annotated, self.embed_changed(store, on_progress))
            finally:
                if serve_after:
                    self.slots.apply_profile(Profile.SERVE)

    def close(self) -> None:
        self.slots.release()
        self.cache.close()
