import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from contextweave.core.config import Config
from contextweave.core.paths import derived_db_path
from contextweave.index.annotate import annotate_project
from contextweave.index.chunk_tokens import ChunkTokenizer, embed_tokenizer
from contextweave.index.derived import DerivedCache
from contextweave.index.embed import embed_project
from contextweave.index.store import Store
from contextweave.models.residency import Profile
from contextweave.models.slots import ModelSlots


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
        slots = ModelSlots.from_config(config.models)
        cache = DerivedCache(derived_db_path(env))
        tokenizer = embed_tokenizer(config.models.embed.model, env)
        return cls(slots, cache, config, tokenizer)

    def embed_changed(self, store: Store) -> None:
        """Embed chunks lacking a current vector."""
        embed_project(store, self.slots, self.cache, self.config, tokenizer=self.tokenizer)

    def annotate_changed(self, store: Store) -> None:
        """Annotate chunks lacking a current annotation, reusing cached ones."""
        annotate_project(store, self.slots, self.cache, self.config)

    def derive_changed(self, store: Store) -> None:
        """Annotate, then embed, so the annot vector space sees the new annotations."""
        self.annotate_changed(store)
        self.embed_changed(store)

    def apply(self, profile: Profile) -> None:
        self.slots.apply_profile(profile)

    def bulk[T](self, work: Callable[[], T]) -> T:
        """Run work under the bulk residency profile, then return to serve."""
        with self.profile_lock:
            self.slots.apply_profile(Profile.BULK)
            try:
                return work()
            finally:
                self.slots.apply_profile(Profile.SERVE)

    def close(self) -> None:
        self.cache.close()
