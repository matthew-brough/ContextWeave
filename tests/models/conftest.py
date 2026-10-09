import pytest
from models_support import Clock

from contextweave.core.config import AnnotateConfig, EmbedConfig, ModelsConfig


@pytest.fixture
def models() -> ModelsConfig:
    return ModelsConfig(
        embed=EmbedConfig(dim=3, query_prefix="Q:", doc_prefix="D:", memory_prefix="M:"),
        annotate=AnnotateConfig(think=True),
    )


@pytest.fixture
def clock() -> Clock:
    return Clock()
