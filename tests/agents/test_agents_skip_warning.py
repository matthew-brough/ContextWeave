from pathlib import Path

import pytest

from contextweave.agents.transport import InprocTransport
from contextweave.core.config import Config, load_config
from contextweave.core.project import ResolvedProject
from contextweave.core.refs import project_key
from contextweave.daemon.models import ModelRuntime
from contextweave.daemon.skipwarn import skip_rate_warnings
from contextweave.index.derived import DerivedCache
from contextweave.models.providers import Slot
from contextweave.models.slots import ModelSlots


@pytest.fixture
def runtime(tmp_path: Path) -> ModelRuntime:
    config = Config()
    return ModelRuntime(
        ModelSlots.from_config(config.models), DerivedCache(tmp_path / "d.db"), config, None
    )


def test_no_models_no_warnings() -> None:
    assert skip_rate_warnings(None)() == ()


def test_quiet_below_threshold(runtime: ModelRuntime) -> None:
    health = runtime.slots.healths[Slot.EMBED]
    for _ in range(100):
        health.record_success()
    assert skip_rate_warnings(runtime)() == ()
    runtime.close()


def test_warns_over_skip_warn(runtime: ModelRuntime) -> None:
    health = runtime.slots.healths[Slot.EMBED]
    for _ in range(10):
        health.record_unavailable()
    for _ in range(90):
        health.record_success()
    lines = skip_rate_warnings(runtime)()
    runtime.close()
    assert lines == ("warning: skip_rate slot embed 10% over last 100 calls",)


def test_session_start_includes_skip_warning(
    root: Path, env: dict[str, str], runtime: ModelRuntime
) -> None:
    for _ in range(10):
        runtime.slots.healths[Slot.EMBED].record_unavailable()
    runtime.close()
    project = ResolvedProject(project_key(root), root, "cwd")
    transport = InprocTransport(env, skip_rate_warnings(runtime))
    context = transport.session_start(project, "s1", load_config(root, strict=False, env=env))
    assert context is not None
    assert "warning: skip_rate slot embed 100%" in context
