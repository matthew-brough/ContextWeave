from collections.abc import Callable, Mapping
from dataclasses import dataclass

from contextweave.core.registry import Registry

PROVIDER_ENV = "CW_MODELS__CLASSIFY__PROVIDER"
DEVICES_ENV = "CW_MODELS__CLASSIFY__DEVICES"
MODEL_ENV = "CW_MODELS__CLASSIFY__MODEL"
BUILD_ENV = "CW_MODELS__CLASSIFY__BUILD_ID"
DEVICE_CHOICES = ("auto", "gpu", "cpu")
SKIP_PROVIDER = "none"
VARIANT_PROFILE = "full"


@dataclass(frozen=True, slots=True)
class VariantOptions:
    """Per-run knobs a variant may honour."""

    device: str = "auto"
    model: str | None = None
    build_id: str | None = None


@dataclass(frozen=True, slots=True)
class Variant:
    """A classify configuration under evaluation, as env overrides on the served process."""

    name: str
    env: Callable[[VariantOptions], Mapping[str, str]]

    def label(self, options: VariantOptions) -> str:
        return f"{self.name}/{options.device}" if self.name != "skip" else self.name


def devices_literal(device: str) -> str:
    return '["gpu", "cpu"]' if device == "auto" else f'["{device}"]'


def provider_env(provider: str) -> Callable[[VariantOptions], Mapping[str, str]]:
    def build(options: VariantOptions) -> Mapping[str, str]:
        env = {PROVIDER_ENV: provider, DEVICES_ENV: devices_literal(options.device)}
        if options.model is not None:
            env[MODEL_ENV] = options.model
        if options.build_id is not None:
            env[BUILD_ENV] = options.build_id
        return env

    return build


def skip_env(options: VariantOptions) -> Mapping[str, str]:
    return {PROVIDER_ENV: SKIP_PROVIDER}


def default_variants() -> Registry[str, Variant]:
    """Registry holding kev, clef and skip."""
    registry: Registry[str, Variant] = Registry()
    registry.register("kev", Variant("kev", provider_env("kev")))
    registry.register("clef", Variant("clef", provider_env("clef")))
    registry.register("skip", Variant("skip", skip_env))
    return registry
