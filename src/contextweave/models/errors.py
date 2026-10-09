from typing import ClassVar

from contextweave.core.errors import StageError


class ModelError(StageError):
    """Base for every model-slot failure; carries a stable code and the device tried."""

    code_value: ClassVar[str] = "model_error"

    def __init__(self, message: str, *, device_tried: str | None = None) -> None:
        super().__init__(self.code_value, message, device_tried=device_tried)


class ModelDisabledError(ModelError):
    """The slot's provider is none."""

    code_value = "model_disabled"


class ModelDeadlineError(ModelError):
    """A call exceeded the per-call deadline."""

    code_value = "model_deadline"


class ModelTransportError(ModelError):
    """The model endpoint was unreachable or answered with an error status."""

    code_value = "model_transport"


class ModelResponseError(ModelError):
    """The model endpoint answered with a payload of the wrong shape."""

    code_value = "model_bad_response"


class ModelParseError(ModelError):
    """Generated text was not valid structured output."""

    code_value = "model_parse"


class ProviderSlotError(ModelError):
    """A provider was used in a slot it does not belong to."""

    code_value = "provider_slot"
