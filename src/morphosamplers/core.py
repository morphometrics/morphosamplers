from abc import abstractmethod
from typing import TypeVar, Protocol, Generic

from pydantic import ConfigDict
# this may warn, but works until v3
from pydantic.generics import GenericModel

M = TypeVar("M", bound="MorphoModel")
S = TypeVar("S")


class MorphoModel(GenericModel, Generic[M]):
    """A set of attributes defining a geometrical support."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )


class SamplerProtocol(Protocol[M, S]):
    """Protocol for classes that sample a `MorphoModel`."""

    @abstractmethod
    def sample(self, obj: M, *args) -> S:
        """Sample a `MorphoModel` to produces samples of `Sa`."""
        ...


SamplerType = TypeVar("SamplerType", bound=SamplerProtocol)


class MorphoSampler(GenericModel, Generic[SamplerType]):
    """Concrete samplers should subclass this generic model."""
    model_config = ConfigDict(
        frozen=True,
    )
