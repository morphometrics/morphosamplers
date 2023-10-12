from typing import Tuple

from pydantic import validator

from morphosamplers.core import MorphoModel


class Sphere(MorphoModel):
    """A sphere in 3D."""
    center: Tuple[float, float, float]
    radius: float

    @validator('center', pre=True)
    def coerce_to_tuple(cls, value):
        return tuple(value)

