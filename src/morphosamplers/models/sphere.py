from typing import Tuple

from morphosamplers.core import MorphoModel


class Sphere(MorphoModel):
    """A sphere in 3D."""
    center: Tuple[float, float, float]
    radius: float
