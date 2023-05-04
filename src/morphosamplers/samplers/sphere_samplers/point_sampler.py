import numpy as np

from morphosamplers.core import MorphoSampler
from morphosamplers import Sphere

from ._sphere_utils import fibonacci_sphere, n_from_spacing


class PointSampler(MorphoSampler):
    spacing: float

    def sample(self, obj: Sphere) -> np.ndarray:
        n = n_from_spacing(self.spacing / obj.radius)
        return fibonacci_sphere(n) * obj.radius + np.asarray(obj.center)


