import numpy as np
from pydantic import validator

from morphosamplers.core import MorphoModel


class Path(MorphoModel):
    """A 3D path defined by an `(n, 3)` array of control points."""
    control_points: np.ndarray

    @validator('control_points', pre=True)
    def coerce_to_n_by_3_array(cls, value):
        value = np.atleast_2d(np.asarray(value))
        if value.ndim != 2 or value.shape[-1] != 3:
            raise ValueError('`control_points` must be an (n, 3) array.')
        return value

    @validator('control_points')
    def check_at_least_two_points(cls, value):
        if len(value) < 2:
            raise ValueError('A Path must contain at least two points.')
        return value

    def __len__(self) -> int:
        return len(self.control_points)
