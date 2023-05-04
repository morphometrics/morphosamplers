from typing import List

import numpy as np
from pydantic import validator

from morphosamplers.core import MorphoModel


class Surface(MorphoModel):
    """A 3D surface defined by control points in a series of levels."""
    control_points: List[np.ndarray]

    @validator('control_points')
    def check_at_least_two_points(cls, value):
        if len(value) < 2:
            raise ValueError('A Path must contain at least two levels.')
        for array in value:
            if len(array) < 2:
                raise ValueError('Each level must contain at least two points.')
        return value

    @validator('control_points', pre=True)
    def ensure_list_of_float_arrays(cls, value):
        return [np.asarray(v, dtype=np.float32) for v in value]

    def __len__(self) -> int:
        return len(self.control_points)