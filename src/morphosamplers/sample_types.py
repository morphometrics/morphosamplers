import numpy as np
from pydantic import BaseModel, root_validator, validator
from typing import Union
from typing_extensions import TypeAlias

Image: TypeAlias = np.ndarray
Mask: TypeAlias = np.ndarray
Points: TypeAlias = np.ndarray  # (n, 3)


class PoseSet(BaseModel):
    """Model for a set of 3D poses."""
    positions: np.ndarray  # (n, 3)
    orientations: np.ndarray  # (n, 3, 3)

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    @validator('positions', pre=True)
    def coerce_to_n_by_3_array(cls, value):
        value = np.atleast_2d(np.asarray(value))
        if value.ndim != 2 or value.shape[-1] != 3:
            raise ValueError('positions must be an (n, 3) array.')
        return value

    @validator('orientations', pre=True)
    def check_n_by_3_3_array(cls, value):
        value = np.asarray(value)
        if value.ndim != 3 or value.shape[-2:] != (3, 3):
            raise ValueError('orientations must be an (n, 3, 3) array.')
        return value

    @root_validator
    def check_same_length(cls, values):
        positions, orientations = values.get('positions'), values.get('orientations')
        if len(positions) != len(orientations):
            raise ValueError("lengths of positions and orientations don't match")
        return values

    def __len__(self) -> int:
        return len(self.positions)


SampleType = Union[PoseSet, Image, Mask, Points]
