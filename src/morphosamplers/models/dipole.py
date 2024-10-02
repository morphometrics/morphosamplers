import numpy as np

from pydantic_compat  import validator

from morphosamplers.core import MorphoModel


class Dipole(MorphoModel):
    center: np.ndarray
    direction: np.ndarray
