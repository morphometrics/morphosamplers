import numpy as np
from morphosamplers.sample_types import PoseSet

from morphosamplers.core import MorphoSampler
from morphosamplers import Dipole


class PoseSampler(MorphoSampler):
    def sample(self, obj: Dipole) -> PoseSet:
        positions = obj.center.reshape((1, 3))
        orientations = np.empty((1, 3, 3))
        z_vector = obj.direction / np.linalg.norm(obj.direction)
        arbitrary_vector = np.array([1, 0, 0])
        if np.allclose(z_vector, arbitrary_vector) or np.allclose(z_vector, -arbitrary_vector):
            arbitrary_vector = np.array([0, 1, 0])
        y_vector = np.cross(z_vector, arbitrary_vector)
        y_vector /= np.linalg.norm(y_vector)
        x_vector = np.cross(y_vector, z_vector)
        orientations[:, :, 0] = x_vector
        orientations[:, :, 1] = y_vector
        orientations[:, :, 2] = z_vector
        return PoseSet(positions=positions, orientations=orientations)
