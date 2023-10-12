import numpy as np
from scipy.spatial.transform import Rotation as R

from morphosamplers.core import MorphoSampler
from morphosamplers import Sphere
from morphosamplers.sample_types import PoseSet
from morphosamplers.samplers.sphere_samplers.point_sampler import PointSampler


class PoseSampler(MorphoSampler):
    """Sample poses packed on the surface of a sphere with a defined spacing."""
    spacing: float

    def sample(self, obj: Sphere) -> PoseSet:
        # get equally spaced points on surface
        point_sampler = PointSampler(spacing=self.spacing)
        positions = point_sampler.sample(obj)

        # get some initial orientation with z normal to the surface
        z_vectors = positions - np.asarray(obj.center)
        z_vectors /= np.linalg.norm(z_vectors, axis=-1, keepdims=True)
        y_vectors = np.cross(z_vectors, [1, 1, 1])
        y_vectors /= np.linalg.norm(y_vectors, axis=-1, keepdims=True)
        x_vectors = np.cross(y_vectors, z_vectors)
        x_vectors /= np.linalg.norm(x_vectors, axis=-1, keepdims=True)
        orientations = np.empty(shape=(len(positions), 3, 3), dtype=np.float32)
        orientations[:, :, 0] = x_vectors
        orientations[:, :, 1] = y_vectors
        orientations[:, :, 2] = z_vectors

        # randomise in plane rotation
        angles = np.random.uniform(0, 360, size=(len(positions)))
        Rz = R.from_euler('z', angles=angles, degrees=True).as_matrix()
        orientations = orientations @ Rz
        return PoseSet(positions=positions, orientations=orientations)
