import einops
import numpy as np
from scipy.interpolate import splev
from scipy.spatial.transform import Rotation, Slerp

from morphosamplers import Path
from morphosamplers.core import MorphoSampler
from morphosamplers.sample_types import PoseSet
from morphosamplers.utils import coaxial_y_vectors_from_z_vectors


class PoseSampler(MorphoSampler):
    spacing: float

    def sample(self, obj: Path) -> PoseSet:
        """Sample a `Path` to produces a `PoseSet`.

        The positions in the `PoseSet` are equidistant points sampled according to
        `spacing`. The orientations in the `PoseSet` are as close to parallel to each
        other as curvature of the path allows. The z-axis is oriented along a
        smooth curve through the control points.
        """
        # sample equidistant points along path
        from .point_sampler import PointSampler
        tck, total_length = PointSampler.prepare_spline(obj)
        n_samples = total_length // self.spacing
        max_u = 1 - ((total_length % self.spacing) / total_length)
        u = np.linspace(0, max_u, num=int(n_samples))
        positions = einops.rearrange(splev(u, tck), 'xyz b -> b xyz')

        # oversample orientations along path which are maximally coaxial (parallel)
        u_oversampled = np.linspace(0, 1, num=5000)
        z = einops.rearrange(splev(u_oversampled, tck, der=1), 'xyz b -> b xyz')
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        y = coaxial_y_vectors_from_z_vectors(z)
        x = np.cross(y, z)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        rotations = np.empty(shape=(5000, 3, 3))
        rotations[:, :, 0] = x
        rotations[:, :, 1] = y
        rotations[:, :, 2] = z
        # rotations = einops.rearrange([x, y, z], 'xyz b vec -> b vec xyz')

        # interpolate rotations
        rotation_sampler = Slerp(u_oversampled, Rotation.from_matrix(rotations))
        orientations = rotation_sampler(u).as_matrix()
        return PoseSet(positions=positions, orientations=orientations)
