import einops
import numpy as np
from scipy.interpolate import splev
from morphosamplers import Path
from morphosamplers.core import MorphoSampler
from morphosamplers.sample_types import PoseSet
from morphosamplers.utils import coaxial_y_vectors_from_z_vectors


class RingPoseSampler(MorphoSampler):
    spacing: float
    n_points_per_ring: int
    radius: float

    def sample(self, obj: Path) -> PoseSet:
        # sample equidistant points along path (same as parallel path)
        # these will be the positions of the rings
        from .point_sampler import PointSampler
        tck, total_length = PointSampler.prepare_spline(obj)
        n_path_samples = int(total_length // self.spacing)
        max_u = 1 - ((total_length % self.spacing) / total_length)
        u = np.linspace(0, max_u, num=n_path_samples)
        path_sample_positions = einops.rearrange(splev(u, tck), 'xyz b -> b xyz')


        # Construct a coordinate system at each sampled point along the path
        # - z points along the path
        # - y is perpendicular to z
        # - x is perpendicular to y and z

        # z in our coordinate system is the direction of spline at the sample position
        # (derivative of the spline at the position)
        z_vectors = splev(u, tck, der=1)  # list of 3 (n, ) arrays
        z_vectors = einops.rearrange(z_vectors, 'xyz b -> b xyz')
        z_vectors /= np.linalg.norm(z_vectors, axis=1, keepdims=True)

        # Construct y in our coordinate system, this is a vector
        # perpendicular to our z-vectors.
        # We do this by taking the cross product with a random vector.
        # Find rows where the cross product is zero (i.e., the vectors are parallel or identical)
        # and replace with a vector perpendicular to the random vector used in cross product
        # calculation above then normalise
        y_vectors = np.cross(z_vectors, np.array([0, 0, 1]))
        idx_zero = np.linalg.norm(y_vectors, axis=1) == 0
        y_vectors[idx_zero] = np.array([1, 0, 0])
        y_vectors /= np.linalg.norm(y_vectors, axis=1, keepdims=True)

        # Finally, construct x in our coordinate system
        x_vectors = np.cross(y_vectors, z_vectors)

        # Construct (n_path_samples, 3, 3) rotation matrices from x, y and z vectors
        orientations = np.empty(shape=(n_path_samples, 3, 3))
        orientations[:, :, 0] = x_vectors
        orientations[:, :, 1] = y_vectors
        orientations[:, :, 2] = z_vectors

        # Generate points on a circle in the xy-place for each ring
        theta = np.linspace(0, 2 * np.pi, self.n_points_per_ring, endpoint=False)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        circle_z = np.zeros_like(theta)
        circle_xyz = einops.rearrange([circle_x, circle_y, circle_z], 'xyz b -> b xyz')
        circle_xyz *= self.radius  # (n_points_per_ring, 3)

        # Rotate each point in the circle by each rotation
        # target shape (n_points_per_ring, n_path_samples, 3)
        orientations = einops.rearrange(orientations, 'n_path_samples i j -> 1 n_path_samples i j')
        circle_xyz = einops.rearrange(circle_xyz, 'n_points_per_ring xyz -> n_points_per_ring 1 xyz 1')
        rotated_circle_xyz = orientations @ circle_xyz  # (n_points_per_ring, n_path_samples, 3, 1)
        rotated_circle_xyz = einops.rearrange(rotated_circle_xyz, 'b1 b2 i 1 -> b1 b2 i')

        # place points around each path sample
        final_positions = rotated_circle_xyz + path_sample_positions  # (n_points_per_ring, n_path_samples, 3)

        ## ayse's job - construct the orientations for each particle
        # finding vector to corrsponding path sample for each particle
        # - z vector pointing perpendicular to path
        # - y vector pointing along path
        # - x vector perpendicular to both
        # construct (n_points_per_ring, n_path_samples, 3, 3) orientations



        return PoseSet(positions=ring_positions, orientations=orientations)
