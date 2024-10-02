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

        # z in our coordinate system is the direction of spline at the sample position (i.e. parallel to the path)
        # (derivative of the spline at the position)
        path_sample_z_vectors = splev(u, tck, der=1)  # list of 3 (n, ) arrays
        path_sample_z_vectors = einops.rearrange(path_sample_z_vectors, 'xyz b -> b xyz')
        path_sample_z_vectors /= np.linalg.norm(path_sample_z_vectors, axis=1, keepdims=True)

        # Construct y in our coordinate system, this is a vector
        # perpendicular to our z-vectors.
        # We do this by taking the cross product with a random vector.
        # Find rows where the cross product is zero (i.e., the vectors are parallel or identical)
        # and replace with a vector perpendicular to the random vector used in cross product
        # calculation above then normalise
        path_sample_y_vectors = np.cross(path_sample_z_vectors, np.array([0, 0, 1]))
        idx_zero = np.linalg.norm(path_sample_y_vectors, axis=1) == 0
        path_sample_y_vectors[idx_zero] = np.array([1, 0, 0])
        path_sample_y_vectors /= np.linalg.norm(path_sample_y_vectors, axis=1, keepdims=True)

        # Finally, construct x in our coordinate system
        path_sample_x_vectors = np.cross(path_sample_y_vectors, path_sample_z_vectors)

        # Construct (n_path_samples, 3, 3) rotation matrices from x, y and z vectors
        orientations = np.empty(shape=(n_path_samples, 3, 3))
        orientations[:, :, 0] = path_sample_x_vectors
        orientations[:, :, 1] = path_sample_y_vectors
        orientations[:, :, 2] = path_sample_z_vectors

        # Generate points on a circle in the xy-place for each ring
        theta = np.linspace(0, 2 * np.pi, self.n_points_per_ring, endpoint=False)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        circle_z = np.zeros_like(theta)
        circle_xyz = einops.rearrange([circle_x, circle_y, circle_z], 'xyz b -> b xyz') # (n_points_per_ring, 3)
        circle_xyz *= self.radius  # (n_points_per_ring, 3)

        # Rotate each point in the circle by each rotation
        # target shape (n_points_per_ring, n_path_samples, 3)
        orientations = einops.rearrange(orientations, 'n_path_samples i j -> 1 n_path_samples i j')
        circle_xyz = einops.rearrange(circle_xyz, 'n_points_per_ring xyz -> n_points_per_ring 1 xyz 1')
        rotated_circle_xyz = orientations @ circle_xyz  # (n_points_per_ring, n_path_samples, 3, 1)
        rotated_circle_xyz = einops.rearrange(rotated_circle_xyz, 'b1 b2 i 1 -> b1 b2 i') # (n_points_per_ring, n_path_samples, 3)

        # place points around each path sample
        final_positions = rotated_circle_xyz + path_sample_positions # (n_points_per_ring, n_path_samples, 3)

        # Construct the orientations for each particle
        # - z vector pointing perpendicular to path
        # - y vector pointing along path
        # - x vector perpendicular to both
        # (n_points_per_ring, n_path_samples, 3, 3) orientations
        z_vectors = final_positions - path_sample_positions
        z_vectors /= np.linalg.norm(z_vectors, axis=2, keepdims=True)
        path_sample_parallel_vectors = einops.repeat(path_sample_z_vectors, 'n xyz -> b n xyz', b=self.n_points_per_ring)
        y_vectors = path_sample_parallel_vectors # along the path, calculated above
        y_vectors /= np.linalg.norm(y_vectors, axis=2, keepdims=True)
        x_vectors = np.cross(z_vectors, y_vectors)
        x_vectors /= np.linalg.norm(x_vectors, axis=2, keepdims=True)

        # Make the final_positions and final_orientations suitable for the PoseSet
        final_positions = einops.rearrange(final_positions, 'b1 b2 xyz -> (b1 b2) xyz')
        final_orientations = einops.rearrange([x_vectors, y_vectors, z_vectors], 'v b1 b2 xyz -> (b1 b2) xyz v')


        return PoseSet(positions=final_positions, orientations=final_orientations)






