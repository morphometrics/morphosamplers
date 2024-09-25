import einops
import numpy as np
from scipy.interpolate import splev
from morphosamplers import Path
from morphosamplers.core import MorphoSampler
from morphosamplers.sample_types import PoseSet
from morphosamplers.utils import coaxial_y_vectors_from_z_vectors


class RingPoseSampler(MorphoSampler):
    spacing: float
    points_on_ring: int
    radius: float

    def sample(self, obj: Path) -> PoseSet:
        # sample equidistant points along path (same as parallel path)
        from .point_sampler import PointSampler
        tck, total_length = PointSampler.prepare_spline(obj)
        n_samples = int(total_length // self.spacing)
        max_u = 1 - ((total_length % self.spacing) / total_length)
        u = np.linspace(0, max_u, num=n_samples)
        positions = einops.rearrange(splev(u, tck), 'xyz b -> b xyz')

        # direction of spline at the position
        # (derivative of the spline at the position)
        tangents = splev(u, tck, der=1)
        tangents = np.array(tangents).T
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        # local coordinate system at the position (for circle points)
        # (normal vectors to the direction at the position)
        normals = np.cross(tangents, np.array([0, 0, 1]))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        binormals = np.cross(tangents, normals)
        normals = normals[:, np.newaxis, :] # reshape (a, 3) to (a, 1, 3)
        binormals = binormals[:, np.newaxis, :] # reshape (a, 3) to (a, 1, 3)

        # circle points
        # circle angles based on how many points on ring
        theta = np.linspace(0, 2 * np.pi, self.points_on_ring, endpoint=False)
        # circles local coordinates based on the angles
        cos_theta = np.cos(theta)[:, np.newaxis] # reshape (b,) to (b, 1)
        sin_theta = np.sin(theta)[:, np.newaxis] # reshape (b,) to (b, 1)

        circle_points = (sin_theta * normals + cos_theta * binormals)
        # scale circle points by radius
        circle_points *= self.radius
        # take each ring to its corresponding position on the spline
        ring_positions = positions[:, np.newaxis, :] + circle_points
        ring_positions = ring_positions.reshape(-1, 3) # reshape back to (a*b, 3)

        orientations = np.empty((ring_positions.shape[0], 3, 3))
        # generate orientations based on the backbone positions and circle positions
        # (associate each circle point to its corresponding backbone here)
        for i, (pos, ring_pos) in enumerate(zip(positions.repeat(self.points_on_ring, axis=0), ring_positions)):
            # outward directing vector: vector pointing from the backbone to the ring position
            new_z = ring_pos - pos
            new_z /= np.linalg.norm(new_z)
            # rest of the new coordinate system (vectors orthogonal to the new z)
            new_x = np.cross(new_z, tangents[i // self.points_on_ring]) # tangent at the corresponding backbone position
            new_x /= np.linalg.norm(new_x)
            new_y = np.cross(new_z, new_x)

            # orientation matrix for each circle point
            # local coordinate system at each circle point with z vector pointing outwards
            orientations[i, :, 0] = new_x
            orientations[i, :, 1] = new_y
            orientations[i, :, 2] = new_z

        return PoseSet(positions=ring_positions, orientations=orientations)
