import einops
import numpy as np
from scipy.interpolate import splev
from morphosamplers import Path
from morphosamplers.core import MorphoSampler
from morphosamplers.sample_types import PoseSet
from morphosamplers.utils import coaxial_y_vectors_from_z_vectors
from .pose_sampler_parallel import PoseSampler


class RingPoseSampler(MorphoSampler):
    spacing: float
    n_points_per_ring: int
    radius: float

    def sample(self, obj: Path) -> PoseSet:
        # Get parallel poses along filament axis
        sampler = PoseSampler(spacing=self.spacing)
        poses = sampler.sample(obj)

        # Generate points on a circle in the xy-place for each ring
        theta = np.linspace(0, 2 * np.pi, self.n_points_per_ring, endpoint=False)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        circle_z = np.zeros_like(theta)
        circle_xyz = einops.rearrange([circle_x, circle_y, circle_z], 'xyz b -> b xyz') # (n_points_per_ring, 3)
        circle_xyz *= self.radius  # (n_points_per_ring, 3)

        # Rotate each point in the circle by each rotation
        # target shape (n_points_per_ring, n_path_samples, 3)
        backbone_orientations = einops.rearrange(poses.orientations, 'n_path_samples i j -> 1 n_path_samples i j')
        circle_xyz = einops.rearrange(circle_xyz, 'n_points_per_ring xyz -> n_points_per_ring 1 xyz 1')
        rotated_circle_xyz = backbone_orientations @ circle_xyz  # (n_points_per_ring, n_path_samples, 3, 1)
        rotated_circle_xyz = einops.rearrange(rotated_circle_xyz,'b1 b2 i 1 -> b1 b2 i')  # (n_points_per_ring, n_path_samples, 3)

        # Repeat points at each path sample (all ring positions = backbone positions + ring positions)
        backbone_positions = einops.repeat(poses.positions, 'n xyz -> b n xyz', b=self.n_points_per_ring)
        final_positions = backbone_positions + rotated_circle_xyz # (n_points_per_ring, n_path_samples, 3)

        # Define the orientation for each ring point
        # (n_points_per_ring, n_path_samples, 3, 3) orientations
        z_vectors = rotated_circle_xyz / np.linalg.norm(rotated_circle_xyz, axis=2, keepdims=True)
        tangent_vectors = poses.orientations[:, :, 2] # along the path (parallel poses)
        y_vectors = einops.repeat(tangent_vectors, 'n xyz -> b n xyz', b=self.n_points_per_ring)
        y_vectors /= np.linalg.norm(y_vectors, axis=2, keepdims=True)
        x_vectors = np.cross(y_vectors, z_vectors)
        x_vectors /= np.linalg.norm(x_vectors, axis=2, keepdims=True)

        # Make the final_positions and final_orientations suitable for the PoseSet
        final_positions = einops.rearrange(final_positions, 'b1 b2 xyz -> (b1 b2) xyz')
        final_orientations = einops.rearrange([x_vectors, y_vectors, z_vectors], 'v b1 b2 xyz -> (b1 b2) xyz v')

        return PoseSet(positions=final_positions, orientations=final_orientations)






