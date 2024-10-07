import numpy as np
import einops

from morphosamplers.core import MorphoSampler
from morphosamplers import Dipole
from morphosamplers.sample_types import Points
from morphosamplers.samplers.dipole_samplers.pose_sampler import PoseSampler


class DiskSampler(MorphoSampler):
    spacing: float
    radius: float

    def sample(self, obj: Dipole) -> Points:
        """Sample points in a circle arranged in a hexagonal packing"""
        # Hexagonal grid
        hexagon_ratio = np.sqrt(3) / 2
        hexagon_grid_x = np.arange(-self.radius, self.radius + self.spacing, self.spacing)
        hexagon_grid_y = np.arange(-self.radius, self.radius + self.spacing, self.spacing * hexagon_ratio)
        hexagon_x, hexagon_y = np.meshgrid(hexagon_grid_x, hexagon_grid_y)
        hexagon_x[1::2] += self.spacing / 2 # shift the coordinates by half spacing for the hexagonal packing
        inside_circle = (hexagon_x ** 2 + hexagon_y ** 2) <= self.radius ** 2
        hexagon_x = hexagon_x[inside_circle]
        hexagon_y = hexagon_y[inside_circle]
        hexagon_z = np.zeros_like(hexagon_x)
        hexagon_xyz = einops.rearrange([hexagon_x, hexagon_y, hexagon_z], 'xyz b -> b xyz 1')
        pose_sampler = PoseSampler()
        pose_set = pose_sampler.sample(obj)
        centers = pose_set.positions
        orientations = pose_set.orientations
        rotated_points = orientations @ hexagon_xyz
        rotated_points = einops.rearrange(rotated_points, 'b1 xyz 1 -> b1 xyz')
        final_points = rotated_points + centers
        return final_points
