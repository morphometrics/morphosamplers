import numpy as np
import einops

from morphosamplers.core import MorphoSampler
from morphosamplers import Dipole
from morphosamplers.sample_types import Points


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
        hexagon_xyz = einops.rearrange([hexagon_x, hexagon_y, hexagon_z], 'xyz b -> b xyz') # (n, 3)
        hexagon_xyz = einops.rearrange(hexagon_xyz, 'b xyz -> b 1 xyz')
        centers = np.array(obj.center) # (m, 3)
        centers = einops.rearrange(centers, 'b xyz -> 1 b xyz')
        final_points = hexagon_xyz + centers
        final_points = einops.rearrange(final_points, 'b1 b2 xyz -> (b1 b2) xyz')
        return final_points





