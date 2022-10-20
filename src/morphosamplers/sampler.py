from scipy.ndimage import map_coordinates
import numpy as np


from .spline import Spline3D


def sample_with_grid(positions, orientations, image, grid_shape=(10, 10), grid_spacing=1, spline_interpolation_order=3):
    width = np.linspace(-1, 1, grid_shape[0]) * grid_shape[0] * grid_spacing
    height = np.linspace(-1, 1, grid_shape[1]) * grid_shape[1] * grid_spacing
    grid = np.stack(np.meshgrid(width, height, 0), axis=3).reshape(-1, 3)
    rotated = []
    for r in orientations:
        rotated.append(r.apply(grid))
    rotated_shifted = rotated + positions.reshape(-1, 1, 3)
    sampled_image = map_coordinates(image, rotated_shifted.T, order=spline_interpolation_order)
    return sampled_image.reshape(*grid_shape, -1).T


def sample_along_spline(points, image, n_samples=100, spline_order=3, **kwargs):
    spline = Spline3D(points=points, order=spline_order)
    u = np.linspace(0, 1, n_samples)
    pos = spline.sample_spline(u)
    ori = spline.sample_spline_orientations(u)
    return sample_with_grid(pos, ori, image, **kwargs)
