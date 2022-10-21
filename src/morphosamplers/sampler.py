"""Tools for image resampling."""

import einops
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from .spline import Spline3D


def generate_2D_grid(grid_shape=(10, 10), grid_spacing=(1, 1)) -> np.ndarray:
    """
    Generate a 2D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, lying on the xy plane,
    has shape (*grid_shape, 3) and spacing grid_spacing between neighboring points.
    """
    # create indices for x and y with correct spacing
    width = np.linspace(-1, 1, grid_shape[0]) * grid_shape[0] * grid_spacing[0]
    height = np.linspace(-1, 1, grid_shape[1]) * grid_shape[1] * grid_spacing[1]
    # convert to coordinate form (x, y, z, 3) (at z = 0)
    grid = np.stack(np.meshgrid(width, height, 0, indexing="ij"), axis=3)
    # drop degenerate dim
    return einops.rearrange(grid, "x y 1 d -> x y d")


def generate_3D_grid(grid_shape=(10, 10, 10), grid_spacing=(1, 1, 1)) -> np.ndarray:
    """
    Generate a 3D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, has shape (*grid_shape, 3)
    and spacing grid_spacing between neighboring points.
    """
    # create indices for x and y with correct spacing
    width = np.linspace(-1, 1, grid_shape[0]) * grid_shape[0] * grid_spacing[0]
    height = np.linspace(-1, 1, grid_shape[1]) * grid_shape[1] * grid_spacing[1]
    depth = np.linspace(-1, 1, grid_shape[2]) * grid_shape[2] * grid_spacing[2]
    # convert to stack form (x, y, z, 3)
    grid = np.stack(np.meshgrid(width, height, depth, indexing="ij"), axis=3)
    return grid


def generate_sampling_coordinates(
    sampling_grid: np.ndarray, positions: np.ndarray, orientations: Rotation
):
    """
    Copy and transform a given sampling grid onto a set of positions and orientations.

    Returns an (n, *grid_shape, 3) grid of sampling points.
    """
    rotated = []
    grid_shape = sampling_grid.shape
    grid_coords = sampling_grid.reshape(-1, 3)
    # apply each orientation to the grid and store the result
    for ori in orientations:
        rotated.append(ori.apply(grid_coords))
    # shift each rotated
    rotated_shifted = np.stack(rotated, axis=1) + positions
    return rotated_shifted.reshape(-1, *grid_shape)


def sample_image_with_coordinates(image, coordinates, image_interpolation_order=3):
    """
    Sample an image with spline interpolation at specific coordinates.

    The output shape is determined by the input coordinate shape such that
    if coordinates have shape (n_samples, *grid_shape, 3), the output image will have
    shape (n_samples, *grid_shape).
    """
    n_samples, *grid_shape, _ = coordinates.shape
    # map_coordinates wants transposed coordinate array
    sampled_image = map_coordinates(
        image, coordinates.reshape(-1, 3).T, order=image_interpolation_order
    )
    # reshape back (need to invert due to previous transposition)
    # and retranspose to get n_samples back to the 0th dimension
    return sampled_image.reshape(*grid_shape, n_samples)


def sample_image_along_spline(
    image: np.ndarray,
    spline: Spline3D,
    n_samples=100,
    grid_shape=(10, 10),
    grid_spacing=(1, 1),
    image_interpolation_order=3,
):
    """Extract image planes from a volume following a spline path."""
    u = np.linspace(0, 1, n_samples)
    pos = spline.sample_spline(u)
    ori = spline.sample_spline_orientations(u)
    grid = generate_2D_grid(grid_shape=grid_shape, grid_spacing=grid_spacing)
    sampling_coords = generate_sampling_coordinates(grid, pos, ori)
    return sample_image_with_coordinates(
        image, sampling_coords, image_interpolation_order=image_interpolation_order
    )


def sample_image_subvolumes(
    image,
    positions,
    orientations,
    grid_shape=(10, 10, 10),
    grid_spacing=(1, 1, 1),
    image_interpolation_order=3,
):
    """Extract arbitrarily oriented subvolumes from a volume."""
    grid = generate_3D_grid(grid_shape=grid_shape, grid_spacing=grid_spacing)
    sampling_coords = generate_sampling_coordinates(grid, positions, orientations)
    return sample_image_with_coordinates(
        image, sampling_coords, image_interpolation_order=image_interpolation_order
    )
