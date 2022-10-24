"""Tools for image resampling."""

from typing import Tuple

import einops
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from .spline import Spline3D


def generate_2D_grid(
    grid_shape: Tuple[int, int] = (10, 10), grid_spacing: Tuple[int, int] = (1, 1)
) -> np.ndarray:
    """
    Generate a 2D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, lying on the xy plane,
    has shape (*grid_shape, 3) and spacing grid_spacing between neighboring points.

    Parameters
    ----------
    grid_shape : Tuple[int, int]
        Shape of the 2D grid.
    grid_spacing : Tuple[int, int]
        Spacing between points in the sampling grid.

    Returns
    -------
    np.ndarray
        Coordinate of points forming the 2D grid.
    """
    # create indices for x and y with correct spacing
    width = np.linspace(-1, 1, grid_shape[0]) * grid_shape[0] * grid_spacing[0]
    height = np.linspace(-1, 1, grid_shape[1]) * grid_shape[1] * grid_spacing[1]
    # convert to coordinate form (x, y, z, 3) (at z = 0)
    grid = np.stack(np.meshgrid(width, height, 0, indexing="ij"), axis=3)
    # drop degenerate dim
    return einops.rearrange(grid, "x y 1 d -> x y d")


def generate_3D_grid(
    grid_shape: Tuple[int, int, int] = (10, 10, 10),
    grid_spacing: Tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    """
    Generate a 3D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, has shape (*grid_shape, 3)
    and spacing grid_spacing between neighboring points.

    Parameters
    ----------
    grid_shape : Tuple[int, int, int]
        Shape of the 3D grid.
    grid_spacing : Tuple[int, int, int]
        Spacing between points in the sampling grid.

    Returns
    -------
    np.ndarray
        Coordinate of points forming the 2D grid.
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
) -> np.ndarray:
    """
    Copy and transform a given sampling grid onto a set of positions and orientations.

    Returns an (n, *grid_shape, 3) grid of sampling points.

    Parameters
    ----------
    sampling_grid : np.ndarray
        Grid of points to be copied and transformed.
    positions : np.ndarray
        Array of coordinates used to shift sampling grids.
    orientations : Rotation
        Rotations to be applied to each grid before shifts.

    Returns
    -------
    np.ndarray
        Coordinate of points combining all the transformed grids.
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


def sample_volume_with_coordinates(
    volume: np.ndarray, coordinates: np.ndarray, interpolation_order: int = 3
) -> np.ndarray:
    """
    Sample a volume with spline interpolation at specific coordinates.

    The output shape is determined by the input coordinate shape such that
    if coordinates have shape (n_samples, *grid_shape, 3), the output array will have
    shape (*grid_shape, n_samples).

    Parameters
    ----------
    volume : np.ndarray
        Volume to be sampled.
    coordinates : np.ndarray
        Array of coordinates at which to sample the volume. The shape of this array
        should be (n_samples, *grid_shape, 3) to allow reshaping back correctly
    interpolation_order : int
        Spline order for image interpolation.

    Returns
    -------
    np.ndarray
        Array of shape (*grid_shape)
    """
    n_samples, *grid_shape, _ = coordinates.shape
    # map_coordinates wants transposed coordinate array
    sampled_volume = map_coordinates(
        volume, coordinates.reshape(-1, 3).T, order=interpolation_order
    )
    # reshape back (need to invert due to previous transposition)
    # and retranspose to get n_samples back to the 0th dimension
    return np.swapaxes(sampled_volume.reshape(*grid_shape, n_samples), -1, 0)


def sample_volume_along_spline(
    volume: np.ndarray,
    spline: Spline3D,
    n_samples: int = 100,
    grid_shape: Tuple[int, int] = (10, 10),
    grid_spacing: Tuple[int, int] = (1, 1),
    interpolation_order: int = 3,
) -> np.ndarray:
    """
    Extract volume planes from a volume following a spline path.

    Parameters
    ----------
    volume : np.ndarray
        Volume to be sampled.
    spline : Spline3D
        Spline object along which to sample the volume.
    n_samples : int
        Number of samples to take along the spline.
    grid_shape : Tuple[int, int]
        Shape of the 2D grid.
    grid_spacing : Tuple[int, int]
        Spacing between points in the sampling grid.
    interpolation_order : int
        Spline order for image interpolation.

    Returns
    -------
    np.ndarray
        Sampled volume.
    """
    u = np.linspace(0, 1, n_samples)
    positions = spline.sample_spline(u)
    orientations = spline.sample_spline_orientations(u)
    grid = generate_2D_grid(grid_shape=grid_shape, grid_spacing=grid_spacing)
    sampling_coords = generate_sampling_coordinates(grid, positions, orientations)
    return sample_volume_with_coordinates(
        volume, sampling_coords, interpolation_order=interpolation_order
    )


def sample_volume_subvolumes(
    volume: np.ndarray,
    positions: np.ndarray,
    orientations: Rotation,
    grid_shape: Tuple[int, int, int] = (10, 10, 10),
    grid_spacing: Tuple[int, int, int] = (1, 1, 1),
    interpolation_order: int = 3,
) -> np.ndarray:
    """
    Extract arbitrarily oriented subvolumes from a volume.

    Parameters
    ----------
    volume : np.ndarray
        Volume to be sampled.
    positions : np.ndarray
        Positions of subvolumes.
    orientations : Rotation
        Orientations of subvolumes.
    grid_shape : Tuple[int, int, int]
        Shape of the 3D grid.
    grid_spacing : Tuple[int, int, int]
        Spacing between points in the sampling grid.
    interpolation_order : int
        Spline order for image interpolation.

    Returns
    -------
    np.ndarray
        Sampled volume.
    """
    grid = generate_3D_grid(grid_shape=grid_shape, grid_spacing=grid_spacing)
    sampling_coords = generate_sampling_coordinates(grid, positions, orientations)
    return sample_volume_with_coordinates(
        volume, sampling_coords, interpolation_order=interpolation_order
    )
