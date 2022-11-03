"""Tools for image resampling."""

from typing import Tuple

import numpy as np
import einops
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from .spline import Spline3D


def generate_3d_grid(
    grid_shape: Tuple[int, int, int] = (10, 10, 10),
    grid_spacing: Tuple[float, float, float] = (1, 1, 1),
) -> np.ndarray:
    """
    Generate a 3D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, has shape (w, h, d, 3) for
    grid_shape (w, h, d), and spacing grid_spacing between neighboring points.

    Parameters
    ----------
    grid_shape : Tuple[int, int, int]
        The number of grid points along each axis.
    grid_spacing : Tuple[float, float, float]
        Spacing between points in the sampling grid.

    Returns
    -------
    np.ndarray
        Coordinate of points forming the 3D grid.
    """
    # generate a grid of points at each integer from 0 to grid_shape for each dimension
    grid = np.indices(grid_shape).astype(float)
    grid = einops.rearrange(grid, 'xyz w h d -> w h d xyz')
    # shift the grid to be centered on the origin
    grid -= (np.array(grid_shape)) // 2
    # scale the grid to get correct spacing
    grid *= grid_spacing
    return grid


def generate_2d_grid(
    grid_shape: Tuple[int, int] = (10, 10), grid_spacing: Tuple[float, float] = (1, 1)
) -> np.ndarray:
    """
    Generate a 2D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, lying on the plane with normal
    vector [0, 0, 1], has shape (w, h, 3) for grid_shape (w, h), and spacing
    grid_spacing between neighboring points.

    Parameters
    ----------
    grid_shape : Tuple[int, int]
        The number of grid points along each axis.
    grid_spacing : Tuple[float, float]
        Spacing between points in the sampling grid.

    Returns
    -------
    np.ndarray
        Coordinate of points forming the 2D grid.
    """
    grid = generate_3d_grid(grid_shape=(*grid_shape, 1), grid_spacing=(*grid_spacing, 1))
    return einops.rearrange(grid, 'w h 1 xyz -> w h xyz')


def place_sampling_grids(
    sampling_grid: np.ndarray, positions: np.ndarray, orientations: Rotation
) -> np.ndarray:
    """
    Copy and transform a given sampling grid onto a set of positions and orientations.

    Returns a (batch, *grid_shape, 3) batch of coordinate grids for sampling.

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
    for orientation in orientations:
        rotated.append(orientation.apply(grid_coords))
    # shift each rotated
    rotated_shifted = np.stack(rotated, axis=1) + positions
    return rotated_shifted.reshape(-1, *grid_shape)


def sample_volume_at_coordinates(
    volume: np.ndarray, coordinates: np.ndarray, interpolation_order: int = 3
) -> np.ndarray:
    """
    Sample a volume with spline interpolation at specific coordinates.

    The output shape is determined by the input coordinate shape such that
    if coordinates have shape (batch, *grid_shape, 3), the output array will have
    shape (*grid_shape, batch).

    Parameters
    ----------
    volume : np.ndarray
        Volume to be sampled.
    coordinates : np.ndarray
        Array of coordinates at which to sample the volume. The shape of this array
        should be (batch, *grid_shape, 3) to allow reshaping back correctly
    interpolation_order : int
        Spline order for image interpolation.

    Returns
    -------
    np.ndarray
        Array of shape (*grid_shape)
    """
    batch, *grid_shape, _ = coordinates.shape
    # map_coordinates wants transposed coordinate array
    sampled_volume = map_coordinates(
        volume, coordinates.reshape(-1, 3).T, order=interpolation_order
    )
    # reshape back (need to invert due to previous transposition)
    sampled_volume = sampled_volume.reshape(*grid_shape, batch)
    # and retranspose to get batch back to the 0th dimension
    return einops.rearrange(sampled_volume, '... batch -> batch ...')


def sample_volume_along_spline(
    volume: np.ndarray,
    spline: Spline3D,
    batch: int = 100,
    grid_shape: Tuple[int, int] = (10, 10),
    grid_spacing: Tuple[float, float] = (1, 1),
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
    batch : int
        Number of samples to take along the spline.
    grid_shape : Tuple[int, int]
        Shape of the 2D grid.
    grid_spacing : Tuple[float, float]
        Spacing between points in the sampling grid.
    interpolation_order : int
        Spline order for image interpolation.

    Returns
    -------
    np.ndarray
        Sampled volume.
    """
    u = np.linspace(0, 1, batch)
    positions = spline.sample_spline(u)
    orientations = spline.sample_spline_orientations(u)
    grid = generate_2d_grid(grid_shape=grid_shape, grid_spacing=grid_spacing)
    sampling_coords = place_sampling_grids(grid, positions, orientations)
    return sample_volume_at_coordinates(
        volume, sampling_coords, interpolation_order=interpolation_order
    )


def sample_subvolumes(
    volume: np.ndarray,
    positions: np.ndarray,
    orientations: Rotation,
    grid_shape: Tuple[int, int, int] = (10, 10, 10),
    grid_spacing: Tuple[float, float, float] = (1, 1, 1),
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
    grid_spacing : Tuple[float, float, float]
        Spacing between points in the sampling grid.
    interpolation_order : int
        Spline order for image interpolation.

    Returns
    -------
    np.ndarray
        Sampled volume.
    """
    grid = generate_3d_grid(grid_shape=grid_shape, grid_spacing=grid_spacing)
    sampling_coords = place_sampling_grids(grid, positions, orientations)
    return sample_volume_at_coordinates(
        volume, sampling_coords, interpolation_order=interpolation_order
    )
