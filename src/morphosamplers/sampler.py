"""Tools for image resampling."""

from typing import Tuple, Union

import numpy as np
import einops
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from .spline import Spline3D
from .surface_spline import GriddedSplineSurface


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


def generate_1d_grid(
    grid_shape: int = 10, grid_spacing: float = 1
) -> np.ndarray:
    """
    Generate a 1D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, lying along the vector
    [0, 0, 1], has shape (m, 3) for grid_shape m, and spacing grid_spacing
    between neighboring points.

    Parameters
    ----------
    grid_shape : int
        The number of grid points.
    grid_spacing : float
        Spacing between points in the sampling grid.

    Returns
    -------
    np.ndarray
        Coordinate of points forming the 1D grid.
    """
    grid = generate_3d_grid(grid_shape=(1, 1, grid_shape), grid_spacing=(1, 1, grid_spacing))
    return einops.rearrange(grid, '1 1 d xyz -> d xyz')


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
    spline: Union[np.ndarray, Spline3D],
    sampling_shape: Tuple[int, int] = (10, 10),
    sampling_spacing: float = 1,
    interpolation_order: int = 3,
) -> np.ndarray:
    """
    Extract planes from a volume following a spline path.

    Extracted planes are equidistant along the spline, and perpendicular to
    the spline derivative. The size of the planes is defined by the sampling_shape,
    and the distance between planes and between sample points on the plane is
    equal to sampling_spacing.

    For example: a spline of euclidean length 30, sampling_shape of (10, 10) and spacing of
    2, will result in a sampled volume of shape (15, 10, 10) with twice the pixel size
    of the original volume.

    Parameters
    ----------
    volume : np.ndarray
        Volume to be sampled.
    spline : Spline3D
        Array of points to use to generate the spline, or Spline object,
        along which to sample the volume.
    sampling_shape : Tuple[int, int]
        The number of points along each axis of the grid used for plane sampling.
    sampling_spacing : Tuple[float, float]
        Spacing between points in the sampling grid and along the spline.
    interpolation_order : int
        Spline order for image interpolation.

    Returns
    -------
    np.ndarray
        Sampled volume.
    """
    if not isinstance(spline, Spline3D):
        spline = Spline3D(points=spline)
    positions = spline.sample(separation=sampling_spacing)
    orientations = spline.sample_orientations(separation=sampling_spacing)
    grid = generate_2d_grid(grid_shape=sampling_shape, grid_spacing=(sampling_spacing, sampling_spacing))
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


def sample_volume_around_surface(
    volume: np.ndarray,
    surface: Union[np.ndarray, GriddedSplineSurface],
    sampling_thickness: int,
    sampling_spacing: float,
    interpolation_order: int = 3,
    masked: bool = False,
) -> np.ndarray:
    """
    Sample a volume around an arbitrary gridded surface.

    For each "root" point on the surface, samples are extracted along a line normal
    to the surface, so that n=sampling_thickness points are extracted, centered on the surface.
    The spacing between points on the surface (unless a pre-generated surface is given) and
    between the sampled points is equal to sampling_spacing. The sampled lines are then
    re-packed into a volume with the same dimensions as the surface, but "flattened".

    For example: a gridded surface of shape (50, 30), sampling_thickness of 10 and spacing of 3,
    will result in a sampled volume of shape (50, 30, 10) with three times the pixel size
    of the original volume.

    Parameters
    ----------
    volume : np.ndarray
        Volume to be sampled.
    surface : Union[np.ndarray, SplineSurfaceGrid]
        Array of points to be used to generate a surface, or SplineSurfaceGrid object,
        along which to sample the volume.
    sampling_thickness : int
        Thickness of sampled surface in pixels.
    sampling_spacing : float
        Spacing between sampled pixels.
    interpolation_order : int
        Spline order for image interpolation.

    Returns
    -------
    np.ndarray
        Sampled volume.
    """
    if not isinstance(surface, GriddedSplineSurface):
        surface = GriddedSplineSurface(points=surface, separation=sampling_spacing)
    positions = surface.sample()
    orientations = surface.sample_orientations()
    grid = generate_1d_grid(grid_shape=sampling_thickness, grid_spacing=sampling_spacing)
    sampling_coords = place_sampling_grids(grid, positions, orientations)
    sampled = sample_volume_at_coordinates(volume, sampling_coords, interpolation_order=interpolation_order)
    if masked:
        sampled[~surface.mask] = np.nan
    return sampled.reshape(*surface.grid_shape, sampling_thickness)
