import numpy as np
from scipy.spatial.transform import Rotation

from morphosamplers.sampler import generate_2d_grid, generate_3d_grid, place_sampling_grids, sample_volume_at_coordinates, sample_volume_along_spline, sample_subvolumes
from morphosamplers.spline import Spline3D


def test_generate_2D_grid():
    shape = (3, 4)
    spacing = (2, 3)
    grid = generate_2d_grid(grid_shape=shape, grid_spacing=spacing)
    assert grid.shape == (*shape, 3)
    # x and y
    for axis in range(2):
        vals = np.swapaxes(grid[..., axis], axis, 0)
        np.testing.assert_allclose(vals[1:] - vals[:-1], spacing[axis])
    # z axis
    z = grid[..., 2]
    assert np.all(z == 0)


def test_generate_3D_grid():
    shape = (2, 5, 3)
    spacing = (2, 1, 3)
    grid = generate_3d_grid(grid_shape=shape, grid_spacing=spacing)
    assert grid.shape == (*shape, 3)
    for axis in range(3):
        vals = np.swapaxes(grid[..., axis], axis, 0)
        np.testing.assert_allclose(vals[1:] - vals[:-1], spacing[axis])


def test_generate_sampling_coordinates():
    grid = generate_2d_grid((2, 2), (1, 1))
    pos = np.zeros((1, 3))
    ori = Rotation.identity(1)
    coords = place_sampling_grids(grid, pos, ori)
    assert coords.shape == (1, 2, 2, 3)


def test_sample_volume_at_coordinates():
    vol = np.zeros((10, 10, 10))
    vol[1, 2, 3] = 1
    coords = np.array([1, 2, 3]).reshape(1, 1, 1, 3)
    sample = sample_volume_at_coordinates(vol, coords)
    np.testing.assert_allclose(sample, 1)


def test_sample_volume_along_spline():
    vol = np.zeros((10, 10, 10))
    vol[1, 2, :] = 1
    pts = np.array([[1, 2, 0], [1, 2, 9]])
    spline = Spline3D(points=pts, order=1)
    sampled = sample_volume_along_spline(vol, spline, sampling_shape=(3, 3), sampling_spacing=1)

    # line of ones
    np.testing.assert_allclose(sampled[:, 1, 1], 1, atol=1e-8)


def test_sample_volume_subvolumes():
    vol = np.zeros((10, 10, 10))
    vol[1, 2, 3] = 1
    pos = np.array([[1, 2, 3]])
    ori = Rotation.identity(1)
    sampled = sample_subvolumes(vol, pos, ori, grid_shape=(3, 3, 3))

    np.testing.assert_allclose(sampled[0, 1, 1, 1], 1, atol=1e-8)
