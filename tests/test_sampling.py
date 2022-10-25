import numpy as np
from scipy.spatial.transform import Rotation

from morphosamplers.sampler import generate_2D_grid, generate_3D_grid, generate_sampling_coordinates, sample_volume_at_coordinates, sample_volume_along_spline, sample_subvolumes
from morphosamplers.spline import Spline3D


def test_generate_2D_grid():
    shape = (2, 5)
    spacing = (2, 1)
    grid = generate_2D_grid(grid_shape=shape, grid_spacing=spacing)
    assert grid.shape == (2, 5, 3)
    # x axis
    assert grid[..., 0].min() == -4
    assert grid[..., 0].max() == 4
    # y axis
    assert grid[..., 1].min() == -5
    assert grid[..., 1].max() == 5
    # z axis
    assert np.all(grid[:, :, 2] == 0)


def test_generate_3D_grid():
    shape = (2, 5, 3)
    spacing = (2, 1, 3)
    grid = generate_3D_grid(grid_shape=shape, grid_spacing=spacing)
    assert grid.shape == (2, 5, 3, 3)
    # x axis
    assert grid[..., 0].min() == -4
    assert grid[..., 0].max() == 4
    # y axis
    assert grid[..., 1].min() == -5
    assert grid[..., 1].max() == 5
    # z axis
    assert grid[..., 2].min() == -9
    assert grid[..., 2].max() == 9


def test_generate_sampling_coordinates():
    grid = generate_2D_grid((2, 2), (1, 1))
    pos = np.zeros((1, 3))
    ori = Rotation.identity(1)
    coords = generate_sampling_coordinates(grid, pos, ori)
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
    sampled = sample_volume_along_spline(vol, spline, batch=10, grid_shape=(3, 3))

    # line of ones
    np.testing.assert_allclose(sampled[:, 1, 1], 1, atol=1e-8)


def test_sample_volume_subvolumes():
    vol = np.zeros((10, 10, 10))
    vol[1, 2, 3] = 1
    pos = np.array([[1, 2, 3]])
    ori = Rotation.identity(1)
    sampled = sample_subvolumes(vol, pos, ori, grid_shape=(3, 3, 3))

    np.testing.assert_allclose(sampled[0, 1, 1, 1], 1, atol=1e-8)
