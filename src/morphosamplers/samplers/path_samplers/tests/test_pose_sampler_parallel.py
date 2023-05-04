import einops
import numpy as np

from morphosamplers.sample_types import PoseSet
from morphosamplers.samplers.path_samplers import PathSamplers
from morphosamplers.models import MorphoModels


def test_instantiation():
    sampler = PathSamplers.PoseSampler(spacing=5)
    assert isinstance(sampler, PathSamplers.PoseSampler)


def test_sampling():
    """Samples should interpolate control points with consistent orientations."""
    n_points = 50
    total_length = 100
    spacing = total_length / n_points
    x = y = np.zeros(shape=(n_points, ))
    z = np.linspace(0, total_length, num=n_points)
    control_points = einops.rearrange([x, y, z], 'xyz b -> b xyz')
    path = MorphoModels.Path(control_points=control_points)
    sampler = PathSamplers.PoseSampler(spacing=spacing)
    samples = sampler.sample(path)

    # positions should be an interpolation along z from 0 -> total_length
    assert isinstance(samples, PoseSet)
    expected = np.linspace([0, 0, 0], [0, 0, total_length], num=n_points)
    assert np.allclose(samples.positions, expected)

    # z axis should be a unit vector along +z
    z = samples.orientations[:, :, 2]
    assert np.allclose(z, [0, 0, 1])

    # y0 and y1 should be ~identical unit vectors...
    y0 = samples.orientations[0, :, 1]
    y1 = samples.orientations[1, :, 1]
    projection = np.dot(y0, y1)
    assert np.allclose(np.linalg.norm(y0), 1)
    assert np.allclose(np.linalg.norm(y1), 1)
    assert np.allclose(projection, 1)

    # check for determinant 1 on all rotation matrices
    determinants = np.linalg.det(samples.orientations)
    assert np.allclose(determinants, 1)
