import einops
import numpy as np

from morphosamplers import path_samplers, Path

def test_instantiation():
    sampler = path_samplers.PointSampler(spacing=5)
    assert isinstance(sampler, path_samplers.PointSampler)


def test_sampling():
    """Samples should interpolate control points"""
    n_points = 50
    total_length = 100
    spacing = total_length / n_points
    x = y = np.zeros(shape=(n_points, ))
    z = np.linspace(0, total_length, num=n_points)
    control_points = einops.rearrange([x, y, z], 'xyz b -> b xyz')
    path = Path(control_points=control_points)
    sampler = path_samplers.PointSampler(spacing=spacing)
    samples = sampler.sample(path)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    expected = np.linspace([0, 0, 0], [0, 0, total_length], num=n_points)
    assert np.allclose(samples, expected)