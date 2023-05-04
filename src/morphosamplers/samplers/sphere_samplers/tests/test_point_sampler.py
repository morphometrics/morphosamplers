import numpy as np
from scipy.spatial import KDTree

from morphosamplers import MorphoModels
from morphosamplers.samplers import sphere_samplers


def test_point_sampler():
    """Should produce points with correct spacing on surface of sphere."""
    sphere = MorphoModels.Sphere(center=(5, 5, 5), radius=10)
    sampler = sphere_samplers.PointSampler(spacing=2)
    points = sampler.sample(sphere)

    # all points should be centered around (5, 5, 5)
    eps = 1e-2
    assert np.mean(points) - 5 < eps

    # spacing between points should be ~2
    tree = KDTree(points)
    distances, idx = tree.query(points, k=4)
    distances = distances[:, 1:]  # exclude distance to self
    assert 1.75 <= np.mean(distances) < 2.25


def test_pose_sampler():
    """Should produce poses oriented with Z normal to the surface of the sphere."""
    sphere = MorphoModels.Sphere(center=(5, 5, 5), radius=10)
    sampler = sphere_samplers.PoseSampler(spacing=2)
    poses = sampler.sample(sphere)

    # all points should be centered around (5, 5, 5)
    eps = 1e-2
    assert np.mean(poses.positions) - 5 < eps

    # points should be oriented with z normal to the sphere
    z_vecs_poses = poses.orientations[:, :, 2]
    z_dir_expected = poses.positions - np.array([5, 5, 5])
    z_dir_expected /= np.linalg.norm(z_dir_expected, axis=-1, keepdims=True)
    assert np.allclose(z_vecs_poses, z_dir_expected)

    # check that rotation matrices are in SO3
    assert np.allclose(np.linalg.det(poses.orientations), 1)
