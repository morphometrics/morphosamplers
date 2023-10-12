import numpy as np
import einops

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

# calculate using calculate_best_fit_parameters in dev/sphere_params.py
A = 0.07726122
B = 0.02657319


def fibonacci_sphere(n):
    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-
    # sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    i = np.arange(n)
    phi = 2 * np.pi * i / GOLDEN_RATIO
    # this is the version that optimizes for mean distance
    eps = 0.36
    theta = np.arccos(1 - 2 * (i + eps) / (n - 1 + 2 * eps))
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return einops.rearrange([x, y, z], 'xyz b -> b xyz')


def spacing_from_n(n):
    return 1 / np.sqrt(A * n + B)


def n_from_spacing(spacing):
    return int(1 / (spacing ** 2 * A) - B / A)