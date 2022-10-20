from typing import Optional

import numpy as np


def calculate_y_vectors_from_z_vectors(
    z: np.ndarray,
    initial_y_vector: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate y vectors starting from z vectors.

    This function will return the set of unit vectors perpendicular to the z-vectors which are
    maximally coaxial to their neighbours. It assumes that z vectors (n, 3) vary
    smoothly with increasing n.
    """
    # normalise z vectors and initialise y
    z = z.copy()
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = np.empty((len(z), 3))

    # normalise initial y vector so that dot product is the projection
    if initial_y_vector is None:
        initial_y_vector = np.array([0, 0, 1])
    else:
        initial_y_vector = initial_y_vector / np.linalg.norm(initial_y_vector)

    # initialise first y
    yz = np.dot(initial_y_vector, z[0])  # projection of y on first z
    y[0] = initial_y_vector - yz * z[0]  # subtract z component of initial y

    # update y vectors in order
    for i in range(len(y) - 1):
        yz = np.dot(y[i], z[i + 1])  # projection of y on next z
        y[i + 1] = y[i] - yz * z[i + 1]  # subtract z component of current y
        y[i + 1] /= np.linalg.norm(y[i + 1])  # normalize each iteration to prevent precision issues
    return np.atleast_2d(y)
