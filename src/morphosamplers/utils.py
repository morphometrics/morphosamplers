"""Utility functions."""

from typing import Tuple, Union

import numpy as np


def _project_vector_onto_plane(vector, plane_normal):
    """Project vector onto a plane defined by its normal.

    Inputs should be normalized; output will be normalized.
    """
    normal_component = np.dot(vector, plane_normal)
    if normal_component == 0:
        # perfectly aligned
        return vector
    proj_plane = vector - normal_component * plane_normal
    proj_plane /= np.linalg.norm(proj_plane)
    return proj_plane


def calculate_y_vectors_from_z_vectors(
    z: np.ndarray,
    initial_y_vector: Union[np.ndarray, Tuple[float, float, float]] = (0.3234, 0.6543, 0.978),
) -> np.ndarray:
    """Calculate y vectors starting from z vectors.

    This function will return the set of unit vectors perpendicular to the z-vectors
    which are maximally coaxial to their neighbours. It assumes that z vectors (n, 3)
    vary smoothly with increasing n.

    Parameters
    ----------
    z : np.ndarray (n, 3)
        The z vectors used to generate y vectors.
    initial_y_vector: Union[np.ndarray, Tuple[float, float, float]]
        The starting y vector projected onto the first z vector to start the projection
        procedure. The default value is weird to make it less likely to match a manual
        entry such as [0, 0, 1] or other simple values.

    Returns
    -------
    y : np.ndarray (n, 3)
        The computed y vectors.
    """
    # normalise z vectors and initialise y
    z = z.copy()
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = np.empty((len(z), 3))

    if np.dot(initial_y_vector, z[0]) == 1:
        raise ValueError('cannot generate y vectors because the provided initial_y_vector '
                         'and the first z vector are perfectly aligned.')
    # normalise initial y vector so that dot product is the projection
    initial_y_vector = initial_y_vector / np.linalg.norm(initial_y_vector)

    # initialise first y
    y[0] = _project_vector_onto_plane(initial_y_vector, z[0])

    # update y vectors in order
    for i in range(len(y) - 1):
        y[i + 1] = _project_vector_onto_plane(y[i], z[i + 1])
    return np.atleast_2d(y)
