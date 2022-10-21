"""Utility functions."""


from typing import Optional

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


def calculate_y_vectors_from_z_vectors(
    z: np.ndarray, initial_y_vector: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate y vectors starting from z vectors.

    This function will return the set of unit vectors perpendicular to the z-vectors
    which are maximally coaxial to their neighbours. It assumes that z vectors (n, 3)
    vary smoothly with increasing n.
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
        y[i + 1] /= np.linalg.norm(
            y[i + 1]
        )  # normalize each iteration to prevent precision issues
    return np.atleast_2d(y)


def deduplicate_points(coords: np.ndarray, exclusion_radius: float) -> np.ndarray:
    """Remove "duplicates" points from an array of coordinates.

    Points are clustered with their neighbours if they are closer than exclusion_radius.
    Points are iteratively removed from each cluster,
    prioritizing keeping hub-like points.
    """
    tree = KDTree(coords)
    clusters = tree.query_ball_point(coords, exclusion_radius, workers=-1)
    clusters_sorted = sorted(clusters, key=len, reverse=True)

    duplicates = []
    visited = []
    for cluster in clusters_sorted:
        if len(cluster) == 1:
            break
        duplicates_in_cluster = [el for el in cluster[1:] if el not in visited]
        duplicates.extend(duplicates_in_cluster)
        visited.append(cluster[0])

    duplicates = np.array(sorted(list(set(duplicates)))).astype(int)
    mask = np.ones(len(coords), dtype=bool)
    mask[duplicates] = False

    return coords[mask]


def generate_surface_normals(surface_points, inside_point):
    """Generate normal vectors for points on a surface based on a point inside."""
    vectors = surface_points - inside_point
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def align_orientations_to_z_vectors(orientations, vectors):
    """Align orientations to given z vectors with minimal change."""
    z = np.array([[0, 0, 1]])
    aligned = []
    for ori, vec in zip(orientations, vectors):
        aligned.append(ori.align_vectors(z, np.atleast_2d(vec)))
    return Rotation.concatenate(aligned)


# equidistant but keep edges is needed for surface!
