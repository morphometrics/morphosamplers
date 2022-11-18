"""Utility functions."""

from typing import Tuple, Union
import warnings

import numpy as np
from scipy.spatial import KDTree


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


def deduplicate_points(coords: np.ndarray, exclusion_radius: float) -> np.ndarray:
    """Remove "duplicates" points from an array of coordinates.

    Points are clustered with their neighbours if they are closer than exclusion_radius.
    Clusters are iteratively replaced with their centroid until no clusters remain.
    """
    coords = coords.copy()
    while True:
        tree = KDTree(coords)
        # get clusters sorted by size (biggest first)
        clusters = tree.query_ball_point(coords, exclusion_radius, workers=-1)
        clusters_sorted = sorted(clusters, key=len, reverse=True)
        biggest = clusters_sorted[0]
        if len(biggest) == 1:
            # we reached the degenerate clusters of points with themselves
            break
        centroid = np.mean(coords[biggest], axis=0)
        coords[biggest[0]] = centroid

        mask = np.ones(len(coords), np.bool)
        mask[biggest[1:]] = False
        coords = coords[mask]

    return coords


def minimize_closed_point_strips_pair_distance(strips, expected_dist=None):
    # assume closed, circular strips with equal length
    result = [strips[0]]
    ln = len(strips[0])
    for arr, next in zip(strips, strips[1:]):
        best_dist = None
        for i in range(ln):
            next_rolled = np.roll(next, i, axis=0)
            roll_dist = np.linalg.norm(arr - next, axis=1)
            avg_dist = np.mean(roll_dist)
            if best_dist is None or avg_dist < best_dist:
                best_dist = avg_dist
                best_arr = next_rolled
        if expected_dist is not None and best_dist >= expected_dist * np.sqrt(2):
            warnings.warn('The grid is sheared by more than 1 separation in some places', stacklevel=2)
        result.append(best_arr)
    return result


def minimize_point_strips_pair_distance(strips, mode="crop", expected_dist=None):
    """Minimize average pair distance at the same index between any number of point strips.

    Rolls each strip in order to minimize the euclidean distance
    of each point in the strip relative to the points at the same index in the
    neighbouring strips.

    If mode is crop, crop all the strips so there are only valid values.
    Otherwise, strips are padded with either their edge value or nans.
    """
    modes = ("crop", "nan", "edge")
    if mode not in modes:
        raise ValueError(f"mode must be one of: {modes}")

    min_idx = 0
    max_idx = max(len(s) for s in strips)
    offsets = [0]
    for arr, next in zip(strips, strips[1:]):
        arr_padded = np.pad(
            arr, ((len(next), len(next)), (0, 0)), constant_values=np.nan
        )
        next_padded = np.pad(
            next, ((0, len(next) + len(arr)), (0, 0)), constant_values=np.nan
        )
        tot_len = len(next) + len(arr) + len(next)
        best_roll_idx = None
        best_dist = None
        for i in range(tot_len):
            next_rolled = np.roll(next_padded, i, axis=0)
            roll_dist = np.linalg.norm(arr_padded - next_rolled, axis=1)
            avg_dist = np.nanmean(roll_dist)
            if np.isnan(avg_dist):
                continue
            if best_dist is None or avg_dist < best_dist:
                best_dist = avg_dist
                best_roll_idx = i
        if expected_dist is not None and best_dist >= expected_dist * np.sqrt(2):
            warnings.warn('The grid is sheared by more than 1 separation in some places', stacklevel=2)
        offset = best_roll_idx - len(next)
        total_offset = offset + offsets[-1]
        offsets.append(total_offset)

    # construct final aligned arrays
    aligned = []

    if mode == "crop":
        min_idx = max(offsets)
        max_idx = min(o + len(a) for o, a in zip(offsets, strips))
        for arr, offset in zip(strips, offsets):
            cropped = arr[min_idx - offset : max_idx - offset]
            aligned.append(cropped)
    else:
        if mode == "edge":
            kwargs = {"mode": "edge"}
        elif mode == "nan":
            kwargs = {"mode": "constant", "constant_values": np.nan}
        min_idx = min(offsets)
        max_idx = max(o + len(a) for o, a in zip(offsets, strips))
        for arr, offset in zip(strips, offsets):
            padded = np.pad(
                arr, ((offset - min_idx, max_idx - offset - len(arr)), (0, 0)), **kwargs
            )
            aligned.append(padded)
    return aligned


def extrapolate_point_strips_with_direction(strips, directions, separation):
    """
    Extrapolate point strips padded with nans by continuing along a direction.

    Extrapolates beyond the first and last finite value in strip continuing
    in the correct direction and spacing by separation.
    """
    extrapolated = []
    for strip, dir in zip(strips, directions):
        nans = np.isnan(strip[:, 0])
        left_pad = np.argmax(~nans)
        left_dir = -dir[0]
        left_shift = left_dir / np.linalg.norm(left_dir) * separation
        right_pad = np.argmax(~nans[::-1])
        right_dir = dir[-1]
        right_shift = right_dir / np.linalg.norm(right_dir) * separation
        left_extension = strip[left_pad] + left_shift * np.arange(
            left_pad, 0, -1
        ).reshape(-1, 1)
        right_extension = strip[-right_pad - 1] + right_shift * np.arange(
            1, right_pad + 1
        ).reshape(-1, 1)
        padded = np.concatenate(
            [left_extension, strip[left_pad : -right_pad or None], right_extension]
        )
        extrapolated.append(padded)
    return extrapolated


def within_range(arr, low, high, atol=1e-15):
    """Determine which elements in an array are within a range, with a given tolerance."""
    diff_from_min = arr - low
    diff_from_max = arr - high
    above_min = (diff_from_min >= 0) | np.isclose(diff_from_min, 0, atol=atol)
    below_max = (diff_from_max <= 0) | np.isclose(diff_from_max, 0, atol=atol)
    return above_min & below_max


def get_mask_limits(arr):
    """Return the indices of the first and last True elements in a bool array."""
    first = np.argmax(arr)
    last = -np.argmax(arr[::-1]) or -1
    return first, last


def strip_true(arr):
    """Strip the leading and trailing False from a bool array."""
    first, last = get_mask_limits(arr)
    return arr[first:last]


def get_mask_limits(mask):
    mask = np.pad(mask, 1)
    beginnings = (mask ^ np.roll(mask, 1)) & mask
    beginnings = np.where(beginnings)[0]
    ends = (mask ^ np.roll(mask, -1)) & mask
    ends = np.where(ends)[0]
    return [(b - 1, e - 1) for b, e in zip(beginnings, ends) if b != e]
