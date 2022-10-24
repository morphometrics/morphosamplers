"""Utility functions."""

from typing import Tuple, Union

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


def minimize_point_strips_pair_distance(strips, mode="crop"):
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


def interpolate_and_extrapolate_point_strips(strips):
    """
    Extrapolate and interpolate a point strip containing nan values.

    Perform simple linear interpolation when possible, or if only one value
    is available, duplicates it.
    """
    extended_with_cross_mean = []
    all_z_values = np.stack(strips, axis=1)[:, :, 2]
    z_values = np.nanmean(all_z_values, axis=0)
    for cross_pts in np.stack(strips, axis=1):
        nans = np.isnan(cross_pts[:, 0])
        nan_idx = np.where(nans)[0]
        val_idx = np.where(~nans)[0]
        x, y, _ = cross_pts.T
        interpolated = []
        for coord in (x, y):
            fill_values = np.interp(nan_idx, val_idx, coord[val_idx])
            filled = coord.copy()
            filled[nan_idx] = fill_values
            interpolated.append(filled)
        interpolated.append(z_values)
        interpolated = np.stack(interpolated, axis=1)
        extended_with_cross_mean.append(interpolated)
    return extended_with_cross_mean
