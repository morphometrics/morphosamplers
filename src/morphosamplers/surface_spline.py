"""Tooling to fit and sample surfaces."""

from typing import List, Tuple, Optional
import warnings

import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, conint
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation

from .spline import Spline3D
from .utils import (
    extrapolate_point_strips_with_direction,
    minimize_point_strips_pair_distance,
    minimize_closed_point_strips_pair_distance,
)


class _SplineSurface(EventedModel):
    """Surface model based on splines."""

    points: List[np.ndarray]
    separation: float
    order: conint(ge=1, le=5) = 3
    smoothing: Optional[int] = None
    closed: bool = False
    _splines = PrivateAttr(List[Spline3D])
    _cross_splines = PrivateAttr(List[Spline3D])

    class Config:
        """Pydantic BaseModel configuration."""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Calculate the splines after validating the paramters."""
        super().__init__(**kwargs)
        self._prepare_splines()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("points", "separation", "order", "closed"):
            self._prepare_splines()

    def _prepare_splines(self):
        self._generate_splines()
        self._generate_cross_splines()

    def _fix_spline_edges(self, splines, separation, order):
        """Generate new control points for the given splines to fix edge artifacts"""
        # sample splines to get equidistant points on z slices
        us = [
            spline._get_equidistant_spline_coordinate_values(separation=separation, approximate=True)
            for spline in splines
        ]

        if self.closed:
            # we need the same amount of points on each spline to make a grid, and we can't just extend
            n_points = [len(u) for u in us]
            diff = np.max(n_points) - np.min(n_points)
            if diff > 1:
                warnings.warn('The grid is deformed by more than 1 separation in some places. '
                              'This is inevitable with closed, non-cylindrical grids.')
            best_n = round(np.mean(n_points))

            points = [spline.sample(n_samples=best_n) for spline, u in zip(splines, us)]

            points = minimize_closed_point_strips_pair_distance(points, expected_dist=separation)
        else:
            points = [spline.sample(u) for spline, u in zip(splines, us)]
            directions = [
                spline.sample(u, derivative_order=1)
                for spline, u in zip(splines, us)
            ]

            # extrapolate where nans are present by extending along the spline direction
            points = minimize_point_strips_pair_distance(points, expected_dist=separation, mode="nan")
            points = extrapolate_point_strips_with_direction(
                points, directions, separation
            )

        return points

    def _generate_splines(self):
        self._splines = [
            Spline3D(points=p, order=self.order, smoothing=self.smoothing, closed=self.closed)
            for p in self.points
        ]

    def _generate_cross_splines(self):
        # fix edge artifacts by extending splines until we have a grid
        control_points = self._fix_spline_edges(self._splines, self.separation, self.order)

        # if self.closed:
        #     # _fix_spline_edges returns strips including the extrema, so we need to remove
        #     # one of them if the surface is closed to avoid duplication
        #     control_points = [p[:-1] for p in control_points]

        # stack points in the other direction, so we get the cross-splines
        stacked = np.stack(control_points, axis=1)

        self._cross_splines = [Spline3D(points=p, order=self.order, smoothing=self.smoothing) for p in stacked]


# class SplineSurface(_SplineSurface):
#
#     def _generate_meta_splines(self):
#         equidistant_points = [
#             spline._get_approximate_equidistance_spline_samples(self.separation)
#             for spline in self._splines
#         ]
#         aligned = minimize_point_strips_pair_distance(equidistant_points, crop=False)
#         meta_spline_points = np.stack(aligned, axis=1)
#         self._meta_splines = [
#             Spline3D(points=p, order=self.order, self.smoothing) for p in meta_spline_points
#         ]
#
#     def sample(self, separation):
#         """Sample approximately equidistant points on the surface.
#
#         Samples are optimized for maximum coverage of the surface following
#         closely the annotation, but will result in disordered points
#         unsuitable for grid sampling.
#         """
#         splines = self._generate_meta_splines(separation)
#         equidistant_points = [
#             spline._get_approximate_equidistance_spline_samples(separation)
#             for spline in splines
#         ]
#         return deduplicate_points(np.concatenate(equidistant_points), separation / 2)
#
#
# def sample_orientations(self, separation, inside_point):
#     """Sample approximately equidistant orientations on the surface."""
#     splines = self._generate_meta_splines(separation)
#     equidistant_points = [
#         spline._get_approximate_equidistance_spline_samples(separation)
#         for spline in splines
#     ]
#     equidistant_orientations = [
#         spline._get_approximate_equidistance_orientations(separation)
#         for spline in splines
#     ]
#     normals = generate_surface_normals(equidistant_points, inside_point)
#     return align_orientations_to_z_vectors(equidistant_orientations, normals)


class SplineSurfaceGrid(_SplineSurface):
    """Surface model defined by a regular grid of splines."""

    _grid_splines = PrivateAttr(List[Tuple])
    _grid_meta_splines = PrivateAttr(List[Tuple])

    def _prepare_splines(self):
        super()._prepare_splines()
        self._generate_grid_splines()

    @property
    def grid_shape(self):
        return (len(self._grid_splines), len(self._grid_meta_splines))

    def _generate_grid_splines(self):
        # we finally create grid-like splines with optimized spacing
        # though not optimal for consistent spacing, we have to use the same n
        # for each spline, otherwise we end up with offset points in each lines
        us = [
            spline._get_equidistant_spline_coordinate_values(
                separation=self.separation,
                approximate=True
            )
            for spline in self._cross_splines
        ]
        n_points = [len(u) for u in us]
        diff = np.max(n_points) - np.min(n_points)
        if diff > 1:
            warnings.warn('The grid is deformed by more than 1 separation in some places. '
                          'Consider passing points that belong to parallel planes, or splitting '
                          'the surface in quasi-planar patches.')
        best_n = int(round(np.mean(n_points)))
        u = np.linspace(0, 1, best_n)
        # TODO: actually use non-approximate linspace (with remainder) so we get as close as possible to
        #       a real grid. The problem with this is that we won't reach exactly the last spline

        equidistant_points = [spline.sample(u) for spline in self._cross_splines]

        # these last splines should not be oversampled, because we want exact
        # positions for our knots, which we save in self._meta_meta_splines_u
        splines = []
        for p in np.stack(equidistant_points, axis=1):
            splines.append(splprep(p.T, s=0, k=self.order))
        self._grid_splines = splines

        # then generate the cross grid-splines to ensure they point exactly to the same
        # coordinates (the knots are shared), so we can get perfect orientations
        splines = []
        for p in np.stack(equidistant_points, axis=0):
            splines.append(splprep(p.T, s=0, k=self.order))
        self._grid_meta_splines = splines

    def sample(self):
        """Sample an approximately equidistant grid of points on the surface.

        Samples are optimized for consistent separation and grid-like ordering, which
        results in many discarded edges if the input differs a lot from a rectangle.
        """
        equidistant_points = [
            np.stack(splev(u, tck), axis=1)
            for tck, u in self._grid_splines
        ]
        return np.concatenate(equidistant_points)

    def sample_orientations(self, inside_point=None):
        """Sample an approximately equidistant grid of orientations on the surface."""
        equidistant_x_vecs = [
            np.stack(splev(u, tck, der=1), axis=1)
            for tck, u in self._grid_splines
        ]
        equidistant_y_vecs = [
            np.stack(splev(u, tck, der=1), axis=1)
            for tck, u in self._grid_meta_splines
        ]

        x = np.concatenate(equidistant_x_vecs)
        # y vectors are going across, so we need to swap axes
        y = np.concatenate(np.swapaxes(equidistant_y_vecs, 0, 1))

        y /= np.linalg.norm(y, axis=1, keepdims=True)
        x /= np.linalg.norm(x, axis=1, keepdims=True)

        z = np.cross(x, y)
        z /= np.linalg.norm(z, axis=1, keepdims=True)

        rots = Rotation.from_matrix(np.stack([x, y, z], axis=-1))

        if inside_point is not None:
            # use the inside point to put normal in correct direction
            pts = self.sample()
            # get closest neighbour
            diffs = pts - inside_point
            closest_idx = np.argmin(np.linalg.norm(diffs, axis=1))
            z_closest = z[closest_idx]
            z_inside = diffs[closest_idx]
            # flip is the point is outside
            if np.dot(z_inside, z_closest) < 0:
                rots *= Rotation.from_euler("x", np.pi)

        return rots


class SplineSurfaceHex(_SplineSurface):
    """Surface model defined by a hexagonal grid of splines"""

    _hex_splines_even = PrivateAttr(List[Tuple])
    _hex_splines_odd = PrivateAttr(List[Tuple])
    _hex_meta_splines_even = PrivateAttr(List[Tuple])
    _hex_meta_splines_odd = PrivateAttr(List[Tuple])

    def _prepare_splines(self):
        super()._prepare_splines()
        self._generate_hex_splines()

    def _generate_hex_splines(self):
        # we finally create grid-like splines with optimized spacing
        # though not optimal for consistent spacing, we have to use the same n
        # for each spline, otherwise we end up with offset points in each lines
        us = [spline._get_equidistant_spline_coordinate_values(separation=self.separation, approximate=True) for spline in self._cross_splines]
        best_n = int(np.mean([len(u) for u in us]))

        us = []
        for i, spline in enumerate(self._cross_splines):
            if i % 2:
                us.append(np.linspace(0, 1, best_n))
            else:
                offset = self.separation / spline._length / 2
                us.append(np.linspace(offset, 1 - offset, best_n - 1))

        equidistant_points = [
            spline.sample(u)
            for spline, u in zip(self._cross_splines, us)
        ]

        # alternate each cross spline to have points on same level
        splines_even = []
        for p in np.stack(equidistant_points[::2], axis=1):
            splines_even.append(splprep(p.T, s=0, k=self.order))
        self._hex_splines_even = splines_even
        splines_odd = []
        for p in np.stack(equidistant_points[1::2], axis=1):
            splines_odd.append(splprep(p.T, s=0, k=self.order))
        self._hex_splines_odd = splines_odd

        # cross splines
        splines_even = []
        for p in np.stack(equidistant_points[::2], axis=0):
            splines_even.append(splprep(p.T, s=0, k=self.order))
        self._hex_meta_splines_even = splines_even
        splines_odd = []
        for p in np.stack(equidistant_points[1::2], axis=0):
            splines_odd.append(splprep(p.T, s=0, k=self.order))
        self._hex_meta_splines_odd = splines_odd

    def sample(self):
        """Sample an approximately equidistant grid of points on the surface.

        Samples are optimized for consistent separation and grid-like ordering, which
        results in many discarded edges if the input differs a lot from a rectangle.
        """
        pts = []
        for splines in (self._hex_splines_even, self._hex_splines_odd):
            equidistant_points = [
                np.stack(splev(u, tck), axis=1)
                for tck, u in splines
            ]
            pts.append(np.concatenate(equidistant_points))
        return np.concatenate(pts)

    def sample_orientations(self, inside_point=None):
        """Sample an approximately equidistant grid of orientations on the surface."""
        rots = []
        for splines, meta_splines in (
            (self._hex_splines_even, self._hex_meta_splines_even),
            (self._hex_splines_odd, self._hex_meta_splines_odd),
        ):
            equidistant_x_vecs = [
                np.stack(splev(u, tck, der=1), axis=1)
                for tck, u in splines
            ]
            equidistant_y_vecs = [
                np.stack(splev(u, tck, der=1), axis=1)
                for tck, u in meta_splines
            ]

            x = np.concatenate(equidistant_x_vecs)
            # y vectors are going across, so we need to swap axes
            y = np.concatenate(np.swapaxes(equidistant_y_vecs, 0, 1))

            x /= np.linalg.norm(x, axis=1, keepdims=True)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

            z = np.cross(x, y)
            z /= np.linalg.norm(z, axis=1, keepdims=True)

            rots.append(Rotation.from_matrix(np.stack([x, y, z], axis=-1)))

        rots = Rotation.concatenate(rots)

        if inside_point is not None:
            # use the inside point to put normal in correct direction
            pts = self.sample()
            # get closest neighbour
            diffs = pts - inside_point
            closest_idx = np.argmin(np.linalg.norm(diffs, axis=1))
            z_closest = z[closest_idx]
            z_inside = diffs[closest_idx]
            # flip is the point is outside
            if np.dot(z_inside, z_closest) < 0:
                rots *= Rotation.from_euler("x", np.pi)

        return rots
