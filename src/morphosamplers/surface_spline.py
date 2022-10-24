"""Tooling to fit and sample surfaces."""

from typing import List

import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, conint
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation

from .spline import Spline3D
from .utils import (
    extrapolate_point_strips_with_direction,
    interpolate_and_extrapolate_point_strips,
    minimize_point_strips_pair_distance,
)


class _SplineSurface(EventedModel):
    """Surface model based on splines."""

    points: np.ndarray
    separation: float
    order: conint(ge=1, le=5) = 3
    _splines = PrivateAttr(List[Spline3D])
    _meta_splines = PrivateAttr(List[Spline3D])
    _meta_meta_splines_tck = PrivateAttr(List[Spline3D])
    _meta_meta_splines_u = PrivateAttr(List[Spline3D])
    _meta_meta_meta_splines_tck = PrivateAttr(List[Spline3D])
    _meta_meta_meta_splines_u = PrivateAttr(List[Spline3D])

    class Config:
        """Pydantic BaseModel configuration."""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Calculate the splines after validating the paramters."""
        super().__init__(**kwargs)
        self._prepare_splines()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("points", "separation", "order"):
            self._prepare_splines()

    def _prepare_splines(self):
        self._generate_splines()
        self._generate_meta_splines()
        self._generate_meta_meta_splines()
        self._generate_meta_meta_meta_splines()

    def _generate_splines(self):
        z_change_indices = np.where(np.diff(self.points[:, 2]))[0] + 1
        points_per_spline = np.split(self.points, z_change_indices)
        self._splines = [Spline3D(p, order=self.order) for p in points_per_spline]

    def _generate_meta_splines(self):
        # sample splines to get equidistant points on z slices
        us = [
            spline._get_approximate_equidistance_u(self.separation)
            for spline in self._splines
        ]
        equidistant_points = [
            spline.sample_spline(u) for spline, u in zip(self._splines, us)
        ]
        # align and pad with nans
        aligned = minimize_point_strips_pair_distance(equidistant_points, mode="nan")
        # TODO: keep track of where we are "outside" of the raw data so we can
        # drop it later

        equidistant_directions = [
            spline.sample_spline(u, derivative_order=1)
            for spline, u in zip(self._splines, us)
        ]
        # extrapolate where nans are present by extending along the spline direction
        extended = extrapolate_point_strips_with_direction(
            aligned, equidistant_directions, self.separation
        )
        # extrapolate and interpolate in the other direction using linear interpolation
        extended_across = interpolate_and_extrapolate_point_strips(aligned)

        meta_spline_control_points = np.nanmean(
            [np.stack(extended, axis=1), extended_across], axis=0
        )

        self._meta_splines = [
            Spline3D(p, order=self.order) for p in meta_spline_control_points
        ]

    def _generate_meta_meta_splines(self):
        us = [
            spline._get_approximate_equidistance_u(self.separation)
            for spline in self._meta_splines
        ]
        best_n = int(np.mean([len(u) for u in us]))
        u = np.linspace(0, 1, best_n)
        equidistant_points = [spline.sample_spline(u) for spline in self._meta_splines]

        # these last splines should not be oversampled, because we want exact
        # positions for our knots, which we save in self._meta_meta_splines_u
        tcks = []
        us = []
        for p in np.stack(equidistant_points, axis=1):
            (
                tck,
                u,
            ) = splprep(p.T, s=0, k=self.order)
            tcks.append(tck)
            us.append(u)
        self._meta_meta_splines_tck = tcks
        self._meta_meta_splines_u = us

    def _generate_meta_meta_meta_splines(self):
        equidistant_points = [
            np.stack(splev(u, tck), axis=1)
            for u, tck in zip(self._meta_meta_splines_u, self._meta_meta_splines_tck)
        ]

        # these last splines should not be oversampled, because we want exact
        # positions for our knots, which we save in self._meta_meta_splines_u
        tcks = []
        us = []
        for p in np.stack(equidistant_points, axis=1):
            (
                tck,
                u,
            ) = splprep(p.T, s=0, k=self.order)
            tcks.append(tck)
            us.append(u)
        self._meta_meta_meta_splines_tck = tcks
        self._meta_meta_meta_splines_u = us


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
#             Spline3D(p, order=self.order) for p in meta_spline_points
#         ]
#
#     def sample_surface(self, separation):
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
#         # return np.concatenate(equidistant_points)
#
#
# def sample_surface_orientations(self, separation, inside_point):
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
    """Surface model defined by a grid of splines."""

    def sample_surface(self):
        """Sample an approximately equidistant grid of points on the surface.

        Samples are optimized for consistent separation and grid-like ordering, which
        results in many discarded edges if the input differs a lot from a rectangle.
        """
        equidistant_points = [
            np.stack(splev(u, tck), axis=1)
            for u, tck in zip(self._meta_meta_splines_u, self._meta_meta_splines_tck)
        ]
        return np.concatenate(equidistant_points)

    def sample_surface_orientations(self, inside_point=None):
        """Sample an approximately equidistant grid of orientations on the surface."""
        equidistant_x_vecs = [
            np.stack(splev(u, tck, der=1), axis=1)
            for u, tck in zip(self._meta_meta_splines_u, self._meta_meta_splines_tck)
        ]
        equidistant_y_vecs = [
            np.stack(splev(u, tck, der=1), axis=1)
            for u, tck in zip(
                self._meta_meta_meta_splines_u, self._meta_meta_meta_splines_tck
            )
        ]
        x = np.concatenate(equidistant_x_vecs)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # y vectors are going across, so we need to swap axes
        y = np.concatenate(np.swapaxes(equidistant_y_vecs, 0, 1))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        z = np.cross(x, y)
        z /= np.linalg.norm(z, axis=1, keepdims=True)

        rots = Rotation.from_matrix(np.stack([x, y, z], axis=-1))

        if inside_point is not None:
            # use the inside point to put normal in correct direction
            equidistant_points = [
                np.stack(splev(u, tck), axis=1)
                for u, tck in zip(
                    self._meta_meta_splines_u, self._meta_meta_splines_tck
                )
            ]
            pts = np.concatenate(equidistant_points)
            # get closest neighbour
            diffs = pts - inside_point
            closest_idx = np.argmin(np.linalg.norm(diffs, axis=1))
            z_closest = z[closest_idx]
            z_inside = diffs[closest_idx]
            # TODO: why is closest so small? something is off
            if np.dot(z_inside, z_closest) < 0:
                rots *= Rotation.from_euler("x", np.pi)

        return rots
