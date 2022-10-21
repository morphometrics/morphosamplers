"""Tooling to fit and sample surfaces."""

from typing import List

import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, conint

from .spline import NDimensionalSpline
from .utils import align_orientations_to_z_vectors, generate_surface_normals, minimize_point_strips_pair_distance, deduplicate_points


class SurfaceSpline(EventedModel):
    """Surface model based on splines."""

    points: np.ndarray
    order: conint(ge=1, le=5) = 3
    _splines = PrivateAttr(List[NDimensionalSpline])

    class Config:
        """Pydantic BaseModel configuration."""

        arbitrary_types_allowed = True

    def __init__(self, points: np.ndarray, order: int = 3):
        """Calculate the splines after validating the paramters."""
        super().__init__(points=points, order=order)
        self._prepare_splines()

    def _prepare_splines(self):
        z_change_indices = np.where(np.diff(self.points[:, 2]))[0] + 1
        points_per_spline = np.split(self.points, z_change_indices)
        self._splines = [
            NDimensionalSpline(p, order=self.order) for p in points_per_spline
        ]

    def _generate_cross_splines(self, separation):
        equidistant_points = [
            spline._get_approximate_equidistance_spline_samples(separation)
            for spline in self._splines
        ]
        aligned = minimize_point_strips_pair_distance(equidistant_points, crop=False)
        cross_spline_points = np.stack(aligned, axis=1)
        cross_splines = [
            NDimensionalSpline(p, order=self.order) for p in cross_spline_points
        ]
        return cross_splines

    def sample_surface(self, separation):
        """Sample approximately equidistant points on the surface.

        Samples are optimized for maximum coverage of the surface, but will
        result in disordered points unsuitable for grid sampling.
        """
        splines = self._generate_cross_splines(separation)
        equidistant_points = [
            spline._get_approximate_equidistance_spline_samples(separation)
            for spline in splines
        ]
        return deduplicate_points(np.concatenate(equidistant_points), separation / 2)
        # return np.concatenate(equidistant_points)

    def sample_surface_orientations(self, separation, inside_point):
        """Sample approximately equidistant orientations on the surface."""
        splines = self._generate_cross_splines(separation)
        equidistant_points = [
            spline._get_approximate_equidistance_spline_samples(separation)
            for spline in splines
        ]
        equidistant_orientations = [
            spline._get_approximate_equidistance_orientations(separation)
            for spline in splines
        ]
        normals = generate_surface_normals(equidistant_points, inside_point)
        return align_orientations_to_z_vectors(equidistant_orientations, normals)

    def _generate_grid_cross_splines(self, separation):
        equidistant_points = [
            spline._get_approximate_equidistance_spline_samples(separation)
            for spline in self._splines
        ]
        aligned = minimize_point_strips_pair_distance(equidistant_points, crop=True)
        cross_spline_points = np.stack(aligned, axis=1)
        cross_splines = [
            NDimensionalSpline(p, order=self.order) for p in cross_spline_points
        ]
        return cross_splines

    def sample_surface_grid(self, separation):
        """Sample approximately equidistant points on the surface in a grid-like pattern.

        Samples are optimized for consistent separation and grid-like ordering, which
        results in many discarded edges if the input differs a lot from a rectangle.
        """
        splines = self._generate_grid_cross_splines(separation)
        us = [spline._get_approximate_equidistance_u(separation) for spline in splines]
        best_n = int(np.mean([len(u) for u in us]))
        u = np.linspace(0, 1, best_n)
        equidistant_points = [
            spline.sample_spline(u)
            for spline in splines
        ]
        # return deduplicate_points(np.concatenate(equidistant_points), separation / 2)
        return np.concatenate(equidistant_points)

    def sample_surface_grid_orientations(self, separation, inside_point):
        """Sample approximately equidistant orientations on the surface in a grid-like pattern."""
        splines = self._generate_grid_cross_splines(separation)
        us = [spline._get_approximate_equidistance_u(separation) for spline in splines]
        best_n = int(np.mean([len(u) for u in us]))
        u = np.linspace(0, 1, best_n)
        equidistant_points = [
            spline.sample_spline(u)
            for spline in splines
        ]
        equidistant_orientations = [
            spline.sample_spline_orientations(u)
            for spline in splines
        ]
        normals = generate_surface_normals(equidistant_points, inside_point)
        return align_orientations_to_z_vectors(equidistant_orientations, normals)
