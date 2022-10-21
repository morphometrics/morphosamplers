"""Tooling to fit and sample surfaces."""

import math
from typing import List

import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, conint

from .spline import NDimensionalSpline
from .utils import align_orientations_to_z_vectors, generate_surface_normals


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
        max_length = max(len(p) for p in equidistant_points)
        padded_points = []
        for pts in equidistant_points:
            length = len(pts)
            pad = (max_length - length) / 2
            padded = np.pad(
                pts, ((math.ceil(pad), math.floor(pad)), (0, 0)), mode="edge"
            )
            padded_points.append(padded)

        cross_spline_points = np.stack(padded_points, axis=1)
        cross_splines = [
            NDimensionalSpline(p, order=self.order) for p in cross_spline_points
        ]
        return cross_splines

    def sample_surface(self, separation):
        """Sample equidistant points on the surface with approximate separation."""
        splines = self._generate_cross_splines(separation)
        equidistant_points = [
            spline._get_approximate_equidistance_spline_samples(separation)
            for spline in splines
        ]
        # return deduplicate_points(np.concatenate(equidistant_points), separation / 2)
        return np.concatenate(equidistant_points)

    def sample_surface_orientations(self, separation, inside_point):
        """Sample equidistant orientations on the surface with approx separation."""
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
