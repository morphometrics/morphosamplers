"""Tooling to fit and sample surfaces."""

from typing import List, Tuple, Optional, Union
import warnings

import numpy as np
import einops
from psygnal import EventedModel
from pydantic import PrivateAttr, conint
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation

from .spline import Spline3D
from .utils import (
    extrapolate_point_strips_with_direction,
    minimize_point_row_pair_distance,
    minimize_closed_point_strips_pair_distance,
)


class _SplineSurface(EventedModel):
    """Surface model based on splines."""

    points: List[np.ndarray]
    separation: float
    order: conint(ge=1, le=5) = 3
    smoothing: Optional[int] = None
    closed: bool = False
    inside_point: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None
    _row_splines = PrivateAttr(List[Spline3D])
    _column_splines = PrivateAttr(List[Spline3D])

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
        self._generate_row_splines()
        self._generate_column_splines()

    def _fix_spline_edges(self, splines, separation, order):
        """
        Generate new control points for each spline to ensure a minimally deformed grid.

        For open surfaces, splines are sampled with the same equidistance, and the resulting
        rows of points are then padded with NaNs at each end so that:
        - they have the same amount of samples
        - mean distance between samples at the same index in neighboring rows is minimised
        NaN values are then replaced by extrapolating the existing values along the spline
        derivative.

        For closed surfaces, since they cannot be extended, each spline is sampled with the same
        number of points, which may result in a deformed grid. Each row is then aligned to minimise
        shear by minimising the mean distance between samples at the same intex in neighboring rows.
        """
        # sample splines to get equidistant points on z slices
        us = [
            spline._get_equidistant_spline_coordinate_values(separation=separation, approximate=True)
            for spline in splines
        ]

        if self.closed:
            # We need the same amount of samples on each row to make a rectangular grid,
            # but we can't extend/pad a closed surface, so instead we take n_samples from each
            # so that on average we get a good approximation of the given separation
            n_points = [len(u) for u in us]
            diff = np.max(n_points) - np.min(n_points)
            if diff > 1:
                warnings.warn('The grid is deformed by more than 1 separation in some places. '
                              'This is inevitable with closed, non-cylindrical grids.')
            best_n = round(np.mean(n_points))

            points = [spline.sample(n_samples=best_n) for spline, u in zip(splines, us)]

            points = minimize_closed_point_strips_pair_distance(points, expected_dist=separation)
            masks = [np.ones(len(p), dtype=bool) for p in points]
        else:
            points = [spline.sample(u) for spline, u in zip(splines, us)]
            directions = [
                spline.sample(u, derivative_order=1)
                for spline, u in zip(splines, us)
            ]

            # extrapolate where nans are present by extending along the spline direction
            points = minimize_point_row_pair_distance(points, expected_dist=separation, mode="nan")

            masks = [~np.isnan(p[:, 0]) for p in points]  # just one dim is enough
            points = extrapolate_point_strips_with_direction(
                points, directions, separation
            )

        return points, masks

    def _generate_row_splines(self):
        self._row_splines = [
            Spline3D(points=p, order=self.order, smoothing=self.smoothing, closed=self.closed)
            for p in self.points
        ]

    def _generate_column_splines(self):
        # fix edge artifacts by extending splines until we have a grid
        control_points, masks = self._fix_spline_edges(self._row_splines, self.separation, self.order)

        if self.closed:
            # _fix_spline_edges returns strips including the extrema, so we need to remove
            # one of them if the surface is closed to avoid duplication
            control_points = [p[:-1] for p in control_points]

        # stack points in the other direction, so we get the column-splines
        stacked = einops.rearrange(control_points, 'row column xyz -> column row xyz')
        masks_stacked = einops.rearrange(masks, 'row column -> column row')

        self._column_splines = [
            Spline3D(points=p, order=self.order, smoothing=self.smoothing, mask=mask)
            for p, mask in zip(stacked, masks_stacked)
        ]


class GriddedSplineSurface(_SplineSurface):
    """Surface model defined by a regular grid of splines."""

    _grid_splines = PrivateAttr(List[Tuple])
    _grid_meta_splines = PrivateAttr(List[Tuple])
    _mask = PrivateAttr(np.ndarray)

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
            for spline in self._column_splines
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

        equidistant_points = [spline.sample(u) for spline in self._column_splines]
        masks = [spline.get_mask_for_u(u) for spline in self._column_splines]
        self._mask = np.concatenate(np.stack(masks, axis=1))

        # these last splines should not be oversampled, because we want exact
        # positions for our knots, which we save in self._meta_meta_splines_u
        splines = []
        for p in np.stack(equidistant_points, axis=1):
            splines.append(splprep(p.T, s=0, k=self.order))
        self._grid_splines = splines

        # then generate the column grid-splines to ensure they point exactly to the same
        # coordinates (the knots are shared), so we can get perfect orientations
        splines = []
        for p in np.stack(equidistant_points, axis=0):
            splines.append(splprep(p.T, s=0, k=self.order))
        self._grid_meta_splines = splines

    def sample(self) -> np.ndarray:
        """Sample an approximately equidistant grid of points on the surface.

        Samples are optimized for consistent separation and grid-like ordering, which
        results in many discarded edges if the input differs a lot from a rectangle.

        The sampled array has shape (rows * columns, 3) for a grid of shape
        (rows, columns) and follows the (rows, columns) order.
        """
        equidistant_points = [
            einops.rearrange(splev(u, tck), 'xyz column -> column xyz')
            for tck, u in self._grid_splines
        ]
        return np.concatenate(equidistant_points)

    @property
    def mask(self):
        return self._mask

    def sample_orientations(self):
        """Sample an approximately equidistant grid of orientations on the surface.

        Follows the same pattern as GriddedSplineSurface.sample(). Orientations are
        generated so that the basis z vector is aligned to the surface normal, while
        x and y vectors are aligned to the row and column splines respectively.
        """
        equidistant_x_vecs = [
            einops.rearrange(splev(u, tck, der=1), 'xyz column -> column xyz')
            for tck, u in self._grid_splines
        ]
        equidistant_y_vecs = [
            einops.rearrange(splev(u, tck, der=1), 'xyz row -> row xyz')
            for tck, u in self._grid_meta_splines
        ]

        x = einops.rearrange(equidistant_x_vecs, 'row column xyz -> (row column) xyz')
        y = einops.rearrange(equidistant_y_vecs, 'column row xyz -> (row column) xyz')

        y /= np.linalg.norm(y, axis=1, keepdims=True)
        x /= np.linalg.norm(x, axis=1, keepdims=True)

        z = np.cross(x, y)
        z /= np.linalg.norm(z, axis=1, keepdims=True)

        rots = Rotation.from_matrix(np.stack([x, y, z], axis=-1))

        if self.inside_point is not None:
            # use the inside point to put normal in correct direction
            pts = self.sample()
            # get closest neighbour
            diffs = pts - self.inside_point
            closest_idx = np.argmin(np.linalg.norm(diffs, axis=1))
            z_closest = z[closest_idx]
            z_inside = diffs[closest_idx]
            # flip is the point is outside
            if np.dot(z_inside, z_closest) < 0:
                rots *= Rotation.from_euler("x", np.pi)

        return rots

    def mesh(self) -> Tuple[np.array, np.array]:
        """
        A mesh representation of the surface.

        For each quad in the grid, indices are generated following this pattern:
        0--2  x--2
        | /|  | /|
        |/ |  |/ |
        1--x  0--1

        Returns
        -------
        Tuple[np.array, np.array]
            Vertices coordinates (n, 3) and indices of vertices forming triangle faces (m, 3)
        """
        points = self.sample()
        rows, columns = self.grid_shape
        row_range = np.arange(rows)
        column_range = np.arange(columns)

        # first half of triangles
        first_index = np.repeat(row_range[:-1], columns - 1) * columns + np.tile(column_range[:-1], rows - 1)
        second_index = np.repeat(row_range[1:], columns - 1) * columns + np.tile(column_range[:-1], rows - 1)
        third_index = np.repeat(row_range[:-1], columns - 1) * columns + np.tile(column_range[1:], rows - 1)
        triangles_1 = np.stack([first_index, second_index, third_index], axis=1)

        # second half
        first_index = np.repeat(row_range[1:], columns - 1) * columns + np.tile(column_range[:-1], rows - 1)
        second_index = np.repeat(row_range[1:], columns - 1) * columns + np.tile(column_range[1:], rows - 1)
        third_index = np.repeat(row_range[:-1], columns - 1) * columns + np.tile(column_range[1:], rows - 1)
        triangles_2 = np.stack([first_index, second_index, third_index], axis=1)

        all_triangles = np.concatenate([triangles_1, triangles_2])
        return points, all_triangles


class HexSplineSurface(_SplineSurface):
    """Surface model defined by a grid of splines which generate a hex grid of samples."""

    _row_splines_even = PrivateAttr(List[Tuple])
    _row_splines_odd = PrivateAttr(List[Tuple])
    _column_splines_even = PrivateAttr(List[Tuple])
    _column_splines_odd = PrivateAttr(List[Tuple])
    _mask = PrivateAttr(np.ndarray)

    def _prepare_splines(self):
        super()._prepare_splines()
        self._generate_offset_splines()

    def _generate_offset_splines(self):
        # we finally create grid-like splines with optimized spacing
        # though not optimal for consistent spacing, we have to use the same n
        # for each spline, otherwise we end up with offset points in each lines
        us = [spline._get_equidistant_spline_coordinate_values(separation=self.separation, approximate=True) for spline in self._column_splines]
        best_n = int(np.mean([len(u) for u in us]))

        us = []
        for i, spline in enumerate(self._column_splines):
            if i % 2:
                us.append(np.linspace(0, 1, best_n))
            else:
                offset = self.separation / spline._length / 2
                us.append(np.linspace(offset, 1 - offset, best_n - 1))

        equidistant_points = [
            spline.sample(u)
            for spline, u in zip(self._column_splines, us)
        ]

        masks = [spline.get_mask_for_u(u) for spline, u in zip(self._column_splines, us)]
        masks_even = np.concatenate(np.stack(masks[::2], axis=1))
        masks_odd = np.concatenate(np.stack(masks[1::2], axis=1))
        self._mask = np.concatenate([masks_even, masks_odd])

        # alternate each row spline to have points on same level
        splines_even = []
        for p in np.stack(equidistant_points[::2], axis=1):
            splines_even.append(splprep(p.T, s=0, k=self.order))
        self._row_splines_even = splines_even

        splines_odd = []
        for p in np.stack(equidistant_points[1::2], axis=1):
            splines_odd.append(splprep(p.T, s=0, k=self.order))
        self._row_splines_odd = splines_odd

        # then alternate columns splines
        splines_even = []
        for p in np.stack(equidistant_points[::2], axis=0):
            splines_even.append(splprep(p.T, s=0, k=self.order))
        self._column_splines_even = splines_even
        splines_odd = []
        for p in np.stack(equidistant_points[1::2], axis=0):
            splines_odd.append(splprep(p.T, s=0, k=self.order))
        self._column_splines_odd = splines_odd

    def sample(self):
        """Sample an approximately equidistant grid of points on the surface.

        Samples are optimized for consistent separation and grid-like ordering, which
        results in many discarded edges if the input differs a lot from a rectangle.
        """
        pts = []
        for splines in (self._row_splines_even, self._row_splines_odd):
            equidistant_points = [
                np.stack(splev(u, tck), axis=1)
                for tck, u in splines
            ]
            pts.append(np.concatenate(equidistant_points))
        return np.concatenate(pts)

    @property
    def mask(self):
        return self._mask

    def sample_orientations(self):
        """Sample an approximately equidistant grid of orientations on the surface."""
        rots = []
        for splines, meta_splines in (
            (self._row_splines_even, self._column_splines_even),
            (self._row_splines_odd, self._column_splines_odd),
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
            # y vectors are generated on column-by-column basis, so we need to swap axes
            y = einops.rearrange(equidistant_y_vecs, 'column row xyz-> row column xyz')
            y = np.concatenate(equidistant_y_vecs)

            x /= np.linalg.norm(x, axis=1, keepdims=True)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

            z = np.cross(x, y)
            z /= np.linalg.norm(z, axis=1, keepdims=True)

            rots.append(Rotation.from_matrix(np.stack([x, y, z], axis=-1)))

        rots = Rotation.concatenate(rots)

        if self.inside_point is not None:
            # use the inside point to put normal in correct direction
            pts = self.sample()
            # get closest neighbour
            diffs = pts - self.inside_point
            closest_idx = np.argmin(np.linalg.norm(diffs, axis=1))
            z_closest = z[closest_idx]
            z_inside = diffs[closest_idx]
            # flip is the point is outside
            if np.dot(z_inside, z_closest) < 0:
                rots *= Rotation.from_euler("x", np.pi)

        return rots
