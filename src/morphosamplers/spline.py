"""Tooling to fit and sample splines."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, conint, validator, root_validator
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation, Slerp

from .utils import calculate_y_vectors_from_z_vectors


class NDimensionalSpline(EventedModel):
    """Model for multidimensional splines."""
    points: np.ndarray
    order: conint(ge=1, le=5) = 3
    _n_spline_samples = 10000
    _raw_spline_tck = PrivateAttr(Tuple)
    _equidistant_spline_tck = PrivateAttr(Tuple)
    _length = PrivateAttr(float)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prepare_smooth_spline()

    @property
    def _ndim(self) -> int:
        return self.points.shape[-1]

    @validator('points', pre=True)
    def is_coordinate_array(cls, v, values):
        points = np.atleast_2d(np.array(v))
        if points.ndim != 2:
            raise ValueError('data must be an (n, 2) array')
        return points

    @root_validator(skip_on_failure=True)
    def validate_number_of_points(
        cls, values: Dict[str, Union[np.ndarray, int]]
    ) -> Dict[str, Union[np.ndarray, int]]:
        """Verify that the number of points > spline_order."""
        points: np.ndarray = values.get("points")
        n_points = points.shape[0]
        spline_order: Optional[int] = values.get("order")

        if spline_order is not None and n_points <= spline_order:
            raise ValueError("number of points must be greater than spline order")
        return values

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ('points', 'order'):  # ensure splines stay in sync
            self._prepare_smooth_spline()

    def _prepare_smooth_spline(self):
        self._fit_raw_spline()
        self._fit_equidistant_spline()

    def _fit_raw_spline(self):
        """Spline parametrisation mapping [0, 1] to a smooth curve through spline points.

        Note: equidistant sampling of this spline parametrisation will not yield equidistant
        samples in Euclidean space.
        """
        self._raw_spline_tck, _ = splprep(self.points.T, s=0, k=self.order)

    def _fit_equidistant_spline(self):
        """Calculate a mapping of normalised cumulative distance to linear samples range [0, 1].

        * Normalised cumulative distance is the cumulative euclidean distance along the filament
          rescaled to a range of [0, 1].
        * The spline parametrisation calculated here can be used to map linearly spaced values
        which when used in the filament spline parametrisation, yield equidistant points in
        Euclidean space.
        """
        # sample the current filament spline parametrisation, yielding non-equidistant samples
        u = np.linspace(0, 1, self._n_spline_samples)
        filament_samples = splev(u, self._raw_spline_tck)
        filament_samples = np.stack(filament_samples, axis=1)

        # calculate the cumulative length of line segments as we move along the filament.
        inter_point_differences = np.diff(filament_samples, axis=0)
        inter_point_distances = np.linalg.norm(inter_point_differences, axis=1)
        cumulative_distance = np.cumsum(inter_point_distances)

        # calculate spline mapping normalised cumulative distance to linear samples in [0, 1]
        self._length = cumulative_distance[-1]
        cumulative_distance /= self._length
        # prepend a zero, no distance has been covered at start of spline parametrisation
        cumulative_distance = np.r_[[0], cumulative_distance]
        self._equidistant_spline_tck, _ = splprep([u], u=cumulative_distance, s=0, k=self.order)

    def sample_spline(self, u: Union[float, np.ndarray], derivative_order: int = 0):
        """Sample points or derivatives along the spline of the filament.

        This function
        * maps values in the range [0, 1] to points on the smooth filament spline.
        * yields equidistant samples along the filament for linearly spaced values of u.
        """
        u = splev([np.asarray(u)], self._equidistant_spline_tck)
        samples = splev(u, self._raw_spline_tck, der=derivative_order)
        return np.atleast_2d(np.squeeze(samples).T)

    def _get_equidistant_u(self, separation: float) -> np.ndarray:
        """Get values for u which yield spline samples with a defined Euclidean separation."""
        n_points = int(self._length // separation)
        remainder = (self._length % separation) / self._length
        return np.linspace(0, 1 - remainder, n_points)

    def _get_equidistant_spline_samples(
            self, separation: float, derivative_order: int = 0
    ) -> np.ndarray:
        """Calculate equidistant spline samples with a defined separation in Euclidean space."""
        u = self._get_equidistant_u(separation)
        return self.sample_spline(u, derivative_order=derivative_order)


class Spline3D(NDimensionalSpline):
    """3D spline model with interpolation of orientations along the spline.

    Basis vectors for a local coordinate system along the spline can be calculated.
    In this coordinate system, z vectors are tangent to the spline and the xy-plane
    changes minimally and smoothly.
    """
    _rotation_sampler = PrivateAttr(Slerp)

    @validator('points')
    def _is_n_plus_3d_coordinate_array(cls, v):
        if v.ndim != 2 or v.shape[-1] != 3:
            raise ValueError('must be an (n, 3) array')
        return v

    def _prepare_smooth_spline(self):
        super()._prepare_smooth_spline()
        self._prepare_orientation_parametrisation()

    def _prepare_orientation_parametrisation(self):
        u = np.linspace(0, 1, num=self._n_spline_samples)
        z = self._sample_spline_z(u)
        y = calculate_y_vectors_from_z_vectors(z)
        x = np.cross(y, z)
        r = Rotation.from_matrix(np.stack([x, y, z], axis=-1))
        self._rotation_sampler = Slerp(u, r)

    def sample_spline_orientations(self, u: np.ndarray):
        """Local coordinate system at any point along the spline."""
        return self._rotation_sampler(u)

    def _sample_spline_z(self, u: np.ndarray) -> np.ndarray:
        """Sample vectors tangent to the spline."""
        z = self.sample_spline(u, derivative_order=1)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z

    def _sample_spline_y(self, u: np.ndarray) -> np.ndarray:
        """Sample vectors perpendicular to the spline."""
        rotations = self._sample_spline_orientations(u)
        return rotations.as_matrix()[..., 1]

    def _get_equidistant_orientations(self, separation: float) -> Rotation:
        """Calculate orientations for equidistant samples with a defined separation."""
        u = self._get_equidistant_u(separation)
        return self._sample_spline_orientations(u)
