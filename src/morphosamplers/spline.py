"""Tooling to fit and sample splines."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, conint, root_validator, validator
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation, Slerp

from .utils import calculate_y_vectors_from_z_vectors, within_range, get_mask_limits


class NDimensionalSpline(EventedModel):
    """Model for multidimensional splines."""

    points: np.ndarray
    order: conint(ge=1, le=5) = 3
    smoothing: Optional[int] = None
    mask: Optional[np.ndarray] = None
    closed: bool = False
    _n_spline_samples: int = PrivateAttr(10000)
    _tck = PrivateAttr(Tuple)
    _u_mask_limits: List[Tuple[float, float]] = PrivateAttr([])
    _length = PrivateAttr(float)

    class Config:
        """Pydantic BaseModel configuration."""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Calculate the splines after validating the paramters."""
        super().__init__(**kwargs)
        self._prepare_spline()

    @property
    def _ndim(self) -> int:
        ndim: int = self.points.shape[1]
        return ndim

    @validator("points", pre=True)
    def is_coordinate_array(cls, v: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """Validate and coerce the points values to a 2D numpy array."""
        points = np.atleast_2d(v)
        if points.ndim != 2:
            raise ValueError("points must be an (n, d) array")
        return points

    @root_validator(skip_on_failure=True)
    def validate_number_of_points(
        cls, values: Dict[str, Union[np.ndarray, int]]
    ) -> Dict[str, Union[np.ndarray, int]]:
        """Verify that the number of points > spline_order."""
        points: np.ndarray = values.get("points")
        n_samples = points.shape[0]
        spline_order: Optional[int] = values.get("order")

        if spline_order is not None and n_samples <= spline_order:
            raise ValueError("number of points must be greater than spline order")

        return values

    def __setattr__(self, name: str, value: Any) -> None:
        """Overwritten so that splines are recalculated when points are updated."""
        super().__setattr__(name, value)
        if name in ("points", "order", "smoothing", "mask"):  # ensure splines stay in sync
            self._prepare_spline()

    def _prepare_spline(self) -> None:
        """Spline parametrisation mapping [0, 1] to a smooth curve through spline points.

        Equidistant samples between 0 and 1 will yield points equidistant along
        the spline in euclidean space.
        """
        # oversample an initial spline to ensure better distance parametrisation
        u = np.linspace(0, 1, self._n_spline_samples)

        if self.closed:
            points = np.append(self.points, self.points[:1], axis=0)
        else:
            points = self.points

        tck, raw_u = splprep(points.T, s=0, k=self.order, per=self.closed)
        samples = np.stack(splev(u, tck), axis=1)

        # calculate the cumulative length of line segments
        # as we move along the filament.
        inter_point_differences = np.diff(samples, axis=0)
        inter_point_distances = np.linalg.norm(inter_point_differences, axis=1)
        cumulative_distance = np.cumsum(inter_point_distances)
        # prepend a zero, no distance has been covered
        # at start of spline parametrisation
        cumulative_distance = np.insert(cumulative_distance, 0, 0)
        # save length for later and normalize
        self._length = cumulative_distance[-1]

        if self.mask is not None:
            limits = get_mask_limits(self.mask)
            u_ranges = []
            for low_idx, high_idx in limits:
                u_low, u_high = raw_u[[low_idx, high_idx]]
                u_ranges.append((u_low, u_high))
            self._u_mask_limits = u_ranges

        cumulative_distance /= self._length

        self._tck, u = splprep(
            samples.T, u=cumulative_distance, s=0, k=self.order
        )

    def sample(
        self,
        u: Optional[Union[float, np.ndarray]] = None,
        separation: Optional[float] = None,
        n_samples: Optional[int] = None,
        derivative_order: int = 0,
    ) -> np.ndarray:
        """Sample points or derivatives on the spline.

        This function yields samples equidistant in Euclidean space
        along the spline for linearly spaced values of u.
        Only one of u, separation or n_samples should be provided.

        Parameters
        ----------
        u : Optional[Union[float, np.ndarray]]
            The positions to sample the spline at. These are in the normalized
            spline coordinate, which spans [0, 1].
        separation : Optional[float]
            The desired separation between sampling points in Euclidean space.
        n_samples : Optional[int]
            The total number of equidistant points to sample along the spline.
        derivative_order : int
            Order of the derivative to evaluate at each spline position.
            If 0, the position on the spline is returned.
            If greater than 0, the derivative of position is returned
            (e.g., 1 for tangent vector). The derivative_order must be less than
            or equal to the spline order. The default value is 0.

        Returns
        -------
        values : np.ndarray
            The values along the spline.
            If derivative_order is 0, returns positions.
            If calculate_derivative >0, returns derivative vectors.
        """
        if (derivative_order < 0) or (derivative_order > self.order):
            raise ValueError("derivative order must be [0, spline_order]")
        if sum(arg is not None for arg in (u, separation, n_samples)) != 1:
            raise ValueError("only one of u, separation or n_samples should be provided.")
        if u is None:
            u = self._get_equidistant_spline_coordinate_values(separation=separation, n_samples=n_samples)

        samples = splev(np.atleast_1d(u), self._tck, der=derivative_order)
        samples = np.stack(samples, axis=1)  # (n, d)

        return samples

    def _get_equidistant_spline_coordinate_values(
        self,
        separation: Optional[float] = None,
        n_samples: Optional[int] = None,
        approximate: bool = False,
    ) -> np.ndarray:
        """Calculate spline coordinates for points with a given Euclidean separation.

        Only one of separation or n_samples should be provided.

        Parameters
        ----------
        separation : float
            The Euclidean distance between desired spline samples.
        n_samples : int
            The total number of equidistant points to sample along the spline.
        approximate : bool
            Approximate the separation in order to include the extrema.
            Has no effect if n_samples is provided.

        Returns
        -------
        u : np.ndarray
            The array of spline coordinate values.
        """
        if separation is not None and n_samples is not None:
            raise ValueError("only one of separation and n_samples should be provided.")

        if n_samples is not None:
            remainder = 0
        elif separation is not None:
            n_samples = int(self._length / separation)
            if n_samples == 0:
                raise ValueError(f'separation ({separation}) must be less than '
                                 f'length ({self._length})')
            if approximate:
                remainder = 0
            else:
                remainder = (self._length % separation) / self._length

        return np.linspace(0, 1 - remainder, n_samples)

    def get_mask_for_u(self, u):
        """Get a 1D boolean mask for a given u.

        The mask is True where values should be kept.
        """
        # we need to be careful here because there are precision issues with splines
        # and sometimes we get values below or above the limits when they should be equal
        # so we use a special within_range function
        u = np.atleast_1d(u)
        mask = np.zeros_like(u, bool)
        for low, high in self._u_mask_limits:
            mask[within_range(u, low, high)] = 1
        return mask

    def reverse(self):
        """Reverse the order of points and recompute the spline."""
        self.points = self.points[::-1, :]


class Spline3D(NDimensionalSpline):
    """3D spline model with interpolation of orientations along the spline.

    Basis vectors for a local coordinate system along the spline can be calculated.
    In this coordinate system, z vectors are tangent to the spline and the xy-plane
    changes minimally and smoothly.
    """

    _rotation_sampler = PrivateAttr(Slerp)

    @validator("points")
    def _is_3d_coordinate_array(cls, v):
        if v.ndim != 2 or v.shape[-1] != 3:
            raise ValueError("must be an (n, 3) array")
        return v

    def _prepare_spline(self):
        super()._prepare_spline()
        self._prepare_orientation_sampler()

    def _prepare_orientation_sampler(self):
        """Prepare a sampler yielding smoothly varying orientations along the spline.

        This method constructs a set of rotation matrices which vary smoothly with
        the spline coordinate `u`. A sampler is then prepared which can be queried at
        any point(s) along the spline coordinate `u` and the resulting rotations vary
        smoothly along the spline
        """
        u = np.linspace(0, 1, num=self._n_spline_samples)
        z = self._sample_spline_z(u)
        y = calculate_y_vectors_from_z_vectors(z)
        x = np.cross(y, z)
        r = Rotation.from_matrix(np.stack([x, y, z], axis=-1))
        self._rotation_sampler = Slerp(u, r)

    def sample_orientations(
        self,
        u: Optional[Union[float, np.ndarray]] = None,
        separation: Optional[float] = None,
        n_samples: Optional[int] = None,
    ) -> Rotation:
        """Local coordinate system at any point along the spline.


        Only one of u, separation or n_samples should be provided.
        """
        if sum(arg is not None for arg in (u, separation, n_samples)) != 1:
            raise ValueError("only one of u, separation or n_samples should be provided.")
        if u is None:
            u = self._get_equidistant_spline_coordinate_values(separation=separation, n_samples=n_samples)

        rot = self._rotation_sampler(u)
        if rot.single:
            rot = Rotation.concatenate([rot])

        return rot

    def _sample_spline_z(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Sample vectors tangent to the spline."""
        z = self.sample(u, derivative_order=1)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z

    def _sample_spline_y(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Sample vectors perpendicular to the spline."""
        rotations = self.sample_orientations(u)
        return rotations.as_matrix()[..., 1]
