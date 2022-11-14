"""Tooling to fit and sample splines."""

from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, conint, root_validator, validator
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation, Slerp

from .utils import calculate_y_vectors_from_z_vectors, within_range


class NDimensionalSpline(EventedModel):
    """Model for multidimensional splines."""

    points: np.ndarray
    order: conint(ge=1, le=5) = 3
    smoothing: Optional[int] = None
    mask_limits: Tuple[int, int] = (0, -1)
    _n_spline_samples: int = PrivateAttr(10000)
    _tck = PrivateAttr(Tuple)
    _u_mask_limits: Tuple[float, float] = PrivateAttr(())
    _length = PrivateAttr(float)
    _masked_length: float = PrivateAttr()

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
        n_points = points.shape[0]
        spline_order: Optional[int] = values.get("order")

        if spline_order is not None and n_points <= spline_order:
            raise ValueError("number of points must be greater than spline order")

        return values

    def __setattr__(self, name: str, value: Any) -> None:
        """Overwritten so that splines are recalculated when points are updated."""
        super().__setattr__(name, value)
        if name in ("points", "order", "smoothing", "mask_limits"):  # ensure splines stay in sync
            self._prepare_spline()

    def _prepare_spline(self) -> None:
        """Spline parametrisation mapping [0, 1] to a smooth curve through spline points.

        Note: equidistant sampling of this spline parametrisation will not yield
        equidistant samples in Euclidean space.
        """
        # oversample an initial spline to ensure better distance parametrisation
        u = np.linspace(0, 1, self._n_spline_samples)
        tck, raw_u = splprep(self.points.T, s=0, k=self.order)
        samples = np.stack(splev(u, tck), axis=1)
        raw_u_mask_limits = raw_u[list(self.mask_limits)]

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

        # save masked length for later use
        mask = within_range(u, *raw_u_mask_limits)
        first = np.argmax(mask)
        last = -np.argmax(mask[::-1]) or -1
        self._masked_length = cumulative_distance[last] - cumulative_distance[first]

        cumulative_distance /= self._length

        self._equidistance_spline_tck, u = splprep(
            [u], u=cumulative_distance, s=0, k=self.order
        )
        self._u_mask_limits = u[[first, last]]

    def sample(
            self,
            u: Optional[Union[float, np.ndarray]] = None,
            separation: Optional[float] = None,
            derivative_order: int = 0,
            masked: bool = True,
    ) -> np.ndarray:
        """Sample points or derivatives on the spline.

        This function yields samples equidistant in Euclidean space
        along the spline for linearly spaced values of u.

        Parameters
        ----------
        u : Optional[Union[float, np.ndarray]]
            The positions to sample the spline at. These are in the normalized
            spline coordinate, which spans [0, 1]
        separation: Optional[float]
            The desired separation between sampling points in Euclidean space.
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
        if u is not None and separation is not None:
            raise ValueError("only one of u and separation should be provided.")
        if separation is not None:
            u = self._get_equidistant_spline_coordinate_values(separation)
        samples = splev(np.atleast_1d(u), self._tck)
        return np.stack(samples, axis=1)  # (n, d)

        if masked:
            samples = samples[self._u_mask_limits]
        return samples

    def _get_equidistant_spline_coordinate_values(self, separation: float, approximate: bool, masked: bool) -> np.ndarray:
        """Calculate spline coordinates for points with a given Euclidean separation.

        Parameters
        ----------
        separation : float
            The Euclidean distance between desired spline samples.
        approximate : bool
            Approximate the separation in order to include the extrema (also if masked!).
        masked : bool
            Only give values of u that fall within the mask limits.

        Returns
        -------
        u : np.ndarray
            The array of spline coordinate values.
        """
        if masked:
            length = self._masked_length
            min_u, max_u = self._equidistance_u_mask_limits
        else:
            length = self._length
            min_u, max_u = 0, 1

        n_points = int(length / separation)
        if n_points == 0:
            raise ValueError(f'separation ({separation}) must be less than '
                             f'length ({length})')

        if approximate:
            remainder = 0
        else:
            remainder = (length % separation) / length

        return np.linspace(min_u, max_u - remainder, n_points)

    def _get_equidistance_u_by_n_points(self, n_points: int, masked: bool = True):
        if masked:
            min_u, max_u = self._equidistance_u_mask_limits
        else:
            min_u, max_u = 0, 1

        return np.linspace(min_u, max_u, n_points)


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
        z = self._sample_spline_z(u, masked=False)
        y = calculate_y_vectors_from_z_vectors(z)
        x = np.cross(y, z)
        r = Rotation.from_matrix(np.stack([x, y, z], axis=-1))
        self._rotation_sampler = Slerp(u, r)

    def sample_orientations(
        self,
        u: Optional[Union[float, np.ndarray]] = None,
        separation: Optional[float] = None,
        masked: bool = True,
    ) -> Rotation:
        """Local coordinate system at any point along the spline."""
        if u is not None and separation is not None:
            raise ValueError("only one of u and separation should be provided.")
        if separation is not None:
            u = self._get_equidistant_spline_coordinate_values(separation)

        rot = self._rotation_sampler(u)
        if rot.single:
            rot = Rotation.concatenate([rot])

        if masked:
            rot = rot[self._u_mask_limits]
        return rot

    def _sample_spline_z(self, u: Union[float, np.ndarray], masked: bool = True) -> np.ndarray:
        """Sample vectors tangent to the spline."""
        z = self.sample(u, derivative_order=1, masked=masked)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z

    def _sample_spline_y(self, u: Union[float, np.ndarray], masked: bool = True) -> np.ndarray:
        """Sample vectors perpendicular to the spline."""
        rotations = self.sample_orientations(u, masked=masked)
        return rotations.as_matrix()[..., 1]
