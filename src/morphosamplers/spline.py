"""Tooling to fit and sample splines."""

from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, conint, root_validator, validator
from scipy.interpolate import splev, splprep


class NDimensionalSpline(EventedModel):
    """Model for multidimensional splines."""

    points: np.ndarray
    order: conint(ge=1, le=5) = 3
    _n_spline_samples: int = 10000
    _raw_spline_tck = PrivateAttr(Tuple)
    _equidistance_spline_tck = PrivateAttr(Tuple)
    _length = PrivateAttr(float)

    class Config:
        """Pydantic BaseModel configuration."""

        arbitrary_types_allowed = True

    def __init__(self, points: np.ndarray, order: int = 3):
        """Calculate the splines after validating the paramters."""
        super().__init__(points=points, order=order)
        self._prepare_splines()

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
        if name in ("points", "order"):  # ensure splines stay in sync
            self._prepare_splines()

    def _prepare_splines(self) -> None:
        self._fit_raw_spline_parameters()
        self._fit_equidistance_spline_parameters()

    def _fit_raw_spline_parameters(self) -> None:
        """Spline parametrisation mapping [0, 1] to a smooth curve through spline points.

        Note: equidistant sampling of this spline parametrisation will not yield
        equidistant samples in Euclidean space.
        """
        self._raw_spline_tck, _ = splprep(self.points.T, s=0, k=self.order)

    def _fit_equidistance_spline_parameters(self) -> None:
        """Calculate a mapping of normalised cumulative distance to linear range [0, 1].

        * Normalised cumulative distance is the cumulative euclidean distance along
          the spline rescaled to a range of [0, 1].
        * The spline parametrisation calculated here can be used to
        map linearly spaced valueswhich when used in the spline spline parametrisation,
        yield equidistant points in Euclidean space.
        """
        # sample the current raw spline parametrisation
        # yielding non-equidistant samples
        u = np.linspace(0, 1, self._n_spline_samples)
        filament_samples = splev(u, self._raw_spline_tck)
        filament_samples = np.stack(filament_samples, axis=1)

        # calculate the cumulative length of line segments
        # as we move along the filament.
        inter_point_differences = np.diff(filament_samples, axis=0)
        inter_point_distances = np.linalg.norm(inter_point_differences, axis=1)
        cumulative_distance = np.cumsum(inter_point_distances)

        # calculate spline mapping normalised cumulative distanceto
        # linear samples in [0, 1]
        self._length = cumulative_distance[-1]
        cumulative_distance /= self._length

        # prepend a zero, no distance has been covered
        # at start of spline parametrisation
        cumulative_distance = np.r_[[0], cumulative_distance]
        self._equidistance_spline_tck, _ = splprep(
            [u], u=cumulative_distance, s=0, k=self.order
        )

    def sample_spline(
        self, u: Union[float, np.ndarray], derivative_order: int = 0
    ) -> np.ndarray:
        """Sample points or derivatives along the equidistantly sampled spline.

        This function
        * maps values in the range [0, 1] to points on the smooth spline.
        * yields equidistant samples along the filament for linearly spaced values of u.
        If calculate_derivate=True then the derivative will be evaluated
         and returned instead of spline points.

        Parameters
        ----------
        u : Union[float, np.ndarray]
            The positions to sample the spline at. These are in the normalized
            spline coordinate, which spans [0, 1]
        derivative_order : int
            Order of the derivative to evaluate at each spline position.
            If 0, the position on the spline is returned.
            If >0, the derivative of position is returned (e.g., 1 for tangent vector).
            derivative_order must be <= the spline order.
            Default value is 0.

        Returns
        -------
        values : np.ndarray
            The values along the spline.
            If derivative_order is 0, returns positions.
            If calculate_derivative >0, returns derivative vectors.
        """
        if (derivative_order < 0) or (derivative_order > self.order):
            # derivative order must be 0 < derivative_order < spline_order
            raise ValueError("derivative order must be [0, spline_order]")
        u = np.atleast_1d(u)
        u = splev([np.asarray(u)], self._equidistance_spline_tck)
        samples = splev(u, self._raw_spline_tck, der=derivative_order)
        return einops.rearrange(samples, "c 1 1 b -> b c")

    def _get_equidistance_u(self, separation: float) -> np.ndarray:
        """Get equally spaced values of u.

        Parameters
        ----------
        separation : float
            The distance between the u values in euclidian distance.

        Returns
        -------
        u : np.ndarray
            The array of equally-spaced spline coordinates..
        """
        n_points = int(self._length // separation)
        remainder = (self._length % separation) / self._length
        return np.linspace(0, 1 - remainder, n_points)

    def _get_equidistance_spline_samples(
        self, separation: float, derivative_order: bool = 0
    ) -> np.ndarray:
        """Calculate equidistant spline samples with a defined separation.

        Parameters
        ----------
        sepration : float
            The distance between points in euclidian space.
        derivative_order : int
            Order of the derivative to evaluate at each spline position.
            If 0, the position on the spline is returned.
            If >0, the derivative of position is returned (e.g., 1 for tangent vector).
            derivative_order must be <= the spline order.
            Default value is 0.

        Returns
        -------
        values : np.ndarray
            The values along the spline.
            If derivative_order is 0, returns positions.
            If calculate_derivative >0, returns derivative vectors.
        """
        if (derivative_order < 0) or (derivative_order > self.order):
            # derivative order must be 0 < derivative_order < spline_order
            raise ValueError("derivative order must be [0, spline_order]")
        u = self._get_equidistance_u(separation)
        return self.sample_spline(u, derivative_order=derivative_order)
