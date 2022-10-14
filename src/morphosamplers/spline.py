"""Tooling to fit and sample splines."""

from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
from psygnal import EventedModel
from pydantic import PrivateAttr, root_validator, validator
from scipy.interpolate import splev, splprep


class NDimensionalSpline(EventedModel):
    """Model for multidimensional splines."""

    points: np.ndarray
    spline_order: int = 3
    _n_spline_samples = 10000
    _raw_spline_tck = PrivateAttr(Tuple)
    _equidistant_spline_tck = PrivateAttr(Tuple)
    _length = PrivateAttr(float)

    class Config:
        """Pydantic BaseModel configuration."""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Union[np.ndarray, int]):
        """Calculate the splines after validating the paramters."""
        super().__init__(**kwargs)
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

    @validator("spline_order", pre=True)
    def validate_spline_order(cls, v: int) -> int:
        """Validate and coerce spline_order to int."""
        if not isinstance(v, int):
            raise TypeError("spline_order must be an integer.")
        if (v < 1) or (v > 5):
            raise ValueError("spline_order must be >= 1 and <= 5")
        return v

    @root_validator(skip_on_failure=True)
    def validate_number_of_points(
        cls, values: Dict[str, Union[np.ndarray, int]]
    ) -> Dict[str, Union[np.ndarray, int]]:
        """Verify that the number of points > spline_order."""
        points: np.ndarray = values.get("points")
        n_points = points.shape[0]
        spline_order: Optional[int] = values.get("spline_order")

        if spline_order is not None and n_points <= spline_order:
            raise ValueError("number of points must be greater than spline order")

        return values

    def __setattr__(self, name: str, value: Any) -> None:
        """Overwrite settattr so the splines are recalculated."""
        super().__setattr__(name, value)
        if name in ("points", "spline_order"):  # ensure splines stay in sync
            self._prepare_splines()

    def _prepare_splines(self) -> None:
        self._calculate_raw_spline_parameters()
        self._calculate_equidistant_spline_parameters()

    def _calculate_raw_spline_parameters(self) -> None:
        """Spline parametrisation mapping [0, 1] to a smooth curve through spline points.

        Note: equidistant sampling of this spline parametrisation will not yield
        equidistant samples in Euclidean space.
        """
        self._raw_spline_tck, _ = splprep(self.points.T, s=0, k=self.spline_order)

    def _calculate_equidistant_spline_parameters(self) -> None:
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
        self._equidistant_spline_tck, _ = splprep(
            [u], u=cumulative_distance, s=0, k=self.spline_order
        )

    def _sample_spline(
        self, u: Union[float, np.ndarray], derivative: int = 0
    ) -> np.ndarray:
        """Sample points or derivatives along the equidistantly sampled spline.

        This function
        * maps values in the range [0, 1] to points on the smooth spline.
        * yields equidistant samples along the filament for linearly spaced values of u.
        If calculate_derivate=True then the derivative will be evaluated
         and returned instead of spline points.
        """
        u = np.atleast_1d(u)
        u = splev([np.asarray(u)], self._equidistant_spline_tck)  # [
        samples = splev(u, self._raw_spline_tck, der=derivative)
        return einops.rearrange(samples, "c 1 1 b -> b c")

    def _get_equidistant_u(self, separation: float) -> np.ndarray:
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

    def _get_equidistant_spline_samples(
        self, separation: float, calculate_derivative: bool = False
    ) -> np.ndarray:
        """Calculate equidistant spline samples with a defined separation.

        Parameters
        ----------
        sepration : float
            The distance between points in euclidian space.
        calculate_derivative : bool
            Flag set to True to return tangent vectors.
            Default values is False

        Returns
        -------
        values : np.ndarray
            The values along the spline.
            If calculate_derivative is False, returns positions.
            If calculate_derivative is True, returns tangent vectors.
        """
        u = self._get_equidistant_u(separation)
        return self._sample_spline(u, derivative=calculate_derivative)