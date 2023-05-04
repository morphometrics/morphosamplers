"""Tests for NDimensionalSpline."""

import numpy as np
import pytest
from pydantic import ValidationError

from morphosamplers.spline import NDimensionalSpline, Spline3D

n_points = 10
zeros_column = np.zeros((n_points,))
coordinates = np.linspace(0, 1, n_points)
points_2d = np.column_stack([coordinates, coordinates])
expected_points_2d = np.array([[0.5, 0.5], [1, 1]])
points_3d = np.column_stack([zeros_column, coordinates, coordinates])
expected_points_3d = np.array([[0, 0.5, 0.5], [0, 1, 1]])
points_4d = np.column_stack([zeros_column, zeros_column, coordinates, coordinates])
expected_points_4d = np.array([[0, 0, 0.5, 0.5], [0, 0, 1, 1]])


@pytest.mark.parametrize(
    "points,expected_points",
    [
        (points_2d, expected_points_2d),
        (points_3d, expected_points_3d),
        (points_4d, expected_points_4d),
    ],
)
def test_n_dimensional_spline(points, expected_points):
    """Test NDimensionalSpline for 2D, 3D, and 4D splines."""
    spline_model = NDimensionalSpline(points=points, order=4)

    # test that spline order was set
    assert spline_model.order == 4

    # test samping single point
    sample_values = spline_model.sample(0.5)
    expected_single_value = np.atleast_2d(expected_points[0])
    np.testing.assert_allclose(expected_single_value, sample_values)

    # test sampling array of points
    sample_values = spline_model.sample(u=[0.5, 1])
    np.testing.assert_allclose(expected_points, sample_values)


def test_update_spline_points():
    """Test that updating the points recalculates the spline."""
    # line with slope 1
    initial_points = np.array([[0, 0], [0.2, 0.2], [0.3, 0.3], [1, 1]])
    spline_model = NDimensionalSpline(points=initial_points, order=2)

    value_initial_spline = spline_model.sample(u=0.5)
    np.testing.assert_allclose([[0.5, 0.5]], value_initial_spline)

    # line with slope 2
    updated_points = np.array([[0, 0], [0.2, 0.4], [0.3, 0.6], [1, 2]])
    spline_model.points = updated_points
    value_updated_spline = spline_model.sample(u=0.5)
    np.testing.assert_allclose([[0.5, 1]], value_updated_spline)


def test_update_spline_order():
    """Test that updating the order recalculates the spline.

    This does not test for correctness.
    """
    x_coordinate = np.linspace(0, 1, 10)
    y_coordinate = np.power(x_coordinate, 4)
    points = np.column_stack((x_coordinate, y_coordinate))
    initial_spline_order = 2
    spline_model = NDimensionalSpline(points=points, order=initial_spline_order)

    # get the value with the initial spline
    value_initial_spline = spline_model.sample(u=0.2)

    # update the spline order and get value at the same point
    updated_spline_order = 4
    spline_model.order = updated_spline_order
    value_updated_spline = spline_model.sample(u=0.2)

    with pytest.raises(AssertionError):
        # the updated value should be different because the spline is higher order
        np.testing.assert_allclose(value_initial_spline, value_updated_spline)


def test_spline_model_points_list():
    """Spline model should accept a list."""
    points = [
        [0, 1],
        [1, 1],
        [1, 2],
    ]
    _ = NDimensionalSpline(points=points, order=1)


@pytest.mark.parametrize("spline_order", [0.5, -1, 0, 6])
def test_invalid_spline_order(spline_order):
    """Spline order must be an integer in the range [1, 5]."""
    points = np.array([[0, 1], [1, 1], [1, 3], [1, 5], [1, 7], [1, 10], [1, 20]])
    with pytest.raises(ValidationError):
        _ = NDimensionalSpline(points=points, order=spline_order)


equal_points = np.array([[0, 0], [1, 1]])


def test_spline_too_few_points():
    """Number of points should be greater than the spline order."""
    points = np.array([[0, 0], [1, 1]])

    # one point can never work
    with pytest.raises(ValidationError):
        spline = NDimensionalSpline(points=points[:1], order=2)

    # with other numbers of points, the order should be lowered to match
    with pytest.warns(UserWarning):
        spline = NDimensionalSpline(points=points, order=2)
    assert spline.order == len(points) - 1


@pytest.mark.parametrize("derivative_order", [-1, 4])
def test_invalid_spline_derivatives(derivative_order):
    """Derivative order cannot be negative or > spline order."""
    points = np.array([[0, 1], [1, 1], [1, 3], [1, 5], [1, 7], [1, 10], [1, 20]])
    spline_model = NDimensionalSpline(points=points, order=3)

    with pytest.raises(ValueError):
        _ = spline_model.sample(u=0, derivative_order=derivative_order)

    with pytest.raises(ValueError):
        _ = spline_model.sample(
            separation=1, derivative_order=derivative_order
        )


def test_spline_orientations():
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])
    spline = Spline3D(points=points, order=3)
    pt = spline.sample(u=0)
    # lower atol, as it's very strict by default, much more than np.allclose
    np.testing.assert_allclose(pt, 0, atol=1e-8)
    ori = spline.sample_orientations(u=0)
    np.testing.assert_allclose(ori.apply([0, 0, 1]), [[0, 0, 1]])


def test_spline_reverse():
    """After reversing the last input point is the start when sampling."""
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])
    spline = NDimensionalSpline(points=points)
    spline.reverse()
    sample = spline.sample(u=0)
    assert np.allclose(points[-1], sample)
