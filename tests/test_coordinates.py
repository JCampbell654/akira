import numpy as np
import pytest

from akira import CartesianGrid, degrees_to_radians, radians_to_degrees


def test_odd_grid_is_centred_and_uses_xy_coordinate_order() -> None:
    grid = CartesianGrid(shape=(3, 5), pixel_scale=0.2, centre=(1.0, -2.0))
    coordinates = np.asarray(grid.coordinates())

    assert coordinates.shape == (3, 5, 2)
    np.testing.assert_allclose(coordinates[1, 2], [1.0, -2.0], atol=1e-15)
    np.testing.assert_allclose(coordinates[1, 3] - coordinates[1, 2], [0.2, 0.0])
    np.testing.assert_allclose(coordinates[2, 2] - coordinates[1, 2], [0.0, 0.2])


def test_even_grid_is_symmetric_about_requested_centre() -> None:
    grid = CartesianGrid(shape=(2, 2), pixel_scale=1.0)
    expected = np.array(
        [
            [[-0.5, -0.5], [0.5, -0.5]],
            [[-0.5, 0.5], [0.5, 0.5]],
        ]
    )
    np.testing.assert_allclose(grid.coordinates(), expected)


def test_supersampling_coordinates_average_to_pixel_centres() -> None:
    grid = CartesianGrid(shape=(2, 3), pixel_scale=0.4, supersampling=4)
    assert grid.subpixel_coordinates().shape == (2, 3, 16, 2)
    np.testing.assert_allclose(
        grid.subpixel_coordinates().mean(axis=2), grid.coordinates(), atol=1e-15
    )


def test_angle_conversion_round_trip() -> None:
    angles = np.array([-180.0, -12.5, 0.0, 90.0, 360.0])
    np.testing.assert_allclose(radians_to_degrees(degrees_to_radians(angles)), angles)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": (0, 2), "pixel_scale": 1.0},
        {"shape": (2, 2), "pixel_scale": 0.0},
        {"shape": (2, 2), "pixel_scale": 1.0, "supersampling": 0},
    ],
)
def test_invalid_grids_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CartesianGrid(**kwargs)

