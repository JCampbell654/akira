import jax
import jax.numpy as jnp
import numpy as np
import pytest

from akira.conventions import ISOTHERMAL_3D_DENSITY_SLOPE
from akira.profiles import Gaussian, SIS, gaussian_brightness, sis_deflection


def numpy_sis_deflection(coordinates: np.ndarray, einstein_radius: float) -> np.ndarray:
    radius = np.linalg.norm(coordinates, axis=-1, keepdims=True)
    return einstein_radius * coordinates / radius


def numpy_gaussian_brightness(
    coordinates: np.ndarray,
    intensity: float,
    sigma: float,
    q: float,
    phi: float,
) -> np.ndarray:
    cosine = np.cos(phi)
    sine = np.sin(phi)
    x_major = coordinates[..., 0] * cosine + coordinates[..., 1] * sine
    y_minor = -coordinates[..., 0] * sine + coordinates[..., 1] * cosine
    radius_squared = q * x_major**2 + y_minor**2 / q
    return intensity * np.exp(-0.5 * radius_squared / sigma**2)


def test_isothermal_convention_is_three_dimensional_slope_two() -> None:
    assert ISOTHERMAL_3D_DENSITY_SLOPE == 2.0


def test_sis_deflection_matches_independent_numpy_equation() -> None:
    coordinates = np.array([[0.3, 0.4], [-0.8, 0.6], [1.1, -0.2]])
    expected = numpy_sis_deflection(coordinates, einstein_radius=1.3)
    actual = sis_deflection(jnp.asarray(coordinates), einstein_radius=1.3)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), 1.3, rtol=1e-14)


def test_sis_origin_has_documented_finite_convention() -> None:
    actual = sis_deflection(jnp.array([[0.0, 0.0]]), 1.0)
    np.testing.assert_array_equal(actual, [[0.0, 0.0]])


def test_sis_kernel_jits_and_differentiates() -> None:
    coordinate = jnp.array([0.3, 0.4])
    compiled = jax.jit(sis_deflection)
    np.testing.assert_allclose(compiled(coordinate, 1.2), [0.72, 0.96])

    derivative = jax.jacfwd(lambda radius: sis_deflection(coordinate, radius))(1.2)
    np.testing.assert_allclose(derivative, [0.6, 0.8], rtol=1e-14, atol=1e-14)


def test_elliptical_gaussian_matches_independent_numpy_equation() -> None:
    coordinates = np.array([[0.2, -0.1], [0.8, 0.3], [-0.4, 0.7]])
    expected = numpy_gaussian_brightness(coordinates, 2.3, 0.7, 0.6, 0.37)
    actual = gaussian_brightness(coordinates, 2.3, 0.7, q=0.6, phi=0.37)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_phi_rotates_gaussian_major_axis_anticlockwise() -> None:
    profile = Gaussian(intensity=1.0, sigma=1.0, q=0.5, phi=np.pi / 2)
    along_positive_y = profile.brightness(jnp.array([0.0, 1.0]))
    along_positive_x = profile.brightness(jnp.array([1.0, 0.0]))
    assert along_positive_y > along_positive_x


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: SIS(einstein_radius=0.0),
        lambda: Gaussian(intensity=1.0, sigma=0.0),
        lambda: Gaussian(intensity=1.0, sigma=1.0, q=1.1),
    ],
)
def test_invalid_profiles_are_rejected(constructor) -> None:
    with pytest.raises(ValueError):
        constructor()

