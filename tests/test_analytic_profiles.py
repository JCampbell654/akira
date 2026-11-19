import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import hyp2f1

from akira import (
    CartesianGrid,
    CompositeLight,
    EllipticalPowerLaw,
    ExternalShear,
    Gaussian,
    LensPlane,
    SIS,
    Sersic,
    render,
    render_sis_gaussian,
)
from akira.profiles import (
    epl_deflection,
    external_shear_deflection,
    sersic_bn,
    sersic_brightness,
)


def scipy_epl_reference(coordinates, einstein_radius, slope, q):
    """Independent direct evaluation of Tessore & Metcalf equation 13."""

    t = slope - 1.0
    b = einstein_radius * np.sqrt(q)
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    radius = np.sqrt((q * x) ** 2 + y**2)
    angle = np.arctan2(y, q * x)
    flattening = (1.0 - q) / (1.0 + q)
    argument = -flattening * np.exp(2j * angle)
    angular = np.exp(1j * angle) * hyp2f1(1.0, t / 2.0, 2.0 - t / 2.0, argument)
    alpha = 2.0 * b / (1.0 + q) * (b / radius) ** (t - 1.0) * angular
    return np.stack((alpha.real, alpha.imag), axis=-1)


def test_circular_isothermal_epl_exactly_matches_sis() -> None:
    coordinates = jnp.array([[0.2, 0.7], [-0.4, 0.3], [1.2, -0.8]])
    expected = SIS(1.3).deflection(coordinates)
    actual = EllipticalPowerLaw(1.3, slope=2.0).deflection(coordinates)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("q,slope", [(0.8, 1.7), (0.55, 2.0), (0.3, 2.35)])
def test_epl_matches_independent_scipy_hypergeometric_reference(q, slope) -> None:
    coordinates = np.array([[0.3, 0.7], [-0.8, 0.4], [1.1, -0.25]])
    expected = scipy_epl_reference(coordinates, 1.2, slope, q)
    actual = epl_deflection(coordinates, 1.2, slope, q=q)
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-13)


def test_epl_rotation_is_covariant() -> None:
    angle = 0.43
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    coordinates = np.array([[0.4, 0.8], [-0.7, 0.2], [1.0, -0.5]])
    base = np.asarray(epl_deflection(coordinates, 0.9, 2.1, q=0.65, phi=0.0))
    rotated_coordinates = coordinates @ rotation.T
    rotated = epl_deflection(rotated_coordinates, 0.9, 2.1, q=0.65, phi=angle)
    np.testing.assert_allclose(rotated, base @ rotation.T, rtol=5e-13, atol=5e-13)


def test_epl_gradient_matches_finite_difference() -> None:
    coordinate = jnp.array([0.37, -0.62])

    def x_deflection(slope):
        return epl_deflection(coordinate, 1.1, slope, q=0.72)[0]

    point = 2.08
    autodiff = float(jax.grad(x_deflection)(point))
    step = 1e-5
    finite = float((x_deflection(point + step) - x_deflection(point - step)) / (2 * step))
    np.testing.assert_allclose(autodiff, finite, rtol=2e-7, atol=2e-7)


def test_external_shear_zero_angle_and_rotation_convention() -> None:
    coordinates = jnp.array([[2.0, 3.0], [-1.0, 4.0]])
    actual = external_shear_deflection(coordinates, strength=0.1, phi=0.0)
    expected = np.array([[0.2, -0.3], [-0.1, -0.4]])
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_sersic_bn_encloses_half_light() -> None:
    for index in (0.5, 1.0, 2.5, 4.0):
        bn = sersic_bn(index)
        np.testing.assert_allclose(jax.scipy.special.gammainc(2 * index, bn), 0.5, atol=2e-14)


def test_sersic_intensity_is_defined_at_effective_radius() -> None:
    profile = Sersic(intensity=2.7, effective_radius=0.8, sersic_index=3.0)
    np.testing.assert_allclose(profile.brightness(jnp.array([0.8, 0.0])), 2.7)


def test_sersic_n_half_matches_equivalent_gaussian() -> None:
    effective_radius = 0.9
    intensity_at_re = 1.4
    sigma = effective_radius / np.sqrt(2.0 * np.log(2.0))
    coordinates = jnp.array([[0.0, 0.0], [0.4, 0.7], [-0.8, 0.3]])
    sersic = sersic_brightness(
        coordinates, intensity_at_re, effective_radius, 0.5, q=0.7, phi=0.2
    )
    gaussian = Gaussian(
        intensity=2.0 * intensity_at_re,
        sigma=sigma,
        q=0.7,
        phi=0.2,
    ).brightness(coordinates)
    np.testing.assert_allclose(sersic, gaussian, rtol=2e-13, atol=2e-13)


def test_composable_renderer_matches_specialized_reference_and_jits() -> None:
    grid = CartesianGrid(shape=(30, 32), pixel_scale=0.07, supersampling=2)
    sis = SIS(0.65)
    gaussian = Gaussian(1.2, 0.15, centre=(0.04, -0.02), q=0.8, phi=0.3)
    expected = render_sis_gaussian(grid, sis, gaussian)
    actual = render(grid, LensPlane((sis,)), CompositeLight((gaussian,)))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_composition_adds_mass_deflections_and_light_brightness() -> None:
    coordinates = jnp.array([[0.4, 0.7], [-0.6, 0.2]])
    mass_a = SIS(0.5)
    mass_b = ExternalShear(0.04, phi=0.2)
    lens = LensPlane((mass_a, mass_b))
    np.testing.assert_allclose(
        lens.deflection(coordinates),
        mass_a.deflection(coordinates) + mass_b.deflection(coordinates),
    )

    light_a = Gaussian(1.0, 0.3)
    light_b = Sersic(0.2, 0.7, 1.0)
    source = CompositeLight((light_a, light_b))
    np.testing.assert_allclose(
        source.brightness(coordinates),
        light_a.brightness(coordinates) + light_b.brightness(coordinates),
    )

