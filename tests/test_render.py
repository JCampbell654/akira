import jax
import jax.numpy as jnp
import numpy as np

from akira import CartesianGrid, Gaussian, SIS, render_sis_gaussian
from akira.render import render_sis_gaussian_from_coordinates


def numpy_reference_image(
    coordinates: np.ndarray, einstein_radius: float, sigma: float
) -> np.ndarray:
    radius = np.linalg.norm(coordinates, axis=-1, keepdims=True)
    deflection = einstein_radius * coordinates / radius
    source_coordinates = coordinates - deflection
    brightness = np.exp(
        -0.5 * np.sum(source_coordinates**2, axis=-1) / sigma**2
    )
    return brightness.mean(axis=2)


def test_sis_gaussian_image_matches_independent_numpy_pipeline() -> None:
    grid = CartesianGrid(shape=(20, 24), pixel_scale=0.08, supersampling=2)
    lens = SIS(einstein_radius=0.55)
    source = Gaussian(intensity=1.0, sigma=0.12)

    actual = np.asarray(render_sis_gaussian(grid, lens, source))
    expected = numpy_reference_image(
        np.asarray(grid.subpixel_coordinates()), lens.einstein_radius, source.sigma
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_centred_source_forms_ring_at_einstein_radius() -> None:
    grid = CartesianGrid(shape=(200, 200), pixel_scale=0.02, supersampling=2)
    lens = SIS(einstein_radius=1.0)
    source = Gaussian(intensity=1.0, sigma=0.04)
    image = np.asarray(render_sis_gaussian(grid, lens, source))
    radii = np.linalg.norm(np.asarray(grid.coordinates()), axis=-1)
    peak_radius = radii.reshape(-1)[np.argmax(image)]

    assert abs(peak_radius - lens.einstein_radius) < grid.pixel_scale


def test_renderer_uses_float64_and_is_jitted() -> None:
    grid = CartesianGrid(shape=(8, 8), pixel_scale=0.1)
    image = render_sis_gaussian(grid, SIS(0.3), Gaussian(1.0, 0.2))
    assert image.dtype == jnp.float64

    lowered = render_sis_gaussian_from_coordinates.lower(
        grid.subpixel_coordinates(),
        0.3,
        jnp.array([0.0, 0.0]),
        1.0,
        0.2,
        jnp.array([0.0, 0.0]),
        1.0,
        0.0,
    )
    assert lowered is not None


def test_integrated_flux_is_differentiable_with_respect_to_einstein_radius() -> None:
    grid = CartesianGrid(shape=(40, 40), pixel_scale=0.05, supersampling=2)
    coordinates = grid.subpixel_coordinates()

    def total_brightness(einstein_radius):
        return render_sis_gaussian_from_coordinates(
            coordinates,
            einstein_radius,
            jnp.array([0.0, 0.0]),
            1.0,
            0.15,
            jnp.array([0.08, -0.03]),
            0.8,
            0.2,
        ).sum()

    point = 0.55
    autodiff = float(jax.grad(total_brightness)(point))
    step = 1e-5
    finite_difference = float(
        (total_brightness(point + step) - total_brightness(point - step)) / (2 * step)
    )
    np.testing.assert_allclose(autodiff, finite_difference, rtol=2e-6, atol=2e-6)

