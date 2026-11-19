import numpy as np

from akira import (
    CartesianGrid,
    Gaussian,
    SIS,
    fit_sis_einstein_radius,
    gaussian_psf,
    render_sis_gaussian,
    simulate_imaging,
)
from akira.imaging import convolve_psf
from akira.likelihood import gaussian_log_likelihood


def make_problem(noise_sigma: float = 0.02):
    grid = CartesianGrid(shape=(72, 72), pixel_scale=0.05, supersampling=2)
    lens = SIS(einstein_radius=0.9)
    source = Gaussian(
        intensity=1.0,
        sigma=0.11,
        centre=(0.07, -0.04),
        q=0.72,
        phi=0.35,
    )
    psf = gaussian_psf(sigma_pixels=0.9, size=7)
    unconvolved = render_sis_gaussian(grid, lens, source)
    observation = simulate_imaging(
        unconvolved,
        psf,
        noise_sigma=noise_sigma,
        pixel_scale=grid.pixel_scale,
        seed=123,
    )
    return grid, lens, source, observation


def test_simulation_is_reproducible_for_fixed_seed() -> None:
    grid = CartesianGrid(shape=(10, 12), pixel_scale=0.1)
    model = render_sis_gaussian(grid, SIS(0.3), Gaussian(1.0, 0.2))
    psf = gaussian_psf(0.8, size=5)
    first = simulate_imaging(model, psf, 0.05, grid.pixel_scale, seed=77)
    second = simulate_imaging(model, psf, 0.05, grid.pixel_scale, seed=77)
    third = simulate_imaging(model, psf, 0.05, grid.pixel_scale, seed=78)

    np.testing.assert_array_equal(first.data, second.data)
    assert not np.array_equal(first.data, third.data)


def test_true_lens_has_better_likelihood_than_offset_lenses() -> None:
    grid, lens, source, observation = make_problem()

    def likelihood(radius: float) -> float:
        image = render_sis_gaussian(grid, SIS(radius), source)
        image = convolve_psf(image, observation.psf.kernel)
        return float(
            gaussian_log_likelihood(
                observation.data, image, observation.noise_map, observation.mask
            )
        )

    true_value = likelihood(lens.einstein_radius)
    assert true_value > likelihood(lens.einstein_radius - 0.08)
    assert true_value > likelihood(lens.einstein_radius + 0.08)


def test_fit_recovers_simulated_einstein_radius() -> None:
    grid, lens, source, observation = make_problem()
    result = fit_sis_einstein_radius(
        observation,
        grid,
        source,
        initial_einstein_radius=0.72,
        bounds=(0.5, 1.3),
    )

    assert result.success, result.message
    assert abs(result.einstein_radius - lens.einstein_radius) < 0.002
    assert result.model_image.shape == observation.data.shape

