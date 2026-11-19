"""Simulate and recover an SIS Einstein radius from noisy imaging."""

from akira import (
    CartesianGrid,
    Gaussian,
    SIS,
    fit_sis_einstein_radius,
    gaussian_psf,
    render_sis_gaussian,
    simulate_imaging,
)


def main() -> None:
    grid = CartesianGrid(shape=(100, 100), pixel_scale=0.04, supersampling=2)
    true_lens = SIS(einstein_radius=1.0)
    source = Gaussian(
        intensity=1.0,
        sigma=0.09,
        centre=(0.08, -0.04),
        q=0.75,
        phi=0.3,
    )
    psf = gaussian_psf(sigma_pixels=1.0, size=9)
    observation = simulate_imaging(
        unconvolved_image=render_sis_gaussian(grid, true_lens, source),
        psf=psf,
        noise_sigma=0.02,
        pixel_scale=grid.pixel_scale,
        seed=42,
    )

    result = fit_sis_einstein_radius(
        observation=observation,
        grid=grid,
        source=source,
        initial_einstein_radius=0.8,
        bounds=(0.5, 1.5),
    )

    print(f"true Einstein radius:      {true_lens.einstein_radius:.6f} arcsec")
    print(f"recovered Einstein radius: {result.einstein_radius:.6f} arcsec")
    print(f"optimizer success:         {result.success}")
    print(f"likelihood evaluations:    {result.evaluations}")
    print(f"log likelihood:            {result.log_likelihood:.6f}")


if __name__ == "__main__":
    main()

