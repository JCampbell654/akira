"""Fit an elliptical power-law lens and Sérsic source."""

from akira import (
    CartesianGrid,
    CompositeLight,
    EPLSersicParameters,
    EllipticalPowerLaw,
    ExternalShear,
    LensPlane,
    Sersic,
    fit_epl_sersic_parameters,
    gaussian_psf,
    render,
    simulate_imaging,
)


def main() -> None:
    grid = CartesianGrid(shape=(90, 90), pixel_scale=0.045, supersampling=2)
    true_lens = EllipticalPowerLaw(0.95, slope=2.1, q=0.74, phi=0.3)
    shear = ExternalShear(0.04, phi=0.65)
    true_source = Sersic(
        intensity=0.12,
        effective_radius=0.17,
        sersic_index=1.3,
        centre=(0.075, -0.05),
        q=0.7,
        phi=-0.2,
    )
    noiseless = render(
        grid,
        LensPlane((true_lens, shear)),
        CompositeLight((true_source,)),
    )
    observation = simulate_imaging(
        noiseless,
        gaussian_psf(0.9, size=7),
        noise_sigma=0.004,
        pixel_scale=grid.pixel_scale,
        seed=2026,
    )
    result = fit_epl_sersic_parameters(
        observation,
        grid,
        lens_template=true_lens,
        source_template=true_source,
        shear=shear,
        initial=EPLSersicParameters(0.85, 1.95, 0.02, -0.01),
        bounds=((0.7, 1.2), (1.7, 2.5), (-0.2, 0.2), (-0.2, 0.2)),
    )

    print("parameter          true        recovered")
    print(f"Einstein radius    {true_lens.einstein_radius:8.5f}    {result.parameters.einstein_radius:8.5f}")
    print(f"3D density slope   {true_lens.slope:8.5f}    {result.parameters.slope:8.5f}")
    print(f"source x           {true_source.centre[0]:8.5f}    {result.parameters.source_x:8.5f}")
    print(f"source y           {true_source.centre[1]:8.5f}    {result.parameters.source_y:8.5f}")
    print(f"optimizer success: {result.success}")
    print(f"evaluations:       {result.evaluations}")


if __name__ == "__main__":
    main()

