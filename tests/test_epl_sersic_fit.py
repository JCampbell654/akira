import numpy as np

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


def test_four_parameter_fit_recovers_epl_and_source_parameters() -> None:
    grid = CartesianGrid(shape=(72, 72), pixel_scale=0.05, supersampling=2)
    true_lens = EllipticalPowerLaw(
        einstein_radius=0.92,
        slope=2.08,
        q=0.76,
        phi=0.28,
    )
    shear = ExternalShear(strength=0.035, phi=0.61)
    true_source = Sersic(
        intensity=0.12,
        effective_radius=0.16,
        sersic_index=1.2,
        centre=(0.07, -0.045),
        q=0.68,
        phi=-0.22,
    )
    noiseless = render(
        grid,
        LensPlane((true_lens, shear)),
        CompositeLight((true_source,)),
    )
    observation = simulate_imaging(
        noiseless,
        gaussian_psf(0.85, size=7),
        noise_sigma=0.004,
        pixel_scale=grid.pixel_scale,
        seed=314,
    )

    result = fit_epl_sersic_parameters(
        observation=observation,
        grid=grid,
        lens_template=true_lens,
        source_template=true_source,
        shear=shear,
        initial=EPLSersicParameters(0.84, 1.95, 0.03, -0.01),
        bounds=((0.7, 1.1), (1.7, 2.4), (-0.2, 0.2), (-0.2, 0.2)),
    )

    assert result.success, result.message
    recovered = result.parameters
    np.testing.assert_allclose(recovered.einstein_radius, true_lens.einstein_radius, atol=0.002)
    np.testing.assert_allclose(recovered.slope, true_lens.slope, atol=0.006)
    np.testing.assert_allclose(recovered.source_x, true_source.centre[0], atol=0.001)
    np.testing.assert_allclose(recovered.source_y, true_source.centre[1], atol=0.001)
    assert result.model_image.shape == observation.data.shape

