import numpy as np
import pytest

from akira import (
    CartesianGrid,
    EllipticalPowerLaw,
    ExternalShear,
    FreeParameter,
    LensPlane,
    LogUniformPrior,
    PixelizedLensModel,
    ProfileModel,
    SourceGrid,
    UniformPrior,
    fit_pixelized_model,
    gaussian_psf,
    lensing_operator,
    simulate_imaging,
)


def make_problem():
    image_grid = CartesianGrid((30, 30), 0.08)
    source_grid = SourceGrid((10, 10), 0.06, centre=(0.03, -0.02))
    true_lens = LensPlane(
        (
            EllipticalPowerLaw(0.72, 2.08, q=0.78, phi=0.25),
            ExternalShear(0.025, 0.65),
        )
    )
    coordinates = source_grid.coordinates()
    x, y = coordinates[..., 0], coordinates[..., 1]
    source = np.exp(
        -0.5 * ((x - 0.06) ** 2 / 0.07**2 + (y + 0.03) ** 2 / 0.045**2)
    ) + 0.35 * np.exp(
        -0.5 * ((x + 0.08) ** 2 / 0.035**2 + (y - 0.06) ** 2 / 0.05**2)
    )
    unlensed = (
        lensing_operator(image_grid, source_grid, true_lens) @ source.reshape(-1)
    ).reshape(image_grid.shape)
    observation = simulate_imaging(
        unlensed,
        gaussian_psf(0.7, 5),
        noise_sigma=0.008,
        pixel_scale=image_grid.pixel_scale,
        seed=7,
    )
    return image_grid, source_grid, observation


def make_model(source_grid, kind="zeroth", strength=30.0):
    return PixelizedLensModel(
        lens=(
            ProfileModel(
                EllipticalPowerLaw,
                einstein_radius=FreeParameter(0.66, UniformPrior(0.55, 0.9)),
                slope=FreeParameter(1.95, UniformPrior(1.7, 2.4)),
                centre=(0.0, 0.0),
                q=0.78,
                phi=0.25,
            ),
            ProfileModel(ExternalShear, strength=0.025, phi=0.65),
        ),
        source_grid=source_grid,
        regularization_kind=kind,
        regularization_strength=strength,
    )


def test_pixelized_model_discovers_lens_and_regularization_parameters():
    source_grid = SourceGrid((6, 6), 0.1)
    model = make_model(
        source_grid,
        strength=FreeParameter(20.0, LogUniformPrior(1.0, 1000.0)),
    )
    assert model.parameter_names == (
        "lens.0.EllipticalPowerLaw.einstein_radius",
        "lens.0.EllipticalPowerLaw.slope",
        "regularization.strength",
    )
    lens, strength = model.build(model.initial_vector)
    assert lens.mass_profiles[0].einstein_radius == pytest.approx(0.66)
    assert lens.mass_profiles[0].slope == pytest.approx(1.95)
    assert float(strength) == pytest.approx(20.0)
    assert np.isfinite(float(model.log_prior(model.initial_vector)))


def test_evidence_fit_recovers_free_lens_parameters():
    image_grid, source_grid, observation = make_problem()
    result = fit_pixelized_model(
        make_model(source_grid), observation, image_grid, objective="evidence"
    )
    assert result.success
    assert float(result.parameter_vector[0]) == pytest.approx(0.72, abs=0.001)
    assert float(result.parameter_vector[1]) == pytest.approx(2.08, abs=0.012)
    assert result.log_source_evidence == pytest.approx(
        result.reconstruction.log_evidence, rel=1e-11
    )
    assert result.reconstruction.source.shape == source_grid.shape
    assert result.reconstruction.model_image.shape == image_grid.shape
    assert result.covariance.shape == (2, 2)


def test_suyu_gradient_regularization_supports_evidence():
    image_grid, source_grid, observation = make_problem()
    result = fit_pixelized_model(
        make_model(source_grid, kind="gradient"),
        observation,
        image_grid,
        objective="evidence",
    )
    assert result.success
    assert result.log_source_evidence is not None


def test_gradient_regularization_can_use_profile_likelihood():
    image_grid, source_grid, observation = make_problem()
    model = PixelizedLensModel(
        lens=(
            ProfileModel(
                EllipticalPowerLaw,
                einstein_radius=FreeParameter(0.68, UniformPrior(0.6, 0.8)),
                slope=2.08,
                q=0.78,
                phi=0.25,
            ),
            ProfileModel(ExternalShear, strength=0.025, phi=0.65),
        ),
        source_grid=source_grid,
        regularization_kind="gradient",
        regularization_strength=30.0,
    )
    result = fit_pixelized_model(
        model, observation, image_grid, objective="profile_likelihood"
    )
    assert result.success
    assert float(result.parameter_vector[0]) == pytest.approx(0.72, abs=0.003)
    assert result.log_source_evidence is None


def test_joint_lens_and_log_regularization_optimization():
    image_grid, source_grid, observation = make_problem()
    result = fit_pixelized_model(
        make_model(source_grid),
        observation,
        image_grid,
        objective="evidence",
        optimize_regularization_strength=True,
        log_strength_bounds=(-8.0, 15.0),
    )
    assert result.success
    assert result.parameter_names[-1] == "regularization.log_strength"
    assert result.regularization_strength > 0.0
    assert abs(result.stationarity_residual) < 5e-4
    assert float(result.parameter_vector[0]) == pytest.approx(0.72, abs=0.001)
    assert float(result.parameter_vector[1]) == pytest.approx(2.08, abs=0.012)


def test_invalid_objective_and_empty_free_parameter_set_are_rejected():
    image_grid, source_grid, observation = make_problem()
    with pytest.raises(ValueError, match="objective must"):
        fit_pixelized_model(
            make_model(source_grid), observation, image_grid, objective="invalid"
        )
    fixed = PixelizedLensModel(
        lens=(ProfileModel(EllipticalPowerLaw, einstein_radius=0.72, slope=2.08),),
        source_grid=source_grid,
    )
    with pytest.raises(ValueError, match="no free"):
        fit_pixelized_model(fixed, observation, image_grid)
