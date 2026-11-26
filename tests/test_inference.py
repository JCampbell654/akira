import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from akira import (
    CartesianGrid,
    CompositeLight,
    EllipticalPowerLaw,
    ExternalShear,
    Fixed,
    FreeParameter,
    GaussianPrior,
    InferenceProblem,
    LensModel,
    LensPlane,
    LogUniformPrior,
    ProfileModel,
    Sersic,
    TruncatedGaussianPrior,
    UniformPrior,
    fit_model,
    gaussian_log_likelihood,
    gaussian_psf,
    render,
    simulate_imaging,
)
from akira.imaging import convolve_psf


@pytest.mark.parametrize(
    "prior,integration_bounds",
    [
        (UniformPrior(-2.0, 3.0), (-2.0, 3.0)),
        (GaussianPrior(0.4, 1.7), (-12.0, 12.0)),
        (LogUniformPrior(0.01, 100.0), (0.01, 100.0)),
        (TruncatedGaussianPrior(0.2, 0.7, -0.5, 1.1), (-0.5, 1.1)),
    ],
)
def test_prior_is_normalized(prior, integration_bounds):
    integral, _ = quad(
        lambda value: np.exp(float(prior.log_prob(value))),
        *integration_bounds,
        epsabs=1e-10,
    )
    assert integral == pytest.approx(1.0, rel=2e-8, abs=2e-8)


@pytest.mark.parametrize(
    "prior",
    [
        UniformPrior(-2.0, 3.0),
        GaussianPrior(0.4, 1.7),
        LogUniformPrior(0.01, 100.0),
        TruncatedGaussianPrior(0.2, 0.7, -0.5, 1.1),
    ],
)
def test_unit_transform_is_monotonic_and_hits_median(prior):
    transformed = np.asarray(prior.from_unit(jnp.asarray([0.1, 0.5, 0.9])))
    assert np.all(np.diff(transformed) > 0.0)
    assert np.all(np.isfinite(transformed))


def test_log_uniform_has_inverse_value_density():
    prior = LogUniformPrior(1.0, 100.0)
    assert float(prior.log_prob(2.0) - prior.log_prob(4.0)) == pytest.approx(
        np.log(2.0)
    )
    assert float(prior.from_unit(0.5)) == pytest.approx(10.0)


def make_general_model():
    return LensModel(
        lens=(
            ProfileModel(
                EllipticalPowerLaw,
                einstein_radius=FreeParameter(0.86, UniformPrior(0.7, 1.2)),
                slope=FreeParameter(1.96, GaussianPrior(2.0, 0.3)),
                centre=(0.0, 0.0),
                q=0.76,
                phi=0.28,
            ),
            ProfileModel(ExternalShear, strength=0.035, phi=0.61),
        ),
        source=(
            ProfileModel(
                Sersic,
                intensity=0.12,
                effective_radius=0.16,
                sersic_index=1.2,
                centre=(
                    FreeParameter(0.03, UniformPrior(-0.2, 0.2)),
                    FreeParameter(-0.01, UniformPrior(-0.2, 0.2)),
                ),
                q=Fixed(0.68),
                phi=-0.22,
            ),
        ),
    )


def test_model_discovers_names_reconstructs_profiles_and_differentiates():
    model = make_general_model()
    assert model.parameter_names == (
        "lens.0.EllipticalPowerLaw.einstein_radius",
        "lens.0.EllipticalPowerLaw.slope",
        "source.0.Sersic.centre.x",
        "source.0.Sersic.centre.y",
    )
    lens, source = model.build(model.initial_vector)
    assert isinstance(lens.mass_profiles[0], EllipticalPowerLaw)
    assert isinstance(lens.mass_profiles[1], ExternalShear)
    assert isinstance(source.profiles[0], Sersic)
    assert source.profiles[0].q == pytest.approx(0.68)
    gradient = jax.grad(model.log_prior)(model.initial_vector)
    assert gradient.shape == (4,)
    assert np.all(np.isfinite(gradient))


def test_model_unit_cube_transform():
    model = make_general_model()
    values = np.asarray(model.from_unit(jnp.full(4, 0.5)))
    assert values == pytest.approx([0.95, 2.0, 0.0, 0.0])


def make_inference_problem():
    grid = CartesianGrid(shape=(24, 20), pixel_scale=0.05, supersampling=2)
    lens, source = make_general_model().build(make_general_model().initial_vector)
    noiseless = render(grid, lens, source)
    observation = simulate_imaging(
        noiseless,
        gaussian_psf(0.7, size=5),
        noise_sigma=0.01,
        pixel_scale=grid.pixel_scale,
        seed=9,
    )
    return InferenceProblem(make_general_model(), observation, grid)


def test_inference_problem_exposes_stable_parameter_contract():
    problem = make_inference_problem()
    assert problem.parameter_names == problem.model.parameter_names
    assert np.asarray(problem.initial_vector) == pytest.approx(
        np.asarray(problem.model.initial_vector)
    )
    assert problem.bounds == (
        (0.7, 1.2),
        (None, None),
        (-0.2, 0.2),
        (-0.2, 0.2),
    )
    assert np.asarray(problem.from_unit(jnp.full(4, 0.5))) == pytest.approx(
        [0.95, 2.0, 0.0, 0.0]
    )


def test_inference_problem_builds_predicts_and_evaluates_posterior():
    problem = make_inference_problem()
    values = problem.initial_vector
    lens, source = problem.build_model(values)
    expected_image = render(problem.grid, lens, source)
    expected_image = convolve_psf(expected_image, problem.observation.psf.kernel)
    predicted = problem.predict(values)
    expected_likelihood = gaussian_log_likelihood(
        problem.observation.data,
        expected_image,
        problem.observation.noise_map,
        problem.observation.mask,
    )
    assert np.asarray(predicted) == pytest.approx(np.asarray(expected_image))
    assert float(problem.log_likelihood(values)) == pytest.approx(
        float(expected_likelihood)
    )
    assert float(problem.log_prior(values)) == pytest.approx(
        float(problem.model.log_prior(values))
    )
    assert float(problem.log_posterior(values)) == pytest.approx(
        float(problem.log_likelihood(values) + problem.log_prior(values))
    )
    gradient = jax.grad(problem.log_posterior)(values)
    assert gradient.shape == values.shape
    assert np.all(np.isfinite(gradient))


def test_inference_problem_rejects_mismatched_observation_and_grid():
    problem = make_inference_problem()
    wrong_grid = CartesianGrid(shape=(10, 10), pixel_scale=0.05)
    with pytest.raises(ValueError, match="shapes must match"):
        InferenceProblem(problem.model, problem.observation, wrong_grid)


def test_general_fitter_recovers_epl_sersic_parameters():
    grid = CartesianGrid(shape=(72, 72), pixel_scale=0.05, supersampling=2)
    true_lens = EllipticalPowerLaw(0.92, slope=2.08, q=0.76, phi=0.28)
    shear = ExternalShear(0.035, phi=0.61)
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
    result = fit_model(make_general_model(), observation, grid, mode="maximum_likelihood")
    fitted = np.asarray(result.parameter_vector)
    truth = np.asarray([0.92, 2.08, 0.07, -0.045])
    assert result.success
    assert fitted == pytest.approx(truth, abs=0.006)
    assert result.model_image.shape == grid.shape
    assert result.covariance.shape == (4, 4)


def test_invalid_parameter_and_model_inputs_are_rejected():
    with pytest.raises(ValueError, match="prior support"):
        FreeParameter(-1.0, LogUniformPrior(0.1, 10.0))
    with pytest.raises(TypeError, match="Unknown"):
        ProfileModel(Sersic, intensity=1.0, effective_radius=0.2, sersic_index=2.0, typo=3)
    with pytest.raises(ValueError, match="wrong number"):
        make_general_model().from_unit(jnp.ones(2))
