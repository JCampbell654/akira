import numpy as np
import pytest

from akira import sample_pixelized_model
from tests.test_pixelized_inference import make_model, make_problem


def test_ensemble_sampler_returns_finite_source_marginalized_chain():
    image_grid, source_grid, observation = make_problem()
    model = make_model(source_grid)
    result = sample_pixelized_model(
        model,
        observation,
        image_grid,
        n_walkers=6,
        n_steps=8,
        burn_in=2,
        thin=2,
        seed=12,
    )
    assert result.parameter_names == model.parameter_names
    assert result.chain.shape == (3, 6, 2)
    assert result.log_probability.shape == (3, 6)
    assert result.samples.shape == (18, 2)
    assert np.all(np.isfinite(result.log_probability))
    assert np.all((result.acceptance_fraction >= 0.0) & (result.acceptance_fraction <= 1.0))


def test_sampler_can_include_log_regularization_strength():
    image_grid, source_grid, observation = make_problem()
    model = make_model(source_grid)
    result = sample_pixelized_model(
        model,
        observation,
        image_grid,
        sample_regularization_strength=True,
        log_strength_bounds=(np.log(1.0), np.log(1000.0)),
        n_walkers=8,
        n_steps=5,
        burn_in=1,
        seed=3,
    )
    assert result.parameter_names[-1] == "regularization.log_strength"
    assert result.chain.shape == (4, 8, 3)


def test_sampler_rejects_too_few_walkers():
    image_grid, source_grid, observation = make_problem()
    with pytest.raises(ValueError, match="at least twice"):
        sample_pixelized_model(
            make_model(source_grid),
            observation,
            image_grid,
            n_walkers=2,
            n_steps=4,
            burn_in=1,
        )
