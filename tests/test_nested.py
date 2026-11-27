import sys
import types

import numpy as np
import pytest

from akira import sample_pixelized_model_nautilus
from tests.test_pixelized_inference import make_model, make_problem


class _FakeSampler:
    last_instance = None

    def __init__(self, prior, likelihood, n_dim, **kwargs):
        self.prior = prior
        self.likelihood = likelihood
        self.n_dim = n_dim
        self.kwargs = kwargs
        self.log_z = -123.5
        self.n_eff = 17.0
        self.points = None
        self.log_l = None
        _FakeSampler.last_instance = self

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        units = np.stack(
            [
                np.full(self.n_dim, 0.25),
                np.full(self.n_dim, 0.50),
                np.full(self.n_dim, 0.75),
            ]
        )
        self.points = np.stack([self.prior(unit) for unit in units])
        self.log_l = np.asarray([self.likelihood(point) for point in self.points])
        return True

    def posterior(self, return_as_dict=False, equal_weight=False):
        assert return_as_dict is False
        assert equal_weight is False
        return self.points, np.log(np.asarray([0.2, 0.5, 0.3])), self.log_l


@pytest.fixture
def fake_nautilus(monkeypatch):
    module = types.ModuleType("nautilus")
    module.Sampler = _FakeSampler
    monkeypatch.setitem(sys.modules, "nautilus", module)


def test_nautilus_samples_marginalized_likelihood_and_reconstructs_map(fake_nautilus):
    image_grid, source_grid, observation = make_problem()
    model = make_model(source_grid)
    result = sample_pixelized_model_nautilus(
        model,
        observation,
        image_grid,
        n_live=100,
        n_eff=20,
        seed=4,
    )

    assert result.parameter_names == model.parameter_names
    assert result.points.shape == (3, 2)
    assert np.isclose(result.weights.sum(), 1.0)
    assert np.all(np.isfinite(result.log_likelihood))
    assert result.log_evidence == pytest.approx(-123.5)
    assert result.effective_sample_size == pytest.approx(17.0)
    assert result.evaluations == 3
    assert result.success
    assert result.point_estimate.reconstruction.source.shape == source_grid.shape
    assert result.point_estimate.log_source_evidence == pytest.approx(
        result.log_likelihood[result.maximum_posterior_index]
    )
    assert _FakeSampler.last_instance.kwargs["pass_dict"] is False


def test_nautilus_can_sample_log_regularization_strength(fake_nautilus):
    image_grid, source_grid, observation = make_problem()
    result = sample_pixelized_model_nautilus(
        make_model(source_grid),
        observation,
        image_grid,
        sample_regularization_strength=True,
        log_strength_bounds=(np.log(1.0), np.log(100.0)),
        n_live=100,
        n_eff=20,
    )

    assert result.parameter_names[-1] == "regularization.log_strength"
    assert result.points.shape == (3, 3)
    assert 1.0 <= result.regularization_strength <= 100.0
    assert result.point_estimate.optimized_regularization_strength


def test_nautilus_validates_log_strength_bounds(fake_nautilus):
    image_grid, source_grid, observation = make_problem()
    with pytest.raises(ValueError, match="finite and ordered"):
        sample_pixelized_model_nautilus(
            make_model(source_grid),
            observation,
            image_grid,
            sample_regularization_strength=True,
            log_strength_bounds=(1.0, 1.0),
        )
