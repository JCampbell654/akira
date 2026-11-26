import json

import numpy as np
import pytest

from akira import (
    PIXELIZED_RESULT_FORMAT,
    PixelizedInferenceProblem,
    fit_pixelized_model,
    load_pixelized_inference_result,
    save_pixelized_inference_result,
)
from tests.test_pixelized_inference import make_model, make_problem


def test_pixelized_result_round_trip_reconstructs_problem_and_resumes_sampling(
    tmp_path,
):
    image_grid, source_grid, observation = make_problem()
    model = make_model(source_grid)
    problem = PixelizedInferenceProblem(model, observation, image_grid)
    fit = fit_pixelized_model(model, observation, image_grid)

    summary_path, arrays_path = save_pixelized_inference_result(
        tmp_path, problem, fit
    )
    assert summary_path.name == "result_summary.json"
    assert arrays_path.name == "result_arrays.npz"

    stored = load_pixelized_inference_result(tmp_path)
    assert stored.summary["format"] == PIXELIZED_RESULT_FORMAT
    assert stored.problem.parameter_names == problem.parameter_names
    assert stored.parameter_names == fit.parameter_names
    assert stored.parameter_vector == pytest.approx(np.asarray(fit.parameter_vector))
    assert np.asarray(stored.problem.observation.data) == pytest.approx(
        np.asarray(observation.data)
    )
    assert stored.problem.image_grid == image_grid
    assert stored.problem.model.source_grid == source_grid

    sampling = stored.sample(
        n_walkers=6,
        n_steps=3,
        burn_in=1,
        seed=8,
    )
    assert sampling.parameter_names == fit.parameter_names
    assert sampling.chain.shape == (2, 6, 2)


def test_pixelized_result_remembers_optimized_log_regularization(tmp_path):
    image_grid, source_grid, observation = make_problem()
    model = make_model(source_grid)
    problem = PixelizedInferenceProblem(model, observation, image_grid)
    bounds = (-8.0, 15.0)
    fit = fit_pixelized_model(
        model,
        observation,
        image_grid,
        optimize_regularization_strength=True,
        log_strength_bounds=bounds,
    )
    save_pixelized_inference_result(tmp_path, problem, fit)

    stored = load_pixelized_inference_result(tmp_path)
    assert stored.optimized_regularization_strength
    assert stored.log_strength_bounds == bounds
    assert stored.parameter_names[-1] == "regularization.log_strength"
    sampling = stored.sample(
        n_walkers=8,
        n_steps=2,
        burn_in=1,
        seed=4,
    )
    assert sampling.parameter_names[-1] == "regularization.log_strength"
    assert sampling.chain.shape[-1] == 3


def test_pixelized_loader_rejects_unknown_version(tmp_path):
    image_grid, source_grid, observation = make_problem()
    model = make_model(source_grid)
    fit = fit_pixelized_model(model, observation, image_grid)
    summary_path, _ = save_pixelized_inference_result(
        tmp_path,
        PixelizedInferenceProblem(model, observation, image_grid),
        fit,
    )
    summary = json.loads(summary_path.read_text())
    summary["format_version"] = 999
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="Unsupported pixelized result format"):
        load_pixelized_inference_result(tmp_path)
