import json

import numpy as np
import pytest

from akira import (
    CartesianGrid,
    CompositeLight,
    FreeParameter,
    Gaussian,
    InferenceProblem,
    LensModel,
    LensPlane,
    ProfileModel,
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    SIS,
    UniformPrior,
    fit_model,
    gaussian_psf,
    load_inference_result,
    render,
    save_inference_result,
    simulate_imaging,
)


def make_fitted_problem():
    grid = CartesianGrid((18, 18), pixel_scale=0.08)
    lens = SIS(0.72)
    source = Gaussian(0.8, 0.14, centre=(0.04, -0.03))
    image = render(grid, LensPlane((lens,)), CompositeLight((source,)))
    observation = simulate_imaging(
        image,
        gaussian_psf(0.6, size=5),
        noise_sigma=0.01,
        pixel_scale=grid.pixel_scale,
        seed=12,
    )
    model = LensModel(
        lens=(
            ProfileModel(
                SIS,
                einstein_radius=FreeParameter(0.68, UniformPrior(0.5, 0.9)),
            ),
        ),
        source=(
            ProfileModel(
                Gaussian,
                intensity=0.8,
                sigma=0.14,
                centre=(0.04, -0.03),
            ),
        ),
    )
    problem = InferenceProblem(model, observation, grid)
    result = fit_model(model, observation, grid, mode="map")
    return problem, result


def test_result_format_round_trip(tmp_path):
    problem, result = make_fitted_problem()
    summary_path, arrays_path = save_inference_result(tmp_path, problem, result)

    assert summary_path.name == "result_summary.json"
    assert arrays_path.name == "result_arrays.npz"
    stored = load_inference_result(tmp_path)
    summary = stored.summary
    assert summary["format"] == RESULT_FORMAT
    assert summary["format_version"] == RESULT_FORMAT_VERSION
    assert summary["provenance"]["objective"] == "map"
    assert summary["problem"]["grid"] == {
        "shape": [18, 18],
        "pixel_scale": 0.08,
        "supersampling": 1,
    }
    assert summary["problem"]["model"]["lens"][0]["type"] == "SIS"
    parameter = summary["problem"]["parameters"][0]
    assert parameter["name"] == "lens.0.SIS.einstein_radius"
    assert parameter["prior"] == {"type": "uniform", "lower": 0.5, "upper": 0.9}
    assert summary["point_estimate"]["parameter_names"] == list(
        result.parameter_names
    )
    assert summary["optimization"]["success"]
    assert summary["sampling"] is None
    assert stored.arrays["parameter_vector"] == pytest.approx(
        np.asarray(result.parameter_vector)
    )
    assert stored.arrays["model_image"] == pytest.approx(
        np.asarray(result.model_image)
    )
    assert stored.arrays["residual_map"] == pytest.approx(
        np.asarray(problem.observation.data - result.model_image)
    )


def test_loader_rejects_unknown_format_version(tmp_path):
    problem, result = make_fitted_problem()
    summary_path, _ = save_inference_result(tmp_path, problem, result)
    summary = json.loads(summary_path.read_text())
    summary["format_version"] = 999
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="Unsupported result format version"):
        load_inference_result(tmp_path)


def test_saver_rejects_result_from_a_different_problem(tmp_path):
    problem, result = make_fitted_problem()
    incompatible = LensModel(
        lens=(ProfileModel(SIS, einstein_radius=0.72),),
        source=(
            ProfileModel(
                Gaussian,
                intensity=FreeParameter(0.8, UniformPrior(0.5, 1.0)),
                sigma=0.14,
            ),
        ),
    )
    other_problem = InferenceProblem(incompatible, problem.observation, problem.grid)
    with pytest.raises(ValueError, match="parameter names"):
        save_inference_result(tmp_path, other_problem, result)
