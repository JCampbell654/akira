import json

import numpy as np

from akira import (
    CartesianGrid,
    EllipticalPowerLaw,
    LensPlane,
    SourceGrid,
    gaussian_psf,
    lensing_operator,
    simulate_imaging,
)
from examples.fit_real_data import _make_model, run
from examples.reconstruct_fixed_lens import run as run_fixed
from examples.fit_real_data_staged import STAGES, _build_stage_model, _starting_values


def test_real_data_script_runs_fit_and_writes_outputs(tmp_path) -> None:
    image_grid = CartesianGrid((14, 14), 0.12)
    source_grid = SourceGrid((4, 4), 0.12)
    true_lens = LensPlane((EllipticalPowerLaw(0.58, 2.0, q=0.82, phi=0.2),))
    source = np.exp(
        -np.sum(np.asarray(source_grid.coordinates()) ** 2, axis=-1) / (2 * 0.1**2)
    )
    unlensed = (
        lensing_operator(image_grid, source_grid, true_lens) @ source.reshape(-1)
    ).reshape(image_grid.shape)
    psf = gaussian_psf(0.6, 3)
    observation = simulate_imaging(
        unlensed, psf, noise_sigma=0.015, pixel_scale=0.12, seed=4
    )
    np.save(tmp_path / "image.npy", np.asarray(observation.data))
    np.save(tmp_path / "noise.npy", np.asarray(observation.noise_map))
    np.save(tmp_path / "psf.npy", np.asarray(psf.kernel))
    np.save(tmp_path / "mask.npy", np.zeros(image_grid.shape, dtype=bool))
    config = {
        "data": "image.npy",
        "noise_map": "noise.npy",
        "psf": "psf.npy",
        "mask": "mask.npy",
        "pixel_scale": 0.12,
        "source_grid": {"shape": [4, 4], "pixel_scale": 0.12},
        "lens": {
            "power_law": {
                "einstein_radius": {
                    "initial": 0.55,
                    "prior": {"type": "uniform", "lower": 0.45, "upper": 0.7},
                },
                "slope": 2.0,
                "q": 0.82,
                "phi": 0.2,
            },
            "external_shear": {"strength": 0.0, "phi": 0.0},
        },
        "regularization": {
            "kind": "zeroth",
            "initial_strength": 10.0,
            "strength_bounds": [0.01, 1000.0],
        },
        "caustic_grid": {"shape": [101, 101], "pixel_scale": 0.02},
        "sampling": {
            "enabled": True,
            "walkers": 4,
            "steps": 4,
            "burn_in": 1,
            "thin": 1,
            "seed": 9,
        },
        "output_directory": "outputs",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    assert run(config_path) == 0
    output = tmp_path / "outputs"
    assert (output / "fit_arrays.npz").is_file()
    assert (output / "fit_diagnostics.png").is_file()
    assert (output / "posterior_samples.npz").is_file()
    assert (output / "posterior_summary.json").is_file()
    assert (output / "posterior_corner.png").is_file()
    summary = json.loads((output / "fit_summary.json").read_text())
    assert summary["success"]
    assert summary["regularization_strength"] > 0
    assert "lens.0.EllipticalPowerLaw.einstein_radius" in summary["parameters"]
    with np.load(output / "fit_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["source"].shape == source_grid.shape
        assert arrays["model_image"].shape == image_grid.shape
    with np.load(output / "posterior_samples.npz", allow_pickle=False) as arrays:
        assert arrays["chain"].shape == (3, 4, 2)


def test_fixed_lens_script_reconstructs_without_optimization(tmp_path) -> None:
    image_grid = CartesianGrid((14, 14), 0.12)
    source_grid = SourceGrid((4, 4), 0.12)
    lens = LensPlane((EllipticalPowerLaw(0.58, 2.0, q=0.82, phi=0.2),))
    source = np.exp(
        -np.sum(np.asarray(source_grid.coordinates()) ** 2, axis=-1) / (2 * 0.1**2)
    )
    unlensed = (
        lensing_operator(image_grid, source_grid, lens) @ source.reshape(-1)
    ).reshape(image_grid.shape)
    psf = gaussian_psf(0.6, 3)
    observation = simulate_imaging(
        unlensed, psf, noise_sigma=0.015, pixel_scale=0.12, seed=4
    )
    for name, values in (
        ("image.npy", observation.data),
        ("noise.npy", observation.noise_map),
        ("psf.npy", psf.kernel),
        ("mask.npy", np.zeros(image_grid.shape, dtype=bool)),
    ):
        np.save(tmp_path / name, np.asarray(values))
    config = {
        "data": "image.npy",
        "noise_map": "noise.npy",
        "psf": "psf.npy",
        "mask": "mask.npy",
        "pixel_scale": 0.12,
        "source_grid": {"shape": [4, 4], "pixel_scale": 0.12},
        "lens": {
            "power_law": {
                "einstein_radius": 0.58,
                "slope": 2.0,
                "q": 0.82,
                "phi": 0.2,
            }
        },
        "regularization": {"kind": "zeroth", "strength": 10.0},
        "caustic_grid": {"shape": [101, 101], "pixel_scale": 0.02},
        "output_directory": "fixed_outputs",
    }
    config_path = tmp_path / "fixed.json"
    config_path.write_text(json.dumps(config))

    assert run_fixed(config_path) == 0
    output = tmp_path / "fixed_outputs"
    assert (output / "reconstruction_arrays.npz").is_file()
    assert (output / "reconstruction_diagnostics.png").is_file()
    summary = json.loads((output / "reconstruction_summary.json").read_text())
    assert summary["success"]
    assert summary["regularization_strength"] == 10.0
    assert summary["source_pixels"] == 16
    with np.load(output / "reconstruction_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["source"].shape == source_grid.shape
        assert arrays["model_image"].shape == image_grid.shape


def test_real_data_model_accepts_fixed_regularization_strength() -> None:
    config = {
        "lens": {
            "power_law": {"einstein_radius": 1.0, "slope": 2.0},
            "external_shear": {"strength": 0.0, "phi": 0.0},
        },
        "regularization": {
            "kind": "gradient",
            "strength": 2.0e9,
            "optimize_strength": False,
        },
    }
    model = _make_model(config, SourceGrid((4, 4), 0.1))
    assert model.regularization_strength == 2.0e9


def test_staged_models_progressively_release_parameters() -> None:
    free = lambda initial, lower, upper: {
        "initial": initial,
        "prior": {"type": "uniform", "lower": lower, "upper": upper},
    }
    config = {
        "lens": {
            "power_law": {
                "einstein_radius": free(1.0, 0.5, 1.5),
                "slope": free(2.0, 1.5, 2.5),
                "centre_x": free(0.0, -0.2, 0.2),
                "centre_y": free(0.0, -0.2, 0.2),
                "q": free(0.8, 0.3, 1.0),
                "phi": free(0.1, -1.5, 1.5),
            },
            "external_shear": {
                "strength": free(0.03, 0.0, 0.2),
                "phi": free(0.2, -1.5, 1.5),
            },
        },
        "regularization": {"kind": "curvature"},
    }
    source_grid = SourceGrid((4, 4), 0.1)
    values = _starting_values(config)
    models = [
        _build_stage_model(config, source_grid, values, stage_free, 10.0)
        for _, stage_free in STAGES
    ]

    assert [len(model.parameter_names) for model in models] == [0, 2, 6, 8]
    assert models[1].parameter_names == (
        "lens.0.EllipticalPowerLaw.einstein_radius",
        "lens.0.EllipticalPowerLaw.slope",
    )
    assert models[-1].parameter_names[-2:] == (
        "lens.1.ExternalShear.strength",
        "lens.1.ExternalShear.phi",
    )
