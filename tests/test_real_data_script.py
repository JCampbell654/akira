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
from examples.fit_real_data import run


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
        "output_directory": "outputs",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    assert run(config_path) == 0
    output = tmp_path / "outputs"
    assert (output / "fit_arrays.npz").is_file()
    assert (output / "fit_diagnostics.png").is_file()
    summary = json.loads((output / "fit_summary.json").read_text())
    assert summary["success"]
    assert summary["regularization_strength"] > 0
    assert "lens.0.EllipticalPowerLaw.einstein_radius" in summary["parameters"]
    with np.load(output / "fit_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["source"].shape == source_grid.shape
        assert arrays["model_image"].shape == image_grid.shape
