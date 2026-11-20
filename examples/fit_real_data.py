#!/usr/bin/env python3
"""Fit real strong-lens imaging with a pixelized source.

The configuration controls input arrays, lens priors, source grid and output
location. Run ``python examples/fit_real_data.py --config my_config.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from akira import (
    CartesianGrid,
    EllipticalPowerLaw,
    ExternalShear,
    FreeParameter,
    GaussianPrior,
    ImagingObservation,
    LogUniformPrior,
    PSF,
    PixelizedLensModel,
    ProfileModel,
    SourceGrid,
    TruncatedGaussianPrior,
    UniformPrior,
    find_lensing_curves,
    fit_pixelized_model,
    plot_caustics,
)


def _load_array(spec: str | dict[str, Any], base: Path) -> np.ndarray:
    """Load a 2D array from FITS, NPY, NPZ, CSV or text."""

    if isinstance(spec, str):
        path_value, hdu, key = spec, 0, None
    else:
        path_value = spec["path"]
        hdu = spec.get("hdu", 0)
        key = spec.get("key")
    path = Path(path_value)
    if not path.is_absolute():
        path = base / path
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        try:
            from astropy.io import fits
        except ImportError as error:
            raise ImportError(
                "FITS input requires the real-data dependencies: "
                "python -m pip install -e '.[real-data]'"
            ) from error
        array = fits.getdata(path, ext=hdu)
    elif suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is None:
                if len(archive.files) != 1:
                    raise ValueError(f"{path} contains multiple arrays; specify 'key'.")
                key = archive.files[0]
            array = archive[key]
    elif suffix == ".csv":
        array = np.loadtxt(path, delimiter=",")
    else:
        array = np.loadtxt(path)
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{path} must contain a two-dimensional array.")
    return array


def _prior(spec: dict[str, Any]):
    kind = spec["type"].lower().replace("-", "_")
    if kind == "uniform":
        return UniformPrior(spec["lower"], spec["upper"])
    if kind == "gaussian":
        return GaussianPrior(spec["mean"], spec["sigma"])
    if kind == "log_uniform":
        return LogUniformPrior(spec["lower"], spec["upper"])
    if kind == "truncated_gaussian":
        return TruncatedGaussianPrior(
            spec["mean"], spec["sigma"], spec["lower"], spec["upper"]
        )
    raise ValueError(f"Unknown prior type: {spec['type']!r}.")


def _parameter(spec: float | dict[str, Any]) -> float | FreeParameter:
    if isinstance(spec, dict):
        return FreeParameter(float(spec["initial"]), _prior(spec["prior"]))
    return float(spec)


def _extent(shape: tuple[int, int], scale: float, centre: tuple[float, float]):
    half_x = shape[1] * scale / 2
    half_y = shape[0] * scale / 2
    return (
        centre[0] - half_x,
        centre[0] + half_x,
        centre[1] - half_y,
        centre[1] + half_y,
    )


def _make_model(config: dict[str, Any], source_grid: SourceGrid) -> PixelizedLensModel:
    lens_config = config["lens"]
    power_law = lens_config["power_law"]
    shear = lens_config["external_shear"]
    power_law_model = ProfileModel(
        EllipticalPowerLaw,
        einstein_radius=_parameter(power_law["einstein_radius"]),
        slope=_parameter(power_law["slope"]),
        centre=(
            _parameter(power_law.get("centre_x", 0.0)),
            _parameter(power_law.get("centre_y", 0.0)),
        ),
        q=_parameter(power_law.get("q", 1.0)),
        phi=_parameter(power_law.get("phi", 0.0)),
    )
    shear_model = ProfileModel(
        ExternalShear,
        strength=_parameter(shear.get("strength", 0.0)),
        phi=_parameter(shear.get("phi", 0.0)),
        centre=tuple(shear.get("centre", (0.0, 0.0))),
    )
    regularization = config["regularization"]
    return PixelizedLensModel(
        lens=(power_law_model, shear_model),
        source_grid=source_grid,
        regularization_kind=regularization["kind"],
        regularization_strength=float(regularization["initial_strength"]),
    )


def _report_memory_scale(
    observation: ImagingObservation,
    source_grid: SourceGrid,
    supersampling: int,
) -> None:
    mask = np.asarray(observation.mask)
    active = ~mask
    footprint = convolve2d(
        active.astype(np.uint8),
        np.ones(observation.psf.kernel.shape, dtype=np.uint8),
        mode="same",
    ) > 0
    active_pixels = int(active.sum())
    support_pixels = int(footprint.sum())
    bytes_per_matrix = active_pixels * source_grid.size * 8
    support_bytes = support_pixels * source_grid.size * supersampling**2 * 8
    print(
        f"Using {active_pixels:,} of {mask.size:,} image pixels; "
        f"PSF support covers {support_pixels:,} pixels."
    )
    print(
        "Approximate base dense storage: "
        f"operator {bytes_per_matrix / 2**20:.1f} MiB, "
        f"ray-tracing workspace {support_bytes / 2**20:.1f} MiB."
    )
    if bytes_per_matrix + support_bytes > 6 * 2**30:
        raise MemoryError(
            "This masked configuration still needs more than 6 GiB for its two "
            "base matrices before differentiation. Reduce the source-grid size, "
            "supersampling, PSF footprint, or number of unmasked pixels."
        )


def _save_results(
    output_directory: Path,
    config: dict[str, Any],
    observation: ImagingObservation,
    source_grid: SourceGrid,
    image_grid: CartesianGrid,
    result,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    reconstruction = result.reconstruction
    covariance = (
        np.empty((0, 0), dtype=np.float64)
        if result.covariance is None
        else np.asarray(result.covariance)
    )
    np.savez_compressed(
        output_directory / "fit_arrays.npz",
        data=np.asarray(observation.data),
        noise_map=np.asarray(observation.noise_map),
        mask=np.asarray(observation.mask),
        model_image=np.asarray(reconstruction.model_image),
        normalized_residuals=np.asarray(reconstruction.residuals),
        source=np.asarray(reconstruction.source),
        source_posterior_std=np.sqrt(
            np.diag(np.asarray(reconstruction.posterior_covariance))
        ).reshape(source_grid.shape),
        nonlinear_covariance=covariance,
    )
    summary = {
        "success": result.success,
        "message": result.message,
        "evaluations": result.evaluations,
        "objective": result.objective,
        "objective_value": result.objective_value,
        "log_likelihood": result.log_likelihood,
        "log_prior": result.log_prior,
        "log_source_evidence": result.log_source_evidence,
        "chi_squared": reconstruction.chi_squared,
        "unmasked_pixels": int(np.sum(~np.asarray(observation.mask))),
        "regularization_kind": reconstruction.regularization_kind,
        "regularization_strength": result.regularization_strength,
        "effective_source_parameters": result.effective_source_parameters,
        "stationarity_residual": result.stationarity_residual,
        "parameters": result.parameters,
    }
    (output_directory / "fit_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    caustic_config = config.get("caustic_grid", {})
    caustic_grid = CartesianGrid(
        tuple(caustic_config.get("shape", (301, 301))),
        float(caustic_config.get("pixel_scale", image_grid.pixel_scale / 5)),
        tuple(caustic_config.get("centre", image_grid.centre)),
    )
    curves = find_lensing_curves(result.lens, caustic_grid)
    image_extent = _extent(image_grid.shape, image_grid.pixel_scale, image_grid.centre)
    source_extent = _extent(source_grid.shape, source_grid.pixel_scale, source_grid.centre)
    masked_residuals = np.asarray(reconstruction.residuals).copy()
    masked_residuals[np.asarray(observation.mask)] = np.nan
    source_uncertainty = np.sqrt(
        np.diag(np.asarray(reconstruction.posterior_covariance))
    ).reshape(source_grid.shape)

    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    panels = (
        (observation.data, "data", "magma", image_extent),
        (reconstruction.model_image, "best-fitting model", "magma", image_extent),
        (masked_residuals, "normalized residual", "coolwarm", image_extent),
        (reconstruction.source, "reconstructed source", "magma", source_extent),
        (source_uncertainty, "source posterior uncertainty", "viridis", source_extent),
        (observation.mask, "mask (white = excluded)", "gray", image_extent),
    )
    for axis, (values, title, colour_map, extent) in zip(axes.flat, panels):
        shown = axis.imshow(values, origin="lower", extent=extent, cmap=colour_map)
        axis.set_title(title)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    plot_caustics(axes[1, 0], curves, color="cyan", linewidth=1.5)
    for axis in axes[0]:
        axis.set_xlabel("image x [arcsec]")
    axes[0, 0].set_ylabel("image y [arcsec]")
    axes[1, 0].set_xlabel("source x [arcsec]")
    axes[1, 0].set_ylabel("source y [arcsec]")
    axes[1, 1].set_xlabel("source x [arcsec]")
    axes[1, 2].set_xlabel("image x [arcsec]")
    figure.savefig(output_directory / "fit_diagnostics.png", dpi=180)
    plt.close(figure)


def run(config_path: Path) -> int:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    base = config_path.parent
    data = _load_array(config["data"], base).astype(np.float64)
    noise_map = _load_array(config["noise_map"], base).astype(np.float64)
    psf_array = _load_array(config["psf"], base).astype(np.float64)
    psf_sum = psf_array.sum()
    if not np.isfinite(psf_sum) or psf_sum <= 0:
        raise ValueError("PSF must have a positive finite sum.")
    psf = PSF(psf_array / psf_sum)
    mask = None
    if config.get("mask") is not None:
        mask = _load_array(config["mask"], base).astype(bool)
    pixel_scale = float(config["pixel_scale"])
    observation = ImagingObservation(data, noise_map, psf, pixel_scale, mask)
    image_grid = CartesianGrid(
        data.shape, pixel_scale, supersampling=int(config.get("supersampling", 1))
    )
    source_config = config["source_grid"]
    source_grid = SourceGrid(
        tuple(source_config["shape"]),
        float(source_config["pixel_scale"]),
        tuple(source_config.get("centre", (0.0, 0.0))),
    )
    model = _make_model(config, source_grid)
    _report_memory_scale(observation, source_grid, image_grid.supersampling)
    strength_bounds = config["regularization"]["strength_bounds"]
    if strength_bounds[0] <= 0 or strength_bounds[0] >= strength_bounds[1]:
        raise ValueError("regularization strength_bounds must be positive and ordered.")
    result = fit_pixelized_model(
        model,
        observation,
        image_grid,
        objective="evidence",
        optimize_regularization_strength=True,
        log_strength_bounds=tuple(math.log(value) for value in strength_bounds),
    )
    output_directory = Path(config.get("output_directory", "fit_output"))
    if not output_directory.is_absolute():
        output_directory = base / output_directory
    _save_results(
        output_directory, config, observation, source_grid, image_grid, result
    )
    print(f"Fit success: {result.success} ({result.message})")
    print(f"Regularization strength: {result.regularization_strength:.6g}")
    for name, value in result.parameters.items():
        print(f"{name}: {value:.8g}")
    print(f"Results written to: {output_directory}")
    return 0 if result.success else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return run(parser.parse_args().config)


if __name__ == "__main__":
    raise SystemExit(main())
